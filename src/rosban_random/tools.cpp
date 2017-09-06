#include "rosban_random/tools.h"

#include <chrono>
#include <stdexcept>

namespace rosban_random
{

std::default_random_engine getRandomEngine()
{
  unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
  return std::default_random_engine(seed);
}

std::default_random_engine * newRandomEngine()
{
  unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
  return new std::default_random_engine(seed);
}

std::vector<std::default_random_engine> getRandomEngines(int nb_engines,
                                                         std::default_random_engine * engine)
{
  std::vector<std::default_random_engine> result(nb_engines);
  bool cleanAtEnd = false;
  if (engine == NULL) {
    cleanAtEnd = true;
    engine = newRandomEngine();
  }
  unsigned int min = std::numeric_limits<unsigned int>::lowest();
  unsigned int max = std::numeric_limits<unsigned int>::max();
  std::uniform_int_distribution<unsigned int> seed_distrib(min, max);
  for (int i = 0; i < nb_engines; i++)
  {
    result[i].seed(seed_distrib(*engine));
  }
  if (cleanAtEnd) {
    delete(engine);
  }
  return result;
}

std::vector<size_t> getKDistinctFromN(size_t k, size_t n,
                                      std::default_random_engine * engine)
{
  if (k > n)
    throw std::runtime_error("getKDistinctFromN: k is greater than n (forbidden)");
  bool cleanAtEnd = false;
  if (engine == NULL) {
    cleanAtEnd = true;
    engine = newRandomEngine();
  }
  // preparing structure
  std::vector<size_t> chosenIndex;
  std::vector<size_t> availableIndex;
  chosenIndex.reserve(k);
  availableIndex.reserve(n);
  for (size_t i = 0; i < n; i++) {
    availableIndex.push_back(i);
  }
  while (chosenIndex.size() < k) {
    int max = n - chosenIndex.size() - 1;
    std::uniform_int_distribution<size_t> distribution(0, max);
    int idx = distribution(*engine);
    chosenIndex.push_back(availableIndex[idx]);
    // availableIndex[max] will not be available at next iteration, therfore
    // we add it at the location where the index was taken
    availableIndex[idx] = availableIndex[max];
  }
  if (cleanAtEnd) {
    delete(engine);
  }
  return chosenIndex;
}

std::vector<double> getUniformSamples(double min,
                                      double max,
                                      size_t nb_samples,
                                      std::default_random_engine * engine)
{
  bool cleanAtEnd = false;
  if (engine == NULL) {
    cleanAtEnd = true;
    engine = newRandomEngine();
  }
  std::vector<double> result;
  result.reserve(nb_samples);
  std::uniform_real_distribution<double> distrib(min, max);
  for (size_t sId = 0; sId < nb_samples; sId++) {
    result.push_back(distrib(*engine));
  }
  if (cleanAtEnd)
    delete(engine);
  return result;

}

Eigen::VectorXd getUniformSample(const Eigen::MatrixXd& limits,
                                 std::default_random_engine * engine)
{
  return getUniformSamples(limits, 1, engine)[0];
}

std::vector<Eigen::VectorXd> getUniformSamples(const Eigen::MatrixXd& limits,
                                               size_t nb_samples,
                                               std::default_random_engine * engine)
{
  bool cleanAtEnd = false;
  if (engine == NULL) {
    cleanAtEnd = true;
    engine = newRandomEngine();
  }
  std::vector<Eigen::VectorXd> result;
  result.reserve(nb_samples);
  std::vector<std::uniform_real_distribution<double>> distribs;
  for (int i = 0; i < limits.rows(); i++)
  {
    std::uniform_real_distribution<double> d(limits(i,0), limits(i,1));
    distribs.push_back(d);
  }
  for (size_t sId = 0; sId < nb_samples; sId++) {
    Eigen::VectorXd sample(limits.rows());
    for (size_t dim = 0; dim < distribs.size(); dim++) {
      sample(dim) = distribs[dim](*engine);
    }
    result.push_back(sample);
  }
  if (cleanAtEnd)
    delete(engine);
  return result;
}

Eigen::MatrixXd getUniformSamplesMatrix(const Eigen::MatrixXd& limits,
                                        size_t nb_samples,
                                        std::default_random_engine * engine)
{
  // Get samples
  std::vector<Eigen::VectorXd> tmp = getUniformSamples(limits, nb_samples, engine);
  // Change layout
  Eigen::MatrixXd result(limits.rows(), nb_samples);
  for (int i = 0; i < nb_samples; i++)
  {
    result.col(i) = tmp[i];
  }
  return result;
}


std::vector<int> sampleWeightedIndices(const std::vector<double> & weights,
                                       int nb_samples,
                                       std::default_random_engine * engine)
{
  std::vector<double> summed_weights(weights.size());
  double sum = 0.0;
  for (size_t i=0; i < weights.size(); i++) {
    sum += weights[i];
    summed_weights[i] = sum;
  }

  if (sum <= 0.0) {
    std::ostringstream oss;
    oss << "sampleWeightedIndices: Negative or null sum for the weights: " << sum << std::endl;
    throw std::runtime_error(oss.str());
  }

  // Preparing the indices
  std::uniform_real_distribution<double> distrib(0.0, sum);
  std::vector<int> indices(nb_samples);
  for (unsigned int i = 0; i < nb_samples; i++) {
    double rand_val = distrib(*engine);
    // Finding index with a dichotomic method
    int start = 0;
    int end = weights.size();
    while (start != end){
      int pivot = (end + start) / 2;
      if (summed_weights[pivot] < rand_val){
        start = pivot + 1;
      }
      else{
        end = pivot;
      }
    }
    indices[i] = start;
  }
  return indices;
}

std::map<int,int> sampleWeightedIndicesMap(const std::vector<double> & weights,
                                           int nb_samples,
                                           std::default_random_engine * engine)
{
  std::map<int,int> indices_occurences;
  std::vector<int> indices = sampleWeightedIndices(weights, nb_samples, engine);
  for(int idx : indices)
  {
    if (indices_occurences.count(idx) == 0)
      indices_occurences[idx] = 1;
    else
      indices_occurences[idx] += 1;
  }
  return indices_occurences;
}

}
