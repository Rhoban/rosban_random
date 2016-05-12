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

}
