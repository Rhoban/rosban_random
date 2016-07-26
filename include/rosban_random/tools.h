#pragma once

#include <Eigen/Core>

#include <random>

namespace rosban_random
{

std::default_random_engine getRandomEngine();
std::default_random_engine * newRandomEngine();

/// Return a vector of random_engines initialized with different seeds, if
/// engine is provided, then it is used to generate the random engines,
/// otherwise a new random_engine is created.
/// This function is particularly important to work with stochastic functions in
/// parallel
std::vector<std::default_random_engine> getRandomEngines(int nb_engines,
                                                         std::default_random_engine * engine = NULL);

/// Create its own engine if no engine is provided
std::vector<size_t> getKDistinctFromN(size_t k, size_t n,
                                      std::default_random_engine * engine = NULL);

/// Create its own engine if no engine is provided
std::vector<double> getUniformSamples(double min,
                                      double max,
                                      size_t nb_samples,
                                      std::default_random_engine * engine = NULL);

/// Create its own engine if no engine is provided
std::vector<Eigen::VectorXd> getUniformSamples(const Eigen::MatrixXd& limits,
                                               size_t nb_samples,
                                               std::default_random_engine * engine = NULL);

/// Each column of the result matrix is a different Sample
Eigen::MatrixXd getUniformSamplesMatrix(const Eigen::MatrixXd& limits,
                                        size_t nb_samples,
                                        std::default_random_engine * engine = NULL);


}
