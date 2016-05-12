#pragma once

#include <Eigen/Core>

#include <random>

namespace rosban_random
{

std::default_random_engine getRandomEngine();

/// Create its own engine if no engine is provided
std::vector<size_t> getKDistinctFromN(size_t k, size_t n,
                                      std::default_random_engine * engine = NULL);

/// Create its own engine if no engine is provided
std::vector<Eigen::VectorXd> getUniformSamples(const Eigen::MatrixXd& limits,
                                               size_t nbSamples,
                                               std::default_random_engine * engine = NULL);

}
