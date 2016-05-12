#pragma once

#include <Eigen/Core>

#include <random>

namespace rosban_random
{

class MultiVariateGaussian
{
public:

  MultiVariateGaussian();
  MultiVariateGaussian(const Eigen::VectorXd & mu,
                       const Eigen::MatrixXd & sigma);

  /// Draw a sample using the given random engine
  Eigen::VectorXd getSample(std::default_random_engine & engine) const;

  void updateCholesky();

private:
  /// The mean of the distribution
  Eigen::VectorXd mu;

  /// The covariance matrix
  Eigen::MatrixXd sigma;

  /// The cholesky decomposition of the matrix
  Eigen::MatrixXd cholesky;
};

}
