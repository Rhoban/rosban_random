#include "rosban_random/multivariate_gaussian.h"

#include <Eigen/Cholesky>

namespace rosban_random
{

MultiVariateGaussian::MultiVariateGaussian()
{
}

MultiVariateGaussian::MultiVariateGaussian(const Eigen::VectorXd & mu_,
                                           const Eigen::MatrixXd & sigma_)
  : mu(mu_), sigma(sigma_)
{
  updateCholesky();
}

Eigen::VectorXd MultiVariateGaussian::getSample(std::default_random_engine & engine) const
{
  Eigen::VectorXd random_vector(mu.rows());
  std::normal_distribution<double> distribution(0,1);
  for (int dim = 0; dim < mu.rows(); dim++)
  {
    random_vector(dim) = distribution(engine);
  }
  return mu + cholesky * random_vector;
}

void MultiVariateGaussian::updateCholesky()
{
  //WARNING: according to Rasmussen 2006, it might be necessary to add epsilon * I
  //         before computing cholesky
  double epsilon = std::pow(10, -10);
  Eigen::MatrixXd I;
  I = Eigen::MatrixXd::Identity(sigma.rows(), sigma.rows());
  cholesky = Eigen::LLT<Eigen::MatrixXd>(sigma + epsilon * I).matrixL();
}

}
