/// This file was imported from Quentin Rouxel PhD thesis repository
/// with some style modifications
#pragma once

#include <vector>
#include <random>
#include <Eigen/Core>

namespace rosban_random {

/// This class implement a Multivariate Gaussian distribution. It provides
/// access to measures such as the density of probability at a given point
/// 
/// !!! Note that the current implementation for angles is not really the
/// !!! wrapped normal distribution. Probability and fitting methods are
/// !!! therefore quite wrong when the circular standard deviation is high
/// !!! (around pi/2 or above)
class MultivariateGaussian
{
public:

  /// Dummy empty initialization
  MultivariateGaussian();

  /// Initialization with mean vector and covariance matrix.  If optional
  /// isCircular is not empty, each non zero value means that associated
  /// dimension is an angle in radian.
  MultivariateGaussian(const Eigen::VectorXd& mean,
                       const Eigen::MatrixXd& covariance,
                       const Eigen::VectorXi& isCircular = Eigen::VectorXi());

  /// Return the gaussian 
  /// dimentionality
  size_t dimension() const;

  const Eigen::VectorXd& getMean() const;
  const Eigen::MatrixXd& getCovariance() const;

  /// Access to the vector specifiying for each dimension if it is circular
  const Eigen::VectorXi& getCircularity() const;

  /// Sample a point from the multivariate gaussian with given random engine
  Eigen::VectorXd getSample(std::default_random_engine * engine) const;

  /// Sample multiple points from the multivariate gaussian with given random engine
  /// Each column is a different point
  Eigen::MatrixXd getSamples(int nb_samples,
                             std::default_random_engine * engine) const;

  /// Return the density of probability at 'point' given the distribution
  /// parameters.
  /// This method is very likely to return extremely small numbers, therefore
  /// it is preferable to use getLogLikelihood
  double getLikelihood(const Eigen::VectorXd& point) const;

  /// Return the logarithm of the given point
  double getLogLikelihood(const Eigen::VectorXd& point) const;

  /// Compute the classic estimation of gaussian mean and covariance from given
  /// data vectors.
  /// If optional isCircular is not empty, each non zero value
  /// means that associated dimension is an angle in radian.
  void fit(const std::vector<Eigen::VectorXd>& points,
           const Eigen::VectorXi& isCircular = Eigen::VectorXi());

private:

  /// The mean vector
  Eigen::VectorXd mu;
  
  /// The covariance matrix (symetrix definite positive)
  Eigen::MatrixXd covar;
  
  /// Not null integer for each dimension where the represented value is an angle
  /// in radian in [-pi,pi].
  Eigen::VectorXi dims_circularity;

  /// Does the distribution has at least one circular dimension
  bool has_circular;
  
  /// The inverse of covariance matrix computed through cholesky decomposition
  Eigen::MatrixXd covar_inv;
  
  /// The left side of the Cholesky decomposition of the covariance matrix
  Eigen::MatrixXd cholesky;
  
  /// The determinant of the covariance matrix
  double determinant;
  
  /// Compute the covariance decomposition and update internal variables
  void computeDecomposition();
  
  /// Return the signed distance between
  /// given point and current mean.
  /// @throw logic_error if dimension of the point does not match dimension of
  ///        the distribution
  Eigen::VectorXd computeDistanceFromMean(const Eigen::VectorXd& point) const;
};

}
