#include "rosban_random/multivariate_gaussian.h"

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>

namespace rosban_random
{

/// Return the angle corresponding in [-pi, pi]
static double AngleBound(double angle)
{
  return angle - 2.0*M_PI*std::floor((angle + M_PI)/(2.0*M_PI));
}

/// Compute the oriented distance between the two given angle
/// in the range -PI/2:PI/2 radian from angleSrc to angleDst
/// (Better than doing angleDst-angleSrc)
static double AngleDistance(double angleSrc, double angleDst) 
{    
  angleSrc = AngleBound(angleSrc);
  angleDst = AngleBound(angleDst);

  double max, min;
  if (angleSrc > angleDst) {
    max = angleSrc;
    min = angleDst;
  } else {
    max = angleDst;
    min = angleSrc;
  }

  double dist1 = max-min;
  double dist2 = 2.0*M_PI - max + min;
 
  if (dist1 < dist2) {
    if (angleSrc > angleDst) {
      return -dist1;
    } else {
      return dist1;
    }
  } else {
    if (angleSrc > angleDst) {
      return dist2;
    } else {
      return -dist2;
    }
  }
}


MultivariateGaussian::MultivariateGaussian() :
  _mean(),
  _covariance(),
  _isCircular(),
  _hasCircular(false),
  _choleskyDecomposition(),
  _determinant(0.0)
{
}

MultivariateGaussian::MultivariateGaussian(const Eigen::VectorXd& mean,
                                           const Eigen::MatrixXd& covariance,
                                           const Eigen::VectorXi& isCircular) :
  _mean(mean),
  _covariance(covariance),
  _isCircular(),
  _hasCircular(false),
  _choleskyDecomposition(),
  _determinant(0.0)
{
  //Check size
  if (mean.size() != covariance.rows() ||
      mean.size() != covariance.cols()) {
    throw std::logic_error("MultivariateGaussian invalid input size");
  }
  if (isCircular.size() != 0 &&
      isCircular.size() != mean.size()) {
    throw std::logic_error("MultivariateGaussian invalid circular size");
  }

  //Circular initialization
  if (isCircular.size() == mean.size()) {
    _isCircular = isCircular;
    //Normalization
    for (size_t i=0;i<(size_t)_isCircular.size();i++) {
      if (_isCircular(i) != 0) {
        _isCircular(i) = 1;
        _mean(i) = AngleBound(_mean(i));
        _hasCircular = true;
      }
    }
  } else {
    _isCircular = Eigen::VectorXi::Zero(mean.size());
    _hasCircular = false;
  }

  //Cholesky decomposition of covariance matrix
  computeDecomposition();
}
        
size_t MultivariateGaussian::dimension() const
{
  return _mean.size();
}

const Eigen::VectorXd& MultivariateGaussian::getMean() const
{
  return _mean;
}
const Eigen::MatrixXd& MultivariateGaussian::getCovariance() const
{
  return _covariance;
}
        
const Eigen::VectorXi& MultivariateGaussian::getCircularity() const
{
  return _isCircular;
}

Eigen::VectorXd
MultivariateGaussian::sample(std::default_random_engine * engine) const
{
  //Draw normal unit vector
  Eigen::VectorXd unitRand(_mean.size());
  std::normal_distribution<double> dist(0.0, 1.0);
  for (size_t i=0;i<(size_t)unitRand.size();i++) {
    unitRand(i) = dist(*engine);
  }
    
  //Compute the random generated point
  Eigen::VectorXd point = 
    _mean + _choleskyDecomposition*unitRand;
  //Angle normalization
  if (_hasCircular) {
    for (size_t i=0;i<(size_t)_isCircular.size();i++) {
      if (_isCircular(i) != 0) {
        point(i) = AngleBound(point(i));
      }
    }
  }

  return point;
}

double MultivariateGaussian::getLikelihood(const Eigen::VectorXd& point) const
{
  size_t size = _mean.size();
  //Compute distance from mean
  Eigen::VectorXd delta = computeDistanceFromMean(point);
  // throw error if point is out of range
  if (delta.size() == 0) {
    throw std::logic_error("MultivariateGaussian: point is out of range");
  }

  //Compute likelihood
  double tmp1 = -0.5*delta.transpose()*_covarianceInv*delta;
  double tmp2 = pow(2.0*M_PI, size)*_determinant;
  return std::exp(tmp1)/std::sqrt(tmp2);
}

double MultivariateGaussian::getLogLikelihood(const Eigen::VectorXd& point) const
{
  size_t size = _mean.size();
  //Compute distance from mean
  Eigen::VectorXd delta = computeDistanceFromMean(point);
  // throw error if point is out of range
  if (delta.size() == 0) {
    throw std::logic_error("MultivariateGaussian: point is out of range");
  }

  //Compute log likelihood
  double tmp1 = delta.transpose()*_covarianceInv*delta;
  return -0.5*(std::log(_determinant) + tmp1 + (double)size*std::log(2.0*M_PI));
}

void MultivariateGaussian::fit(const std::vector<Eigen::VectorXd>& data,
                               const Eigen::VectorXi& isCircular)
{
  //Check sizes
  if (data.size() < 2) {
    throw std::logic_error("MultivariateGaussian not enough data points");
  }
  size_t size = data.front().size();
  for (size_t i=0;i<data.size();i++) {
    if ((size_t)data[i].size() != size) {
      throw std::logic_error("MultivariateGaussian invalid data dimension");
    }
  }
  if (isCircular.size() != 0 && (size_t)isCircular.size() != size) {
    throw std::logic_error("MultivariateGaussian invalid circular dimension");
  } 
    
  //Circular initialization
  _hasCircular = false;
  if ((size_t)isCircular.size() == size) {
    _isCircular = isCircular;
    //Normalization
    for (size_t i=0;i<(size_t)_isCircular.size();i++) {
      if (_isCircular(i) != 0) {
        _isCircular(i) = 1;
        _hasCircular = true;
      }
    }
  } else {
    _isCircular = Eigen::VectorXi::Zero(size);
  }

  //Compute the mean estimation
  if (_hasCircular) {
    //If the dimension is circular,
    //the cartesian mean (sumX, sumY) 
    //is computed.
    //Else, only sumX is used.
    Eigen::VectorXd sumX = Eigen::VectorXd::Zero(size);
    Eigen::VectorXd sumY = Eigen::VectorXd::Zero(size);
    for (size_t i=0;i<data.size();i++) {
      for (size_t j=0;j<size;j++) {
        if (_isCircular(j) == 0) {
          sumX(j) += data[i](j);
        } else {
          sumX(j) += std::cos(data[i](j));
          sumY(j) += std::sin(data[i](j));
        }
      }
    }
    _mean = Eigen::VectorXd::Zero(size);
    for (size_t j=0;j<size;j++) {
      if (_isCircular(j) == 0) {
        _mean(j) = (1.0/(double)data.size())*sumX(j);
      } else {
        double meanX = (1.0/(double)data.size())*sumX(j);
        double meanY = (1.0/(double)data.size())*sumY(j);
        _mean(j) = std::atan2(meanY, meanX);
      }
    }
  } else {
    //Standard non circular mean
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(size);
    for (size_t i=0;i<data.size();i++) {
      sum += data[i];
    }
    _mean = (1.0/(double)data.size())*sum;
  }
    
  //Compute the covariance estimation
  Eigen::MatrixXd sum2 = Eigen::MatrixXd::Zero(size, size);
  for (size_t i=0;i<data.size();i++) {
    Eigen::VectorXd delta = computeDistanceFromMean(data[i]);
    sum2 += delta*delta.transpose();
  }
  _covariance = (1.0/(double)(data.size()-1))*sum2;

  //Update the Cholesky decomposition
  computeDecomposition();
}
        
void MultivariateGaussian::computeDecomposition()
{
  //Add small epsilon on diagonal according 
  //to Rasmussen 2006
  double epsilon = 1e-9;
  size_t size = _mean.size();
  Eigen::MatrixXd noise = epsilon * Eigen::MatrixXd::Identity(size, size);
  Eigen::LLT<Eigen::MatrixXd> llt(_covariance + noise);
  _choleskyDecomposition = llt.matrixL();
  //Check the decomposition
  if(llt.info() == Eigen::NumericalIssue) {
    throw std::logic_error("MultivariateGaussian Cholesky decomposition error");
  }
  //Compute the covariance determinant
  _determinant = 1.0;
  for (size_t i=0;i<(size_t)_mean.size();i++) {
    _determinant *= _choleskyDecomposition(i, i);
  }
  _determinant = pow(_determinant, 2);
  //Compute the covariance inverse
  _covarianceInv = llt.solve(Eigen::MatrixXd::Identity(size, size));
}

Eigen::VectorXd
MultivariateGaussian::computeDistanceFromMean(const Eigen::VectorXd& point) const
{
  size_t size = _mean.size();
  if ((size_t)point.size() != size) {
    throw std::logic_error(
      "MultivariateGaussian: invalid dimension");
  }

  Eigen::VectorXd delta(size);
  if (_hasCircular) {
    for (size_t i=0;i<size;i++) {
      if (_isCircular(i) == 0) {
        //No circular distance
        delta(i) = point(i) - _mean(i);
      } else {
        double angle = point(i);
        //Outside of angle bounds
        angle = AngleBound(angle);
        //Compute circular distance
        delta(i) = AngleDistance(_mean(i), angle);
      }
    }
  } else {
    //No circular distance
    delta = point - _mean;
  }

  return delta;
}

}

