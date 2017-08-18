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
  mu(),
  covar(),
  dims_circularity(),
  has_circular(false),
  cholesky(),
  determinant(0.0)
{
}

MultivariateGaussian::MultivariateGaussian(const Eigen::VectorXd& mean,
                                           const Eigen::MatrixXd& covariance,
                                           const Eigen::VectorXi& isCircular) :
  mu(mean),
  covar(covariance),
  dims_circularity(),
  has_circular(false),
  cholesky(),
  determinant(0.0)
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
    dims_circularity = isCircular;
    //Normalization
    for (size_t i=0;i<(size_t)dims_circularity.size();i++) {
      if (dims_circularity(i) != 0) {
        dims_circularity(i) = 1;
        mu(i) = AngleBound(mu(i));
        has_circular = true;
      }
    }
  } else {
    dims_circularity = Eigen::VectorXi::Zero(mean.size());
    has_circular = false;
  }

  //Cholesky decomposition of covariance matrix
  computeDecomposition();
}
        
size_t MultivariateGaussian::dimension() const
{
  return mu.size();
}

const Eigen::VectorXd& MultivariateGaussian::getMean() const
{
  return mu;
}
const Eigen::MatrixXd& MultivariateGaussian::getCovariance() const
{
  return covar;
}
        
const Eigen::VectorXi& MultivariateGaussian::getCircularity() const
{
  return dims_circularity;
}

Eigen::VectorXd
MultivariateGaussian::getSample(std::default_random_engine * engine) const
{
  //Draw normal unit vector
  Eigen::VectorXd unitRand(mu.size());
  std::normal_distribution<double> dist(0.0, 1.0);
  for (size_t i=0;i<(size_t)unitRand.size();i++) {
    unitRand(i) = dist(*engine);
  }
    
  //Compute the random generated point
  Eigen::VectorXd point = 
    mu + cholesky*unitRand;
  //Angle normalization
  if (has_circular) {
    for (size_t i=0;i<(size_t)dims_circularity.size();i++) {
      if (dims_circularity(i) != 0) {
        point(i) = AngleBound(point(i));
      }
    }
  }

  return point;
}

Eigen::MatrixXd
MultivariateGaussian::getSamples(int nb_samples,
                                 std::default_random_engine * engine) const
{
  Eigen::MatrixXd result(mu.rows(), nb_samples);
  for (int i = 0; i < nb_samples; i++) {
    result.col(i) = getSample(engine);
  }
  return result;
}

double MultivariateGaussian::getLikelihood(const Eigen::VectorXd& point) const
{
  size_t size = mu.size();
  //Compute distance from mean
  Eigen::VectorXd delta = computeDistanceFromMean(point);
  // throw error if point is out of range
  if (delta.size() == 0) {
    throw std::logic_error("MultivariateGaussian: point is out of range");
  }

  //Compute likelihood
  double tmp1 = -0.5*delta.transpose()*covar_inv*delta;
  double tmp2 = pow(2.0*M_PI, size)*determinant;
  return std::exp(tmp1)/std::sqrt(tmp2);
}

double MultivariateGaussian::getLogLikelihood(const Eigen::VectorXd& point) const
{
  size_t size = mu.size();
  //Compute distance from mean
  Eigen::VectorXd delta = computeDistanceFromMean(point);
  // throw error if point is out of range
  if (delta.size() == 0) {
    throw std::logic_error("MultivariateGaussian: point is out of range");
  }

  //Compute log likelihood
  double tmp1 = delta.transpose()*covar_inv*delta;
  return -0.5*(std::log(determinant) + tmp1 + (double)size*std::log(2.0*M_PI));
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
  has_circular = false;
  if ((size_t)isCircular.size() == size) {
    dims_circularity = isCircular;
    //Normalization
    for (size_t i=0;i<(size_t)dims_circularity.size();i++) {
      if (dims_circularity(i) != 0) {
        dims_circularity(i) = 1;
        has_circular = true;
      }
    }
  } else {
    dims_circularity = Eigen::VectorXi::Zero(size);
  }

  //Compute the mean estimation
  if (has_circular) {
    //If the dimension is circular,
    //the cartesian mean (sumX, sumY) 
    //is computed.
    //Else, only sumX is used.
    Eigen::VectorXd sumX = Eigen::VectorXd::Zero(size);
    Eigen::VectorXd sumY = Eigen::VectorXd::Zero(size);
    for (size_t i=0;i<data.size();i++) {
      for (size_t j=0;j<size;j++) {
        if (dims_circularity(j) == 0) {
          sumX(j) += data[i](j);
        } else {
          sumX(j) += std::cos(data[i](j));
          sumY(j) += std::sin(data[i](j));
        }
      }
    }
    mu = Eigen::VectorXd::Zero(size);
    for (size_t j=0;j<size;j++) {
      if (dims_circularity(j) == 0) {
        mu(j) = (1.0/(double)data.size())*sumX(j);
      } else {
        double meanX = (1.0/(double)data.size())*sumX(j);
        double meanY = (1.0/(double)data.size())*sumY(j);
        mu(j) = std::atan2(meanY, meanX);
      }
    }
  } else {
    //Standard non circular mean
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(size);
    for (size_t i=0;i<data.size();i++) {
      sum += data[i];
    }
    mu = (1.0/(double)data.size())*sum;
  }
    
  //Compute the covariance estimation
  Eigen::MatrixXd sum2 = Eigen::MatrixXd::Zero(size, size);
  for (size_t i=0;i<data.size();i++) {
    Eigen::VectorXd delta = computeDistanceFromMean(data[i]);
    sum2 += delta*delta.transpose();
  }
  covar = (1.0/(double)(data.size()-1))*sum2;

  //Update the Cholesky decomposition
  computeDecomposition();
}
        
void MultivariateGaussian::computeDecomposition()
{
  //Add small epsilon on diagonal according 
  //to Rasmussen 2006
  double epsilon = 1e-9;
  size_t size = mu.size();
  Eigen::MatrixXd noise = epsilon * Eigen::MatrixXd::Identity(size, size);
  Eigen::LLT<Eigen::MatrixXd> llt(covar + noise);
  cholesky = llt.matrixL();
  //Check the decomposition
  if(llt.info() == Eigen::NumericalIssue) {
    throw std::logic_error("MultivariateGaussian Cholesky decomposition error");
  }
  //Compute the covariance determinant
  determinant = 1.0;
  for (size_t i=0;i<(size_t)mu.size();i++) {
    determinant *= cholesky(i, i);
  }
  determinant = pow(determinant, 2);
  //Compute the covariance inverse
  covar_inv = llt.solve(Eigen::MatrixXd::Identity(size, size));
}

Eigen::VectorXd
MultivariateGaussian::computeDistanceFromMean(const Eigen::VectorXd& point) const
{
  size_t size = mu.size();
  if ((size_t)point.size() != size) {
    throw std::logic_error("MultivariateGaussian: invalid dimension");
  }

  Eigen::VectorXd delta(size);
  if (has_circular) {
    for (size_t i=0;i<size;i++) {
      if (dims_circularity(i) == 0) {
        //No circular distance
        delta(i) = point(i) - mu(i);
      } else {
        double angle = point(i);
        //Outside of angle bounds
        angle = AngleBound(angle);
        //Compute circular distance
        delta(i) = AngleDistance(mu(i), angle);
      }
    }
  } else {
    //No circular distance
    delta = point - mu;
  }

  return delta;
}

}

