#ifndef ERG_MES_INCLUDE_GUARD_HPP
#define ERG_MES_INCLUDE_GUARD_HPP
/// \file
/// \brief Calculates Ergodic Measurements from demonstrations


#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <ergodiclib/num_utils.hpp>
#include <ergodiclib/fourier_basis.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

namespace ergodiclib
{
/// \brief Class to calculate Spatial Ergodic Measurement Coefficients
class ErgodicMeasure
{
public:
  /// \brief Constructor for Ergodic Measure Class
  /// \param demonstrations Vector of Demonstrations - Trajectories
  /// \param demo_posneg Demonstration weights - length of demonstrations
  /// \param demo_weights Demonstration weights - length of demonstrations
  /// \param dt_demo time difference between each state in the trajectory
  /// \param dimensionLengths Length of Dimensions
  /// \param nDim Number of Dimensions
  /// \param K Fourier Series Coefficient
  ErgodicMeasure(
    std::vector<arma::mat> demonstrations,
    std::vector<int> demo_posneg, std::vector<double> demo_weights,
    double dt_demo, std::vector<std::pair<double, double>> dimensionLengths, int nDim, int K);

  ~ErgodicMeasure() 
  {
    delete Basis;
  }

  /// \brief Returns Spatial distribution of demonstrations for each series coefficient
  /// \return PhiK Vector
  arma::vec const & get_PhiK() const;

  /// \brief Returns lambda_k which places larger weights on lower coefficients of information for each series coefficient
  /// \return LambdaK Vector
  arma::vec const & get_LambdaK() const;

  /// \brief Calculates spacial statistics for a given trajectory and series coefficient value
  ///        ck is given by:
  ///        ck = integral Fk(x(t)) dt from t=0 to t=T
  ///        - where x(t) is a trajectory, mapping t to position vector x
  /// \param x_trajectory x(t) function, mapping position vectors over a period of time
  /// \param k_idx The index of the series coefficient K_vec
  /// \return CK value, Spacial Statistics
  double calculateCk(const arma::mat & x_trajectory, const int k_idx);

  /// \brief Calculates the spatial ergodic metric Phik and weight coefficient lambdaK
  void calcErgodic();

  /// \brief Provides an interface call to Fourier Basis Functions
  /// \param xTrajectory Position vector X at a given time t
  /// \param Kidx The index of the series coefficient K_vec
  /// \return Row vector for direction derivative of normalized state vector
  arma::rowvec calculateFourierDFk(const arma::colvec & xTrajectory, const int Kidx) const;

  /// \brief Size of K Series Coefficients
  unsigned int sizeK;

private:
  /// \brief Calculates coefficients that describe the task definition, phi_k
  ///        phi_k is defined by the following:
  ///        phi_k = sum w_j * c_k_j where j ranges from 1 to num_trajectories
  ///        - w_j is initialized as 1/num_trajectories, the weighting of each trajectory in the spatial coefficient
  /// \return None
  void calculatePhik();

  /// \brief Calculate lambda_k places larger weights on lower frequency information
  ///        lambda_k is defined by the following:
  ///        lambda_k = (1 + ||k||2) − s where s = n+1/2
  /// \return None
  void calculateLambdaK();

  /// \brief Vector of Demonstrations - Each Demonstration is a trajectory of n-dimensions
  const std::vector<arma::mat> D_mat;

  /// \brief Class that contains the fourier basis for the space
  const fourierBasis* Basis; // We can probbably change this to be a unique ptr, and then use a default constructor

  /// \brief Vector of weights for each Demonstration
  const std::vector<int> E_vec;

  /// \brief Time Difference
  const double dt;

  /// \brief Length of dimension of a given trajectory
  const int n_dim;

  /// \brief Number of Trajectories
  const int m_demo;

  /// \brief Vector of Weights for each demonstration
  const std::vector<double> weight_vec;

  /// \brief Vector of Weights for each demonstration
  const std::vector<std::vector<int>> & K_series;

  /// \brief Vector of PhiK Values
  arma::vec PhiK_vec;

  /// \brief Vector of lambdaK Values
  arma::vec lambdaK_vec;
};
}

#endif
