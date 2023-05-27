#ifndef ERG_MES_INCLUDE_GUARD_HPP
#define ERG_MES_INCLUDE_GUARD_HPP
/// \file
/// \brief Calculates Ergodic Measurements from demonstrations


#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <ergodiclib/num_utils.hpp>
#include <ergodiclib/fourier_basis.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

namespace ergodiclib
{
/// \brief
class ErgodicMeasure
{
public:
  /// \brief Constructor for Ergodic Measure Class
  /// \param demonstrations Vector of Demonstrations - Trajectories
  /// \param demo_weights Demonstration weights - length of demonstrations
  /// \param K_coeff Size of Series Coefficient
  /// \param L_dim Size of boundaries for dimensions
  /// \param dt_demo time difference between each state in the trajectory
  /// \param basis Fourier basis class for the demonstration
  ErgodicMeasure(
    std::vector<arma::mat> demonstrations,
    std::vector<int> demo_weights,
    double dt_demo, fourierBasis& basis);

  /// \brief Returns Spatial distribution of demonstrations for each series coefficient
  /// \return PhiK Vector
  arma::vec get_PhiK() const;

  /// \brief Returns lambda_k which places larger weights on lower coefficients of information for each series coefficient
  /// \return LambdaK Vector
  arma::vec get_LambdaK() const;

  /// \brief Calculates spacial statistics for a given trajectory and series coefficient value
  ///        ck is given by:
  ///        ck = integral Fk(x(t)) dt from t=0 to t=T
  ///        - where x(t) is a trajectory, mapping t to position vector x
  /// \param x_trajectory x(t) function, mapping position vectors over a period of time
  /// \param K_vec The series coefficient given as a list of length dimensions
  /// \param k_idx The index of the series coefficient K_vec
  /// \return CK value, Spacial Statistics
  double calculateCk(
    const arma::mat & x_trajectory,
    const std::vector<int> & K_vec, int k_idx);

  void calcErgodic();

private:
  /// \brief Calculates coefficients that describe the task definition, phi_k
  ///        phi_k is defined by the following:
  ///        phi_k = sum w_j * c_k_j where j ranges from 1 to num_trajectories
  ///        - w_j is initialized as 1/num_trajectories, the weighting of each trajectory in the spatial coefficient
  /// \return  PhiK Vector
  arma::vec calculatePhik();

  /// \brief Calculate lambda_k places larger weights on lower coefficients of information
  ///        lambda_k is defined by the following:
  ///        lambda_k = (1 + ||k||2) − s where s = n+1/2
  /// \return  LambdaK Vector
  arma::vec calculateLambdaK();

  /// \brief Vector of Demonstrations - Each Demonstration is a trajectory of n-dimensions
  std::vector<arma::mat> D_mat;

  /// Class that contains the fourier basis for the space
  fourierBasis& Basis;

  /// \brief Vector of weights for each Demonstration
  std::vector<int> E_vec;

  /// \brief Time Difference
  double dt;

  /// \brief Length of dimension of a given trajectory
  int n_dim;

  /// \brief Number of Trajectories
  int m_demo;

  /// \brief Vector of Weights for each demonstration
  std::vector<double> weight_vec;

  /// \brief Vector of Weights for each demonstration
  std::vector<std::vector<int>> K_series;

  /// \brief Vector of PhiK Values
  arma::vec PhiK_vec;

  /// \brief Vector of lambdaK Values
  arma::vec lambdaK_vec;
};
}

#endif
