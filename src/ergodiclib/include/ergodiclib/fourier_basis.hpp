#ifndef BASIS_INCLUDE_GUARD_HPP
#define BASIS_INCLUDE_GUARD_HPP
/// \file
/// \brief Contains functions to calculate fourier basis

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <ergodiclib/num_utils.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

namespace ergodiclib
{
/// \brief Class containing methods for calculating the fourier basis of a trajectory signal
class fourierBasis
{
public:
  /// \brief Constructor for Fourier Basis class
  /// \param dimensionLengths Length of Dimensions
  /// \param nDim Number of Dimensions
  /// \param K Fourier Series Coefficient
  fourierBasis(std::vector<std::pair<double, double>> dimensionLengths, int nDim, int K);

  /// \brief Destructor for Fourier Basis class
  ~fourierBasis() = default;

  /// \brief Gets the fourier dimension series
  /// \return Fourier Dimension Series K vector of vector of int
  std::vector<std::vector<int>> const &  get_K_series() const;

  /// \brief Calculates normalized fourier coeffecient using basis function metric
  ///        Fk is defined by the following:
  ///        Fk = 1/hk * product(cos(k[i] *x[i])) where i ranges for all dimensions of x
  ///        - Where k[i] = (K[i] * pi) / L[i]
  ///        - Where L[i] is the bounds of the variable dimension i
  /// \param xTrajectory Position vector X at a given time t
  /// \param Kidx The index of the series coefficient K_vec
  /// \return Fk Value, normalized fourier coeffecient
  double calculateFk(
    const arma::vec & xTrajectory, const int Kidx) const;

  /// \brief Calculated Directional Derivative of DFk relative to state vector
  /// \param xTrajectory Position vector X at a given time t
  /// \param Kidx The index of the series coefficient K_vec
  /// \return Row vector for direction derivative of normalized state vector
  arma::rowvec calculateDFk(const arma::colvec & xTrajectory, const int Kidx) const;

private:
  /// \brief Normalizing factor for Fk
  ///        hk is defined as:
  ///        hk = Integral cos^2(k[i] * x[i]) dx from L[i][0] to L[i][1]
  /// \return hk, normalizing factor for Fk
  void calculateHk();

  /// \brief Creates the fourier series coefficients
  /// \param K Size of series coefficients
  /// \param n_dim Size of dimension for demonstrations
  /// \return None
  void create_K_series(const int K);

  /// \brief Length of dimension of a given trajectory
  const unsigned int _nDim;

  /// \brief Size of boundaries for dimensions - [Lower boundary, Higher Boundary]
  const std::vector<std::pair<double, double>> _lengthDims;

  /// \brief Vector of hK Values
  std::vector<double> _hkVec;

  /// \brief Vector of Weights for each demonstration
  std::vector<std::vector<int>> _kSeries;
};
}

#endif
