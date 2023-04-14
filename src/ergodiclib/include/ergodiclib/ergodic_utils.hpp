#ifndef ERG_UTIL_INCLUDE_GUARD_HPP
#define ERG_UTIL_INCLUDE_GUARD_HPP
/// \file
/// \brief 

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace ergodiclib
{
    /// \brief Creates the fourier series coefficients
    /// \param K Size of series coefficients
    /// \param n_dim Size of dimension for demonstrations
    /// \return List of Fourier Series Coefficients
    std::vector<std::vector<int>> create_K_series(int K, int n_dim);

    /// \brief Reccursive helper for fourier series coefficients 
    /// \param K_num Size of series coefficients 
    /// \param permutation Current Permutation in sequence
    /// \param n_dim Size of dimension for demonstrations
    /// \param idx Current idx of sequence in permutation
    /// \return List of Fourier Series Coefficients
    std::vector<std::vector<int>> create_K_helper(std::vector<int> K_num, std::vector<int> permutation, int n_dim, int idx);
    
    /// \brief Finds the integral of a trajectory using the trapezoidal rule, using constant dx
    /// \param y_trajec Trajectory to perform integration over
    /// \param dx difference in x for each step of the trajectory 
    /// \return Value of integral
    double integralTrapz(std::vector<double> y_trajec, double dx);
    
    /// \brief Gets the L2 norm of a given vector
    /// \param v Vector of integers
    /// \return L2 Norm Value
    double l2_norm(const std::vector<int>& v);

    /// \brief Gets the L2 norm of a given vector
    /// \param v Vector of Double
    /// \return L2 Norm Value
    double l2_norm(const std::vector<double>& v); 
}

#endif