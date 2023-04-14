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
    /// \brief 
    /// \param K 
    /// \param n_dim 
    /// \return 
    std::vector<std::vector<int>> create_K_series(int K, int n_dim);

    /// \brief 
    /// \param K_num 
    /// \param permutation 
    /// \param n_dim
    /// \param idx
    /// \return 
    std::vector<std::vector<int>> create_K_helper(std::vector<int> K_num, std::vector<int> permutation, int n_dim, int idx);

    double integralTrapz(std::vector<double> y_trajec, double dx);

    double l2_norm(const std::vector<int>& v);

    double l2_norm(const std::vector<double>& v); 
}

#endif