#ifndef BASIS_INCLUDE_GUARD_HPP
#define BASIS_INCLUDE_GUARD_HPP
/// \file
/// \brief

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include </opt/homebrew/include/armadillo>

#include <ergodiclib/num_utils.hpp>

namespace ergodiclib
{
    class fourierBasis
    {
        public:
            fourierBasis(std::vector<std::pair<double, double>> L_dim, int num_dim, int K);

            /// \brief Calculates normalized fourier coeffecient using basis function metric
            ///        Fk is defined by the following:
            ///        Fk = 1/hk * product(cos(k[i] *x[i])) where i ranges for all dimensions of x
            ///        - Where k[i] = (K[i] * pi) / L[i]
            ///        - Where L[i] is the bounds of the variable dimension i
            /// \param x_i_trajectory Position vector X at a given time t
            /// \param K_vec The series coefficient given as a list of length dimensions
            /// \param k_idx The index of the series coefficient K_vec
            /// \return Fk Value, normalized fourier coeffecient
            double calculateFk(const arma::vec & x_i_trajectory, const std::vector<int> & K_vec, int k_idx);

            /// \brief WRITE COMMENT
            arma::rowvec calculateDFk(const arma::colvec& xi_vec, const std::vector<int>& K_vec);

            /// \brief Returns Normalizing factor for Fk for each series coefficient
            /// \return hK vector
            std::vector<double> get_hK();

            /// \brief 
            /// \return 
            std::vector<std::vector<int>> get_K_series(); 

        private:

            /// \brief Normalizing factor for Fk
            ///        hk is defined as:
            ///        hk = Integral cos^2(k[i] * x[i]) dx from L[i][0] to L[i][1]
            /// \param K_vec The series coefficient given as a list of length dimensions
            /// \param k_idx The index of the series coefficient K_vec
            /// \return hk, normalizing factor for Fk
            double calculateHk(const std::vector<int> & K_vec, int k_idx);

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
            std::vector<std::vector<int>> create_K_helper(
            std::vector<int> K_num, std::vector<int> permutation,
            int n_dim, int idx);
            
            /// \brief Length of dimension of a given trajectory
            int n_dim;

            /// \brief Size of boundaries for dimensions - [Lower boundary, Higher Boundary]
            std::vector<std::pair<double, double>> L;

            /// \brief Vector of hK Values
            std::vector<double> hK_vec;

            /// \brief Vector of Weights for each demonstration
            std::vector<std::vector<int>> K_series;
    };
}

#endif