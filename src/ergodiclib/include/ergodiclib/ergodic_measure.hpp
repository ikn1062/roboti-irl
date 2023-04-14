#ifndef ERG_MES_INCLUDE_GUARD_HPP
#define ERG_MES_INCLUDE_GUARD_HPP
/// \file
/// \brief 


#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include </opt/homebrew/include/eigen3/Eigen/Dense>
#include <ergodiclib/ergodic_utils.hpp>

// #include <Eigen/Dense>

namespace ergodiclib
{
    constexpr double PI = 3.14159265358979323846;

    /// \brief 
    class ErgodicMeasure
    {
    public:
        /// \brief Constructor for Ergodic Measure Class
        /// \param demonstrations Vector of Demonstrations - Trajectories 
        /// \param demo_weights Demonstration weights - length of demonstrations
        /// \param K_coeff Size of Series Coefficient 
        /// \param L_dim Size of boundaries for dimensions 
        ErgodicMeasure(std::vector<std::vector<std::vector<double>>> demonstrations, std::vector<int> demo_weights, int K_coeff, std::vector<std::pair<double, double>> L_dim, double dt_demo);

        std::vector<double> get_PhiK();

        std::vector<double> get_LambdaK();

        std::vector<double> get_hK();

    private:
        std::vector<double> calculatePhik();

        std::vector<double> calculateLambdaK();

        double calculateCk(const std::vector<std::vector<double>>& x_trajectory, const std::vector<int>& K_vec, int k_idx);

        double calculateFk(const std::vector<double>& x_i_trajectory, const std::vector<int>& K_vec, int k_idx);

        double calculateHk(const std::vector<int>& K_vec, int k_idx);

        /// \brief Vector of Demonstrations - Each Demonstration is a trajectory of n-dimensions
        std::vector<std::vector<std::vector<double>>> D_mat;

        /// \brief Vector of weights for each Demonstration
        std::vector<int> E_vec;

        /// \brief Size of series coefficient
        int K;

        /// \brief Size of boundaries for dimensions - [Lower boundary, Higher Boundary]
        std::vector<std::pair<double, double>> L;

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

        std::vector<double> PhiK_vec;

        std::vector<double> hK_vec;

        std::vector<double> lambdaK_vec;
    };
}

#endif
