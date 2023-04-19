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
    /// \brief 
    class ErgodicMeasure
    {
    public:
        /// \brief Constructor for Ergodic Measure Class
        /// \param demonstrations Vector of Demonstrations - Trajectories 
        /// \param demo_weights Demonstration weights - length of demonstrations
        /// \param K_coeff Size of Series Coefficient 
        /// \param L_dim Size of boundaries for dimensions 
        ErgodicMeasure(std::vector<std::vector<std::vector<double> > > demonstrations, std::vector<int> demo_weights, int K_coeff, std::vector<std::pair<double, double> > L_dim, double dt_demo);

        /// \brief Returns Spatial distribution of demonstrations for each series coefficient
        /// \return PhiK Vector 
        std::vector<double> get_PhiK();

        /// \brief Returns lambda_k which places larger weights on lower coefficients of information for each series coefficient
        /// \return LambdaK Vector 
        std::vector<double> get_LambdaK();

        /// \brief Returns Normalizing factor for Fk for each series coefficient
        /// \return hK vector
        std::vector<double> get_hK();

    private:
        /// \brief Calculates coefficients that describe the task definition, phi_k
        ///        phi_k is defined by the following:
        ///        phi_k = sum w_j * c_k_j where j ranges from 1 to num_trajectories
        ///        - w_j is initialized as 1/num_trajectories, the weighting of each trajectory in the spatial coefficient
        /// \return  PhiK Vector  
        std::vector<double> calculatePhik();

        /// \brief Calculate lambda_k places larger weights on lower coefficients of information
        ///        lambda_k is defined by the following:
        ///        lambda_k = (1 + ||k||2) − s where s = n+1/2
        /// \return  LambdaK Vector  
        std::vector<double> calculateLambdaK();

        /// \brief Calculates spacial statistics for a given trajectory and series coefficient value
        ///        ck is given by:
        ///        ck = integral Fk(x(t)) dt from t=0 to t=T
        ///        - where x(t) is a trajectory, mapping t to position vector x
        /// \param x_trajectory x(t) function, mapping position vectors over a period of time 
        /// \param K_vec The series coefficient given as a list of length dimensions 
        /// \param k_idx The index of the series coefficient K_vec
        /// \return CK value, Spacial Statistics
        double calculateCk(const std::vector<std::vector<double> >& x_trajectory, const std::vector<int>& K_vec, int k_idx);

        /// \brief Calculates normalized fourier coeffecient using basis function metric
        ///        Fk is defined by the following:
        ///        Fk = 1/hk * product(cos(k[i] *x[i])) where i ranges for all dimensions of x
        ///        - Where k[i] = (K[i] * pi) / L[i]
        ///        - Where L[i] is the bounds of the variable dimension i
        /// \param x_i_trajectory Position vector X at a given time t
        /// \param K_vec The series coefficient given as a list of length dimensions  
        /// \param k_idx The index of the series coefficient K_vec 
        /// \return Fk Value, normalized fourier coeffecient 
        double calculateFk(const std::vector<double>& x_i_trajectory, const std::vector<int>& K_vec, int k_idx);
        
        /// \brief Normalizing factor for Fk
        ///        hk is defined as:
        ///        hk = Integral cos^2(k[i] * x[i]) dx from L[i][0] to L[i][1]
        /// \param K_vec The series coefficient given as a list of length dimensions  
        /// \param k_idx The index of the series coefficient K_vec 
        /// \return hk, normalizing factor for Fk
        double calculateHk(const std::vector<int>& K_vec, int k_idx);

        /// \brief Vector of Demonstrations - Each Demonstration is a trajectory of n-dimensions
        std::vector<std::vector<std::vector<double> > > D_mat;

        /// \brief Vector of weights for each Demonstration
        std::vector<int> E_vec;

        /// \brief Size of series coefficient
        int K;

        /// \brief Size of boundaries for dimensions - [Lower boundary, Higher Boundary]
        std::vector<std::pair<double, double> > L;

        /// \brief Time Difference 
        double dt;
        
        /// \brief Length of dimension of a given trajectory
        int n_dim;

        /// \brief Number of Trajectories
        int m_demo;

        /// \brief Vector of Weights for each demonstration
        std::vector<double> weight_vec;

        /// \brief Vector of Weights for each demonstration
        std::vector<std::vector<int> > K_series;

        /// \brief Vector of PhiK Values
        std::vector<double> PhiK_vec;
        
        /// \brief Vector of hK Values
        std::vector<double> hK_vec;

        /// \brief Vector of lambdaK Values
        std::vector<double> lambdaK_vec;
    };
}

#endif
