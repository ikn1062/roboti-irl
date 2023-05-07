#ifndef ERG_CON_INCLUDE_GUARD_HPP
#define ERG_CON_INCLUDE_GUARD_HPP
/// \file
/// \brief 


#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include<armadillo>
#include <ergodiclib/ergodic_measure.hpp>
#include <ergodiclib/num_utils.hpp>
#include <ergodiclib/fourier_basis.hpp>
#include <ergodiclib/model.hpp>

namespace ergodiclib
{
    class iLQRController
    {
        public:
            iLQRController(ErgodicMeasure ergodicMes, fourierBasis basis, Model model_agent, double q_val, arma::mat R_mat, arma::mat Q_mat, double t0_val, double tf_val, double dt_val, double eps_val, double beta_val);

            arma::mat calc_b(const arma::mat& u_mat);

            arma::mat calc_a(const arma::mat& x_mat);

            std::pair<std::vector<arma::mat>, std::vector<arma::mat>> calculatePr(const arma::mat& at, const arma::mat& bt);

            std::pair<arma::mat, arma::mat> descentDirection(arma::mat xt, arma::mat ut, std::vector<arma::mat> listP, std::vector<arma::mat> listr, arma::mat bt); 

            double DJ(std::pair<arma::mat, arma::mat> zeta_pair, const arma::mat& at, const arma::mat& bt);

            int gradient_descent(arma::vec x0);

        private:
            ErgodicMeasure ergodicMeasure;
            fourierBasis Basis;
            Model model;
            const double q;
            const arma::mat R; 
            const arma::mat Q;
            double t0;
            double tf;
            double dt;
            int n_iter;
            double eps;
            double beta;
    };
}


#endif