#ifndef ERG_CON_INCLUDE_GUARD_HPP
#define ERG_CON_INCLUDE_GUARD_HPP
/// \file
/// \brief 


#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <ergodiclib/ergodic_measure.hpp>
#include <ergodiclib/num_utils.hpp>
#include <ergodiclib/fourier_basis.hpp>
#include <ergodiclib/model.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

namespace ergodiclib
{
    template <class ModelTemplate>
    class ergController
    {
        public:
            ergController(ErgodicMeasure ergodicMes, fourierBasis basis, ModelTemplate model_agent, double q_val, arma::mat R_mat, arma::mat Q_mat, double t0_val, double tf_val, double dt_val, double eps_val, double beta_val) : 
            ergodicMeasure(ergodicMes), 
            Basis(basis),
            model(model_agent),
            q(q_val),
            R(R_mat),
            Q(Q_mat),
            t0(t0_val),
            tf(tf_val),
            dt(dt_val),
            eps(eps_val),
            beta(beta_val)
            {
                n_iter = (int) ((tf - t0)/ dt);
            };

            arma::mat calc_b(const arma::mat& u_mat);

            arma::mat calc_a(const arma::mat& x_mat);

            std::pair<std::vector<arma::mat>, std::vector<arma::mat>> calculatePr(arma::mat xt, arma::mat ut, const arma::mat& at, const arma::mat& bt);

            std::pair<arma::mat, arma::mat> descentDirection(arma::mat xt, arma::mat ut, std::vector<arma::mat> listP, std::vector<arma::mat> listr, arma::mat bt); 

            double DJ(std::pair<arma::mat, arma::mat> zeta_pair, const arma::mat& at, const arma::mat& bt);

            int gradient_descent(arma::vec x0);

        private:
            ErgodicMeasure ergodicMeasure;
            fourierBasis Basis;
            ModelTemplate model;
            double q;
            arma::mat R; 
            arma::mat Q;
            double t0;
            double tf;
            double dt;
            double eps;
            double beta;
            int n_iter;
    };
}


#endif
