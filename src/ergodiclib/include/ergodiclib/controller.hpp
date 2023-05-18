#ifndef CONTROLLER_INCLUDE_GUARD_HPP
#define CONTROLLER_INCLUDE_GUARD_HPP
/// \file
/// \brief 

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <ergodiclib/num_utils.hpp>
#include <ergodiclib/model.hpp>
#include <ergodiclib/cartpole.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

namespace ergodiclib
{
template <class ModelTemplate>
class ilqrController 
{
    public:
        ilqrController(ModelTemplate model_in, arma::vec x0_in, arma::mat Q, arma::mat R, arma::mat P, arma::mat r, double dt_in, double t0_in, double tf_in, double a, double b, double e) :
        model(model_in),
        x0(x0_in),
        Q_mat(Q),
        R_mat(R),
        P_mat(P),
        r_mat(r),
        dt(dt_in),
        t0(t0_in),
        tf(tf_in),
        alpha(a),
        beta(b),
        eps(e)
        {
            num_iter = (int) ((tf - t0) / dt);
        };

        void iLQR();

    private:
        double objectiveJ(arma::mat Xt, arma::mat Ut, arma::mat P1);

        double trajectoryJ(arma::mat Xt, arma::mat Ut);

        std::pair<arma::mat, arma::mat> calculateZeta(arma::mat Xt, arma::mat Ut);

        std::pair<std::vector<arma::mat>, std::vector<arma::mat>> calculatePr(arma::mat Xt, arma::mat Ut, arma::mat aT, arma::mat bT);

        arma::mat calculate_aT(arma::mat Xt);
        
        arma::mat calculate_bT(arma::mat Ut);


        ModelTemplate model;
        arma::vec x0;
        arma::mat Q_mat;
        arma::mat R_mat;
        arma::mat P_mat;
        arma::mat r_mat;
        double dt;
        double t0;
        double tf;
        unsigned int num_iter;
        double alpha;
        double beta;
        double eps;

};
}

#endif