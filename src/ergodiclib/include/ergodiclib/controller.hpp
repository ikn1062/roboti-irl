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

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

class ilqrController 
{
    public:
        ilqrController(Model model_in, arma::vec x0_in, arma::mat Q, arma::mat R, arma::mat P, arma::mat r, double dt_in, double t0_in, double tf_in, double a, double b, double e);

        void ilqrController::ILQR();

    private:
        double objectiveJ(arma::mat Xt, arma::mat Ut, arma::mat P1);

        double trajectoryJ(arma::mat Xt, arma::mat Ut);

        std::pair<arma::mat, arma::mat> calculateZeta(arma::mat Xt, arma::mat Ut);

        std::pair<std::vector<arma::mat>, std::vector<arma::mat>> ilqrController::calculatePr(arma::mat Xt, arma::mat Ut, arma::mat aT, arma::mat bT);

        arma::mat calculate_aT(arma::mat Xt);
        
        arma::mat calculate_bT(arma::mat Ut);


        Model model;
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

#endif