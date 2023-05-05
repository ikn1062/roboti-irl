#ifndef ERG_CON_INCLUDE_GUARD_HPP
#define ERG_CON_INCLUDE_GUARD_HPP
/// \file
/// \brief 


#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include </opt/homebrew/include/armadillo>
#include <ergodiclib/ergodic_measure.hpp>
#include <ergodiclib/ergodic_utils.hpp>

namespace ergodiclib
{
    class iLQRController
    {
        public:
            iLQRController();

            arma::mat calc_b(const arma::mat& u_mat, const arma::mat& R);

            arma::mat calc_a(const arma::mat& x_mat, const std::vector<std::vector<int> > K_series, const double q);

            arma::mat integrate(const arma::mat& x_mat, const arma::mat& u_mat);

            arma::mat dynamics(const arma::mat& x_mat, const arma::mat& u_mat);

            void calculatePr(const arma::mat& at, const arma::mat& bt);

            std::pair<arma::mat, arma::mat> descentDirection(arma::mat xt, arma::mat ut, std::vector<arma::mat> listP, std::vector<arma::mat> listr, arma::mat bt); 

            double DJ(std::pair<arma::mat, arma::mat> zeta_pair, const arma::mat& at, const arma::mat& bt);

            arma::mat make_trajec(arma::mat x0, arma::mat ut);

            int gradient_descent();

        private:

        
    };
}


#endif