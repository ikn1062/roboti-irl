#include <ergodiclib/cartpole.hpp>
#include <ergodiclib/controller.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

using namespace ergodiclib;

int main() 
{
    Model cartpole = CartPole();
    arma::vec x0({0.0, 0.0, PI, 0.0});
    arma::mat r;

    arma::mat Q(4, 4, arma::fill::eye);
    Q(2,2) = 10;
    Q(3,3) = 10;

    arma::mat R(1, 1, arma::fill::eye);
    R(0, 0) = 10;

    arma::mat P(4, 4, arma::fill::eye);

    arma::mat r(4, 1, arma::fill::zeros);

    double dt = 0.1;
    double t0 = 0.0;
    double tf = 15.0;
    double alpha = 0.40;
    double beta = 0.85;
    double eps = 0.01;

    

}