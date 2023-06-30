#include <ergodiclib/cartpole.hpp>
#include <ergodiclib/simple_controller.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

using namespace ergodiclib;

int main()
{
  arma::vec x0({0.0, 0.0, PI, 0.0});
  arma::vec u0({0.0});

  arma::mat Q(4, 4, arma::fill::eye);
  Q(0, 0) = 0.0;
  Q(1, 1) = 0.0;
  Q(2, 2) = 25.0;
  Q(3, 3) = 1.0;

  arma::mat R(1, 1, arma::fill::eye);
  R(0, 0) = 0.01;

  arma::mat P(4, 4, arma::fill::eye);
  P(0, 0) = 0.0001;
  P(1, 1) = 0.0001;
  P(2, 2) = 200;
  P(3, 3) = 2;

  arma::mat r(4, 1, arma::fill::zeros);

  double dt = 0.005;
  double t0 = 0.0;
  double tf = 5.0;
  double alpha = 0.40;
  double beta = 0.85;
  double eps = 2.0;

  CartPole cartpole = CartPole(x0, u0, dt, t0, tf, 10.0, 5.0, 2.0);
  SimpleController controller = SimpleController(cartpole, Q, R, P, r, 1500, alpha, beta, eps);
  controller.iLQR();

  return 0;
}

/*
int main()
{
  arma::vec x0({0.0, 0.0, PI, 0.0});
  arma::vec u0({0.0});

  arma::mat Q(4, 4, arma::fill::eye);
  Q(0, 0) = 0.0;
  Q(1, 1) = 0.0;
  Q(2, 2) = 25.0;
  Q(3, 3) = 1.0;

  arma::mat R(1, 1, arma::fill::eye);
  R(0, 0) = 1;

  arma::mat P(4, 4, arma::fill::eye);
  P(0, 0) = 0.01;
  P(1, 1) = 0.01;
  P(2, 2) = 100;
  P(3, 3) = 2;

  arma::mat r(4, 1, arma::fill::zeros);

  double dt = 0.001;
  double t0 = 0.0;
  double tf = 10.0;
  double alpha = 0.40;
  double beta = 0.85;
  double eps = 1e-6;

  CartPole cartpole = CartPole(x0, u0, dt, t0, tf, 10.0, 5.0, 2.0);
  ilqrController controller = ilqrController(cartpole, Q, R, P, r, 425, alpha, beta, eps);

  std::pair<arma::mat, arma::mat> trajectories = controller.ModelPredictiveControl(x0, u0, 1000, 500);

  return 0;
}
*/
