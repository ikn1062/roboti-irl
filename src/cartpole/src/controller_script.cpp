#include <cartpole/cartpole_sys.hpp>
#include <ergodiclib/simple_controller.hpp>
#include <ergodiclib/file_utils.hpp>
#include <chrono>
using namespace std::chrono;

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
  Q(0, 0) = 1.0;
  Q(1, 1) = 1.0;
  Q(2, 2) = 200.0;
  Q(3, 3) = 10.0;

  arma::mat R(1, 1, arma::fill::eye);
  R(0, 0) = 0.01;

  arma::mat P(4, 4, arma::fill::eye);
  P(0, 0) = 0.0;
  P(1, 1) = 0.0;
  P(2, 2) = 1000.0;
  P(3, 3) = 50.0;

  arma::mat r(4, 1, arma::fill::zeros);

  double dt = 0.005;
  double t0 = 0.0;
  double tf = 3.0;
  double alpha = 0.40;
  double beta = 0.75;
  double eps = 0.0001;

  CartPole cartpole = CartPole(x0, u0, dt, t0, tf, 10.0, 5.0, 2.0);
  std::cout << "start controller" << std::endl;
  SimpleController controller = SimpleController(cartpole, Q, R, P, r, 1000, alpha, beta, eps);
  std::pair<arma::mat, arma::mat> trajectories = controller.iLQR();
  saveTrajectoryCSV("example", trajectories);
  return 0;
}


/*
int main()
{
  arma::mat X, U;

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

  double dt = 0.005;
  double t0 = 0.0;
  double tf = 10.0;
  double alpha = 0.40;
  double beta = 0.85;
  double eps = 1e-6;

  CartPole cartpole = CartPole(x0, u0, dt, t0, tf, 10.0, 5.0, 2.0);
  SimpleController controller = SimpleController(cartpole, Q, R, P, r, 425, alpha, beta, eps);

  auto start = high_resolution_clock::now();
  std::pair<arma::mat, arma::mat> trajectories = controller.ModelPredictiveControl(x0, u0, 200, 500);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);

  X = trajectories.first;
  U = trajectories.second;
  std::cout << "control time : " << duration.count() << std::endl;

  return 0;
}
*/

