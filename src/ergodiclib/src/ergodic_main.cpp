#include <iostream>
#include <string>
#include <vector>
#include <ergodiclib/file_utils.hpp>
#include <ergodiclib/ergodic_measure.hpp>
#include <ergodiclib/ergodic_controller.hpp>
#include <ergodiclib/cartpole.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

using namespace ergodiclib;


int main()
{
  // NEED TO USE SMART POINTERS
  std::cout << "Running Ergodic Exploration and Controller..." << std::endl << std::endl;

  std::cout << "Getting Demonstrations... START" << std::endl;
  std::string file_demonstration_path = "../src/cartpole/demonstrations/";
  std::vector<arma::mat> demonstrations = readDemonstrations(file_demonstration_path, 4);

  std::cout << "Ergodic Basis... START" << std::endl;
  std::vector<int> demo_weights{1};

  std::pair<double, double> pair1(-5, 5);
  std::pair<double, double> pair2(-10, 10);
  std::pair<double, double> pair3(-PI, PI);
  std::pair<double, double> pair4(-10, 10);
  std::vector<std::pair<double, double>> lengths{pair1, pair2, pair3, pair4};
  fourierBasis Basis = fourierBasis(lengths, 4, 4);

  std::cout << "Ergodic Measurements... START" << std::endl;
  ErgodicMeasure ergodicMeasure = ErgodicMeasure(demonstrations, demo_weights, 0.005, Basis);
  ergodicMeasure.calcErgodic();

  // std::vector<double> hk = Basis.get_hK();
  // std::cout << "hk_values" << std::endl;
  // for (unsigned int i = 0; i < hk.size(); i++) {
  //   std::cout << hk[i] << std::endl;
  // }

  std::cout << "Cartpole...  START" << std::endl;
  arma::vec x0({0.0, 0.0, PI, 0.0});
  arma::vec u0({0.0});
  double dt = 0.005;
  double t0 = 0.0;
  double tf = 5.0;
  CartPole cartpole = CartPole(x0, u0, dt, t0, tf, 10.0, 5.0, 2.0);

  std::cout << "Controller...  START" << std::endl;
  double q = 100000.0;

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
  P(2, 2) = 100;
  P(3, 3) = 2;

  arma::mat r(4, 1, arma::fill::zeros);

  double alpha = 0.40;
  double beta = 0.85;
  double eps = 0.000001;

  ergController controller = ergController<CartPole>(ergodicMeasure, Basis, cartpole, q, Q, R, P, r, 500, alpha, beta, eps);
  controller.iLQR();
  return 0;
}
