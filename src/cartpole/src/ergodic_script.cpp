#include <iostream>
#include <string>
#include <vector>
#include <ergodiclib/file_utils.hpp>
#include <ergodiclib/ergodic_measure.hpp>
#include <ergodiclib/ergodic_controller.hpp>
#include <cartpole/cartpole_sys.hpp>

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
  std::string file_demonstration_path = "./src/cartpole/demonstrations/";
  std::vector<arma::mat> demonstrations = readDemonstrations(file_demonstration_path, 4);

  std::cout << "Ergodic Basis... START" << std::endl;
  std::vector<int> demo_posneg{1, 1, -1, -1, -1, -1};
  std::vector<double> demo_weights{0.6, 0.0, 0.1, 0.1, 0.1, 0.1};

  std::pair<double, double> pair1(-15, 15);
  std::pair<double, double> pair2(-15, 15);
  std::pair<double, double> pair3(-PI, PI);
  std::pair<double, double> pair4(-10, 10);
  std::vector<std::pair<double, double>> lengths{pair1, pair2, pair3, pair4};
  fourierBasis Basis = fourierBasis(lengths, 4, 4);
  std::vector<std::vector<int>> kseries = Basis.get_K_series();

  arma::colvec a({1.0, 2.0, 3.0, 4.0});
  Basis.calculateDFk(a, kseries[30], 30);

  Basis.calculateDFk(a,kseries[632], 632);

  // std::cout << "Ergodic Measurements... START" << std::endl;
  // ErgodicMeasure ergodicMeasure = ErgodicMeasure(
  //   demonstrations, demo_posneg, demo_weights, 0.001,
  //   Basis);
  // ergodicMeasure.calcErgodic();

  // // std::vector<double> hk = Basis.get_hK();
  // // std::cout << "hk_values" << std::endl;
  // // for (unsigned int i = 0; i < hk.size(); i++) {
  // //   std::cout << hk[i] << std::endl;
  // // }

  // std::cout << "Cartpole...  START" << std::endl;
  // arma::vec x0({0.0, 0.0, PI, 0.0});
  // arma::vec u0({0.0});
  // double dt = 0.005;
  // double t0 = 0.0;
  // double tf = 5.0;
  // CartPole cartpole = CartPole(x0, u0, dt, t0, tf, 10.0, 5.0, 2.0);

  // std::cout << "Controller...  START" << std::endl;
  // double q = 1e6;

  // arma::mat Q(4, 4, arma::fill::eye);
  // // Q(0, 0) = 0.01;
  // // Q(1, 1) = 0.01;
  // // Q(2, 2) = 20.0;
  // // Q(3, 3) = 1.0;
  // Q(0, 0) = 0.1;
  // Q(1, 1) = 0.01;
  // Q(2, 2) = 10.0;
  // Q(3, 3) = 1.0;

  // arma::mat R(1, 1, arma::fill::eye);
  // R(0, 0) = 0.001;

  // arma::mat P(4, 4, arma::fill::eye);
  // // P(0, 0) = 0.0001;
  // // P(1, 1) = 0.0001;
  // // P(2, 2) = 300;
  // // P(3, 3) = 2;
  // P(0, 0) = 0.001;
  // P(1, 1) = 0.001;
  // P(2, 2) = 10.0;
  // P(3, 3) = 1.0;

  // arma::mat r(4, 1, arma::fill::zeros);

  // double alpha = 0.40;
  // double beta = 0.75;
  // double eps = 0.005;

  // ergController controller = ergController<CartPole>(
  //   ergodicMeasure, Basis, cartpole, q, Q, R, P, r,
  //   500, alpha, beta, eps);
  // controller.iLQR();
  return 0;
}
