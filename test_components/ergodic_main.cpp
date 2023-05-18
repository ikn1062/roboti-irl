#include <iostream>
#include <string>
#include <vector>
#include "file_utils.hpp"
#include "num_utils.hpp"
#include "ergodic_measure.hpp"
#include "ergodic_controller.hpp"
#include "cartpole.hpp"

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

  std::cout << "Getting Demonstrations...  START" << std::endl;
  std::string file_demonstration_path = "./src/cartpole/demonstrations/";
  std::vector<arma::mat> demonstrations = readDemonstrations(file_demonstration_path, 4);
  std::cout << "Getting Demonstrations...  COMPLETE" << std::endl << std::endl;

  std::cout << "Ergodic Measurements...  START" << std::endl;
  std::vector<arma::mat> input_demonstration{demonstrations[6]};
  std::vector<int> demo_weights{1};

  std::pair<double, double> pair1(-PI, PI);
  std::pair<double, double> pair2(-11, 11);
  std::pair<double, double> pair3(-15, 15);
  std::pair<double, double> pair4(-15, 15);
  std::vector<std::pair<double, double>> lengths{pair1, pair2, pair3, pair4};
  fourierBasis Basis = fourierBasis(lengths, 4, 4);

  ErgodicMeasure ergodicMeasure = ErgodicMeasure(demonstrations, demo_weights, 0.1, Basis);
  ergodicMeasure.calcErgodic();
  std::cout << "Ergodic Measurements...  COMPLETE" << std::endl << std::endl;

  std::cout << "Controller...  START" << std::endl;
  CartPole model = CartPole();
  arma::mat R = 0.1 * arma::eye(1,1);
  arma::mat Q = arma::eye(4,4);
  ergController a = ergController(ergodicMeasure, Basis, model, 100.0, R, Q, 0.0, 30.0, 0.1, 0.01, 0.15);
  ergController controller = ergController(ergodicMeasure, Basis, model, 100.0, R, Q, 0.0, 30.0, 0.1, 0.01, 0.15);
  std::cout << "Controller...  COMPLETE" << std::endl;
}