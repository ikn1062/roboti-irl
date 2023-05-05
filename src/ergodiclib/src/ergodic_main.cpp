#include <iostream>
#include <string>
#include <vector>
#include </opt/homebrew/include/armadillo>

#include <ergodiclib/file_utils.hpp>
#include <ergodiclib/num_utils.hpp>
#include <ergodiclib/ergodic_measure.hpp>
#include <ergodiclib/controller.hpp>

using namespace ergodiclib;


int main()
{
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

  ErgodicMeasure ergodicMeasure = ErgodicMeasure(demonstrations, demo_weights, 4, 0.1, Basis);
  ergodicMeasure.calcErgodic();
  std::cout << "Ergodic Measurements...  COMPLETE" << std::endl << std::endl;

  std::cout << "Controller...  START" << std::endl;
  arma::mat R = 0.1 * arma::eye(1,1);
  arma::mat Q = arma::eye(4,4);
  iLQRController controller = iLQRController(ergodicMeasure, Basis, 100, R, Q, 0.0, 30.0, 0.1, 0.01, 0.15);
  std::cout << "Controller...  COMPLETE" << std::endl;
}
