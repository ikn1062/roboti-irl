#include <iostream>
#include <string>
#include <vector>

#include <ergodiclib/file_utils.hpp>
#include <ergodiclib/ergodic_measure.hpp>

using namespace ergodiclib;

int main()
{
  std::cout << "Ergodic Main FIle" << std::endl << std::endl;

  std::cout << "Testing File Utility Functions...  START" << std::endl;
  std::vector<arma::mat> demos;
  std::string file_demonstration_path = "./src/cartpole/demonstrations/";

  demos = readDemonstrations(file_demonstration_path, 4);
  std::cout << "size of demos: " << demos.size() << std::endl;
  std::cout << "Testing File Utility Functions...  COMPLETE" << std::endl << std::endl;

  std::cout << "Testing Ergodic Utility Functions...  START" << std::endl;
  std::pair<double, double> pair1(-PI, PI);
  std::pair<double, double> pair2(-11, 11);
  std::pair<double, double> pair3(-15, 15);
  std::pair<double, double> pair4(-15, 15);
  std::vector<std::pair<double, double>> lengths{pair1, pair2, pair3, pair4};

  fourierBasis Basis = fourierBasis(lengths, 4, 4);
  std::vector<std::vector<int>> K_series = Basis.get_K_series();

  for (unsigned int i = 0; i < demos.size(); i++) {
    std::cout << "demo idx: " << i << ", demo size: " << demos[i].size() << std::endl;
  }

  std::cout << "Testing Ergodic Utility Functions...  COMPLETE" << std::endl << std::endl;

  std::cout << "Testing Ergodic Measure Functions...  START" << std::endl;
  std::vector<arma::mat> input_demonstration{demos[6]};
  std::vector<int> demo_posneg{1};
  std::vector<double> demo_weights{1.0};

  ErgodicMeasure erg = ErgodicMeasure(input_demonstration, demo_posneg, demo_weights, 0.1, Basis);
  erg.calcErgodic();
  arma::vec phik = erg.get_PhiK();
  arma::vec lambdaK = erg.get_LambdaK();
  arma::vec hK = Basis.get_hK();

  for (unsigned int i = 0; i < phik.size(); i++) {
    std::cout << "K: ";
    for (int j = 0; j < 4; j++) {
      std::cout << K_series[i][j];
    }
    std::cout << ", PhiK: " << std::setprecision(10) << phik(i) << std::endl;
  }

  for (unsigned int i = 0; i < lambdaK.size(); i++) {
    std::cout << "K: ";
    for (int j = 0; j < 4; j++) {
      std::cout << K_series[i][j];
    }
    std::cout << ", lambdaK: " << std::setprecision(10) << lambdaK(i) << std::endl;
  }

  for (unsigned int i = 0; i < hK.size(); i++) {
    std::cout << "K: ";
    for (int j = 0; j < 4; j++) {
      std::cout << K_series[i][j];
    }
    std::cout << ", hK: " << std::setprecision(10) << hK(i) << std::endl;
  }

  std::cout << "Testing Ergodic Measure Functions...  COMPLETE" << std::endl << std::endl;

  return 0;
}
