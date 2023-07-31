#include <catch2/catch_test_macros.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <iostream>
#include <armadillo>

#define private public

#include <ergodiclib/file_utils.hpp>
#include <ergodiclib/ergodic_measure.hpp>
#include <ergodiclib/ergodic_controller.hpp>
#include <cartpole/cartpole_sys.hpp>

using namespace ergodiclib;

// // Test case for checking if files can be read
// TEST_CASE("Open File", "[FileUtils]")
// {
//     // Running with 1 demonstration
//     std::string file_demonstration_path = "/home/student/roboti-irl/src/cartpole/demonstrations/";
//     std::vector<arma::mat> demonstrations = readDemonstrations(file_demonstration_path, 4);

//     arma::colvec col0({0.0000000000000000, 0.0000000000000000, 3.1415926535897931, 0.0000000000000000});
//     arma::colvec col999({-4.4582070746963707e00, -1.7044627381818206e-01, 7.5140762258642191e-02, 5.4650465437959617e-01});
//     REQUIRE(demonstrations.size() == 2);
//     REQUIRE(demonstrations[0].n_cols == 1000);

//     REQUIRE(arma::approx_equal(demonstrations[0].col(0), col0, "absdiff", 0.0001));
//     REQUIRE(arma::approx_equal(demonstrations[0].col(999), col999, "absdiff", 0.0001));

// }

TEST_CASE("Test calculateHk - Normalization Factor", "[fourierBasis]")
{
  // During testing of hk, the dx value of the integral was set to 0.0001
  std::pair<double, double> pair1(-PI, PI);
  std::vector<std::pair<double, double>> lengths1{pair1};
  fourierBasis Basis_1D = fourierBasis(lengths1, 1, 4);

  std::vector<int> K_vec1({1});
  double hk1 = Basis_1D._hkVec.at(1);
  double hk_test1 = 1.77245385;
  REQUIRE(ergodiclib::almost_equal(hk1, hk_test1, 1e-3));

  std::vector<int> K_vec2({2});
  double hk2 = Basis_1D._hkVec.at(2);
  double hk_test2 = 1.77245385;
  REQUIRE(ergodiclib::almost_equal(hk2, hk_test2, 1e-3));

  std::pair<double, double> pair3(-PI, PI);
  std::pair<double, double> pair4(-5, 5);
  std::vector<std::pair<double, double>> lengths2{pair3, pair4};
  fourierBasis Basis_2D = fourierBasis(lengths2, 2, 4);

  std::vector<int> K_vec3({0, 0});
  double hk3 = Basis_2D._hkVec.at(0);
  double hk_test3 = 7.9266545;
  REQUIRE(ergodiclib::almost_equal(hk3, hk_test3, 1e-3));

  std::vector<int> K_vec4({1, 3});
  double hk4 = Basis_2D._hkVec.at(8);
  double hk_test4 = 3.96332729;
  REQUIRE(ergodiclib::almost_equal(hk4, hk_test4, 1e-3));

  std::pair<double, double> pair5(-5, 5);
  std::pair<double, double> pair6(-10, 10);
  std::pair<double, double> pair7(-PI, PI);
  std::pair<double, double> pair8(-10, 10);
  std::vector<std::pair<double, double>> lengths3{pair5, pair6, pair7, pair8};
  fourierBasis Basis_4D = fourierBasis(lengths3, 4, 4);

  std::vector<int> K_vec5({0, 0, 0, 0});
  double hk5 = Basis_4D._hkVec.at(0);
  double hk_test5 = 158.533091904;
  REQUIRE(ergodiclib::almost_equal(hk5, hk_test5, 1e-1));

  // std::vector<int> K_vec6({1, 3, 4, 2});
  // double hk6 = Basis_4D.calculateHk(K_vec6, 1);
  // double hk_test6 = 39.6332729;
  // REQUIRE(ergodiclib::almost_equal(hk6, hk_test6, 1e-1));
}

TEST_CASE("Test calculateFk - Fourier Basis", "[fourierBasis]")
{
  // During testing of hk, the dx value of the integral was set to 0.0001
  std::pair<double, double> pair1(-PI, PI);
  std::pair<double, double> pair2(-5, 5);
  std::vector<std::pair<double, double>> lengths1{pair1, pair2};
  fourierBasis Basis_2D = fourierBasis(lengths1, 2, 4);

  arma::vec x1({PI / 4, 1.2});
  arma::vec x2({PI / 8, 3.2});

  double fk1 = Basis_2D.calculateFk(x1, 0);
  double fk_test1 = 0.126156627;
  REQUIRE(ergodiclib::almost_equal(fk1, fk_test1, 1e-2));

  double fk2 = Basis_2D.calculateFk(x1, 8);
  double fk_test2 = 0.09925;
  std::vector<int> kvec = Basis_2D._kSeries[7];
  REQUIRE(ergodiclib::almost_equal(fk2, fk_test2, 1e-2));

  double fk3 = Basis_2D.calculateFk(x2, 8);
  double fk_test3 = -0.24551;
  REQUIRE(ergodiclib::almost_equal(fk3, fk_test3, 1e-2));

  std::pair<double, double> pair5(-5, 5);
  std::pair<double, double> pair6(-10, 10);
  std::pair<double, double> pair7(-PI, PI);
  std::pair<double, double> pair8(-10, 10);
  std::vector<std::pair<double, double>> lengths2{pair5, pair6, pair7, pair8};
  fourierBasis Basis_4D = fourierBasis(lengths2, 4, 4);

  arma::vec x3({5.8, -3.1, PI / 4, 1.2});
  double fk4 = Basis_4D.calculateFk(x3, 0);
  double fk_test4 = 0.00630783;
  REQUIRE(ergodiclib::almost_equal(fk4, fk_test4, 1e-2));
}

TEST_CASE("Test calculateDFk - Jacobian of Fourier Basis", "[fourierBasis]")
{
  std::pair<double, double> pair5(-5, 5);
  std::pair<double, double> pair6(-10, 10);
  std::pair<double, double> pair7(-PI, PI);
  std::pair<double, double> pair8(-10, 10);
  std::vector<std::pair<double, double>> lengths2{pair5, pair6, pair7, pair8};
  fourierBasis Basis_4D = fourierBasis(lengths2, 4, 4);

  arma::vec x2({5.8, -3.1, PI / 8, 1.2});
  std::vector<int> K_vec2 = Basis_4D._kSeries[323];

  arma::rowvec dfk2 = Basis_4D.calculateDFk(x2, 323);
  arma::rowvec dfk_test2({0.0026, -0.0034, 0.0148, 0.0022});
  // dfk2.print("dfk2: ");
  // dfk_test2.print("dfk2_test: ");
  REQUIRE(arma::approx_equal(dfk2, dfk_test2, "absdiff", 0.001));
}

