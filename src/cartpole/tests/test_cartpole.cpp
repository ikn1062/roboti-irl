#include <catch2/catch_test_macros.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <iostream>
#include <armadillo>

#define private public

#include <cartpole/cartpole_sys.hpp>
#include <ergodiclib/num_utils.hpp>

using namespace ergodiclib;

TEST_CASE("Normalize Angle", "[NumUtils]")
{
  double angle_0 = ergodiclib::PI;
  double angle_0_norm = ergodiclib::normalizeAngle(angle_0);
  double angle_0_test = ergodiclib::PI;
  REQUIRE(ergodiclib::almost_equal(angle_0_norm, angle_0_test, 1e-6));

  double angle_1 = -1.0 * ergodiclib::PI;
  double angle_1_norm = ergodiclib::normalizeAngle(angle_1);
  double angle_1_test = ergodiclib::PI;
  REQUIRE(ergodiclib::almost_equal(angle_1_norm, angle_1_test, 1e-6));

  double angle_2 = 0.0;
  double angle_2_norm = ergodiclib::normalizeAngle(angle_2);
  double angle_2_test = 0.0;
  REQUIRE(ergodiclib::almost_equal(angle_2_norm, angle_2_test, 1e-6));

  double angle_3 = ergodiclib::PI / 4;
  double angle_3_norm = ergodiclib::normalizeAngle(angle_3);
  double angle_3_test = ergodiclib::PI / 4;
  REQUIRE(ergodiclib::almost_equal(angle_3_norm, angle_3_test, 1e-6));

  double angle_4 = -1.0 * ergodiclib::PI / 4;
  double angle_4_norm = ergodiclib::normalizeAngle(angle_4);
  double angle_4_test = -1.0 * ergodiclib::PI / 4;
  REQUIRE(ergodiclib::almost_equal(angle_4_norm, angle_4_test, 1e-6));
}

TEST_CASE("Cartpole Dynamics", "[CartPole]")
{
  double M0 = 10.0;
  double m0 = 5.0;
  double l0 = 2.0;
  CartPole cartpole = CartPole(M0, m0, l0);

  arma::vec x0({0.0, 0.0, ergodiclib::PI / 3.0, 0.0});
  arma::vec u0({0.0});
  arma::vec xdot = cartpole.dynamics(x0, u0);
  arma::vec xdot_test({0, -1.1328, 0, 3.3983});
  for (int i = 0; i < 4; i++) {
    REQUIRE(ergodiclib::almost_equal(xdot(i), xdot_test(i), 1e-4));
  }


  arma::vec x1({0.0, 0.0, ergodiclib::PI / 4.0, ergodiclib::PI / 4.0});
  xdot = cartpole.dynamics(x1, u0);
  xdot_test = {0, -1.0691, ergodiclib::PI / 4.0, 2.8848};
  for (int i = 0; i < 4; i++) {
    REQUIRE(ergodiclib::almost_equal(xdot(i), xdot_test(i), 1e-4));
  }


  arma::vec x2({1.0, 1.0, ergodiclib::PI / 4.0, ergodiclib::PI / 4.0});
  arma::vec u2({1.0});
  xdot = cartpole.dynamics(x2, u2);
  xdot_test = {1.0, -0.9929, ergodiclib::PI / 4.0, 2.8646};
  for (int i = 0; i < 4; i++) {
    REQUIRE(ergodiclib::almost_equal(xdot(i), xdot_test(i), 1e-4));
  }
}

TEST_CASE("Cartpole Calculate B Matrix", "[CartPole]")
{
  // Linearized around x = {0.0, 0.0, 0.0, 0.0}
  double M0 = 10.0;
  double m0 = 5.0;
  double l0 = 2.0;
  CartPole cartpole = CartPole(M0, m0, l0);

  arma::vec x0({0.0, 0.0, 0.0, 0.0});
  arma::vec u0({0.0});
  arma::mat B0 = cartpole.getB(x0, u0);
  arma::mat B0_test(4, 1, arma::fill::zeros);
  B0_test.row(0) = {0.0};
  B0_test.row(1) = {0.0889};
  B0_test.row(2) = {0.0};
  B0_test.row(3) = {-0.0333};

  for (int i = 0; i < 4; i++) {
    REQUIRE(ergodiclib::almost_equal(B0(i, 0), B0_test(i, 0), 1e-4));
  }

  // Linearized around x = {1.0, 2.0, PI/4, 1.0}, F = {1.0}
  arma::vec x2({1.0, 2.0, ergodiclib::PI / 4.0, 1.0});
  arma::vec u1({1.0});

  arma::mat B2 = cartpole.getB(x2, u1);
  arma::mat B2_test(4, 1, arma::fill::zeros);
  B2_test.row(0) = {0.0};
  B2_test.row(1) = {0.0762};
  B2_test.row(2) = {0.0};
  B2_test.row(3) = {-0.0202};

  for (int i = 0; i < 4; i++) {
    REQUIRE(ergodiclib::almost_equal(B2(i, 0), B2_test(i, 0), 1e-4));
  }

  // Linearized around x = {10.0, 2.0, PI/3, 3.0}, F = {1.0}
  arma::vec x3({10.0, 2.0, ergodiclib::PI / 3.0, 3.0});

  arma::mat B3 = cartpole.getB(x3, u1);
  arma::mat B3_test(4, 1, arma::fill::zeros);
  B3_test.row(0) = {0.0};
  B3_test.row(1) = {0.0711};
  B3_test.row(2) = {0.0};
  B3_test.row(3) = {-0.0133};

  for (int i = 0; i < 4; i++) {
    REQUIRE(ergodiclib::almost_equal(B3(i, 0), B3_test(i, 0), 1e-4));
  }
}


TEST_CASE("Cartpole Calculate A Matrix", "[CartPole]")
{
  // Linearized around x = {0.0, 0.0, 0.0, 0.0}
  double M0 = 10.0;
  double m0 = 5.0;
  double l0 = 2.0;
  CartPole cartpole = CartPole(M0, m0, l0);

  arma::vec x0({0.0, 0.0, 0.0, 0.0});
  arma::vec u0({0.0});

  arma::mat A0 = cartpole.getA(x0, u0);
  arma::mat A0_test(4, 4, arma::fill::zeros);
  A0_test.row(0) = {0.0, 1.0, 0.0, 0.0};
  A0_test.row(1) = {0.0, 0.0, (-3.2700), 0.0};
  A0_test.row(2) = {0.0, 0.0, 0.0, 1.0};
  A0_test.row(3) = {0.0, 0.0, (4.9050), 0.0};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      REQUIRE(ergodiclib::almost_equal(A0(i, j), A0_test(i, j), 1e-4));
    }
  }

  // Linearized around x = {0.0, 0.0, PI/2, 0.0}
  arma::vec x1({0.0, 0.0, ergodiclib::PI / 2, 0.0});
  arma::mat A1 = cartpole.getA(x1, u0);

  arma::mat A1_test(4, 4, arma::fill::zeros);
  A1_test.row(0) = {0.0, 1.0, 0.0, 0.0};
  A1_test.row(1) = {0.0, 0.0, (2.4525), 0.0};
  A1_test.row(2) = {0.0, 0.0, 0.0, 1.0};
  A1_test.row(3) = {0.0, 0.0, 0.0, 0.0};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      REQUIRE(ergodiclib::almost_equal(A1(i, j), A1_test(i, j), 1e-4));
    }
  }

  // Linearized around x = {1.0, 2.0, PI/4, 0.0}, F = {0.0}
  arma::vec x2({1.0, 2.0, PI / 4, 0.0});

  arma::mat A2 = cartpole.getA(x2, u0);
  arma::mat A2_test(4, 4, arma::fill::zeros);
  A2_test.row(0) = {0.0, 1.0, 0.0, 0.0};
  A2_test.row(1) = {0.0, 0.0, 0.4004, 0.0};
  A2_test.row(2) = {0.0, 0.0, 0.0, 1.0};
  A2_test.row(3) = {0.0, 0.0, 2.1235, 0.0};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      REQUIRE(ergodiclib::almost_equal(A2(i, j), A2_test(i, j), 1e-4));
    }
  }

  // Linearized around x = {0.0, 0.0, PI/4, 3.0}, F = {0.0}
  arma::vec x3({0.0, 0.0, PI / 4, PI / 4});

  arma::mat A3 = cartpole.getA(x3, u0);
  arma::mat A3_test(4, 4, arma::fill::zeros);
  A3_test.row(0) = {0.0, 1.0, 0.0, 0.0};
  A3_test.row(1) = {0.0, 0.0, 0.6378, 0.8463};
  A3_test.row(2) = {0.0, 0.0, 0.0, 1.0};
  A3_test.row(3) = {0.0, 0.0, 2.1487, -0.2244};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      REQUIRE(ergodiclib::almost_equal(A3(i, j), A3_test(i, j), 1e-4));
    }
  }

  // Linearized around x = {0.0, 0.0, PI/4, PI/2}, F = {1.0}
  arma::vec x4({0.0, 0.0, PI / 4, PI / 2});
  arma::vec u1({1.0});

  arma::mat A4 = cartpole.getA(x4, u1);
  arma::mat A4_test(4, 4, arma::fill::zeros);
  A4_test.row(0) = {0.0, 1.0, 0.0, 0.0};
  A4_test.row(1) = {0.0, 0.0, 1.3281, 1.6925};
  A4_test.row(2) = {0.0, 0.0, 0.0, 1.0};
  A4_test.row(3) = {0.0, 0.0, 2.2502, -0.4488};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      REQUIRE(ergodiclib::almost_equal(A4(i, j), A4_test(i, j), 1e-4));
    }
  }

  // Linearized around x = {0.0, 0.0, PI/3, PI/4}, F = {1.0}
  arma::vec x5({0.0, 0.0, PI / 3, PI / 4});

  arma::mat A5 = cartpole.getA(x5, u1);
  arma::mat A5_test(4, 4, arma::fill::zeros);
  A5_test.row(0) = {0.0, 1.0, 0.0, 0.0};
  A5_test.row(1) = {0.0, 0.0, 1.6848, 0.9674};
  A5_test.row(2) = {0.0, 0.0, 0.0, 1.0};
  A5_test.row(3) = {0.0, 0.0, 1.3021, -0.1814};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      REQUIRE(ergodiclib::almost_equal(A5(i, j), A5_test(i, j), 1e-4));
    }
  }
}
