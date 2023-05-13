#include <catch2/catch_test_macros.hpp>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <iostream>
#include <armadillo>

#define private public

#include <ergodiclib/cartpole.hpp>
#include <ergodiclib/model.hpp>
#include <ergodiclib/num_utils.hpp>

TEST_CASE("Cartpole Calculate A Matrix", "[CartPole]")
{
    // Linearized around x = {0.0, 0.0, 0.0, 0.0}
    double M0 = 10.0;
    double m0 = 5.0;
    double l0 = 1.0;
    CartPole cartpole = CartPole(M0, m0, l0);
    arma::vec x0({0.0, 0.0, 0.0, 0.0});
    arma::vec u0({0.0});

    arma::mat A0 = cartpole.getA(x0, u0);
    arma::mat A0_test(4, 4, arma::fill::zeros);
    A0_test.row(0) = {0.0, 1.0, 0.0, 0.0};
    A0_test.row(1) = {0.0, 0.0, ((5.0*9.81)/10.0), 0.0};
    A0_test.row(2) = {0.0, 0.0, 0.0, 1.0};
    A0_test.row(3) = {0.0, 0.0, ((9.81*(10.0 + 5.0))/10.0), 0.0};

    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++) {
            REQUIRE(ergodiclib::almost_equal(A0(i, j), A0_test(i, j), 1e-6));
        }
    }

    // Linearized around x = {2.0, 0.0, PI, 0.0}
    arma::vec x1({2.0, 0.0, ergodiclib::PI, 0.0});
    arma::mat A1 = cartpole.getA(x1, u0);

    arma::mat A1_test(4, 4, arma::fill::zeros);
    A1_test.row(0) = {0.0, 1.0, 0.0, 0.0};
    A1_test.row(1) = {0.0, 0.0, ((5.0*9.81)/10.0), 0.0};
    A1_test.row(2) = {0.0, 0.0, 0.0, 1.0};
    A1_test.row(3) = {0.0, 0.0, ((-9.81*(10.0 + 5.0))/10.0), 0.0};

    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++) {
            REQUIRE(ergodiclib::almost_equal(A1(i, j), A1_test(i, j), 1e-6));
        }
    }

    // Linearized around x = {1.0, 2.0, PI/2, 1.0}, F = {0.0}
    arma::vec x2({1.0, 2.0, ergodiclib::PI/2.0, 1.0});

    arma::mat A2 = cartpole.getA(x2, u0);
    arma::mat A2_test(4, 4, arma::fill::zeros);;
    A2_test.row(0) = {0.0, 1.0, 0.0, 0.0};
    A2_test.row(1) = {0.0, 0.0, -3.27, -0.6666667};
    A2_test.row(2) = {0.0, 0.0, 0.0, 1.0};
    A2_test.row(3) = {0.0, 0.0, 0.3333333, 0.0};

    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++) {
            REQUIRE(ergodiclib::almost_equal(A2(i, j), A2_test(i, j), 1e-6));
        }
    }

    // Linearized around x = {10.0, 2.0, PI/4, 3.0}, F = {0.0}
    arma::vec x3({10.0, 2.0, ergodiclib::PI/4.0, 3.0});

    arma::mat A3 = cartpole.getA(x3, u0);
    arma::mat A3_test(4, 4, arma::fill::zeros);
    A3_test.row(0) = {0.0, 1.0, 0.0, 0.0};
    A3_test.row(1) = {0.0, 0.0, -2.3121506, -1.6970562};
    A3_test.row(2) = {0.0, 0.0, 0.0, 1.0};
    A3_test.row(3) = {0.0, 0.0, 5.7144366, -1.2};

    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++) {
            REQUIRE(ergodiclib::almost_equal(A3(i, j), A3_test(i, j), 1e-6));
        }
    }
}

TEST_CASE("Cartpole Calculate B Matrix", "[CartPole]")
{
    // Linearized around x = {0.0, 0.0, 0.0, 0.0}
    double M0 = 10.0;
    double m0 = 5.0;
    double l0 = 1.0;
    CartPole cartpole = CartPole(M0, m0, l0);
    arma::vec x0({0.0, 0.0, 0.0, 0.0});
    arma::vec u0({0.0});

    arma::mat B0 = cartpole.getB(x0, u0);
    arma::mat B0_test(4, 1, arma::fill::zeros);
    B0_test.row(0) = {0.0};
    B0_test.row(1) = {1.0/10.0};
    B0_test.row(2) = {0.0};
    B0_test.row(3) = {1.0/10.0};

    for (int i = 0; i < 4; i++){
        REQUIRE(ergodiclib::almost_equal(B0(i, 0), B0_test(i, 0), 1e-6));
    }

    // Linearized around x = {1.0, 2.0, PI/2, 1.0}, F = {0.0}
    arma::vec x2({1.0, 2.0, ergodiclib::PI/2.0, 1.0});

    arma::mat B2 = cartpole.getB(x2, u0);
    arma::mat B2_test(4, 1, arma::fill::zeros);
    B2_test.row(0) = {0.0};
    B2_test.row(1) = {1.0/15.0};
    B2_test.row(2) = {0.0};
    B2_test.row(3) = {0.0};

    for (int i = 0; i < 4; i++){
        REQUIRE(ergodiclib::almost_equal(B2(i, 0), B2_test(i, 0), 1e-6));
    }

    // Linearized around x = {10.0, 2.0, PI/4, 3.0}, F = {0.0}
    arma::vec x3({10.0, 2.0, ergodiclib::PI/4.0, 3.0});

    arma::mat B3 = cartpole.getB(x3, u0);
    arma::mat B3_test(4, 1, arma::fill::zeros);
    B3_test.row(0) = {0.0};
    B3_test.row(1) = {0.08};
    B3_test.row(2) = {0.0};
    B3_test.row(3) = {0.05656854};

    for (int i = 0; i < 4; i++){
        REQUIRE(ergodiclib::almost_equal(B3(i, 0), B3_test(i, 0), 1e-6));
    }
}

TEST_CASE("Cartpole Dynamics", "[CartPole]")
{
    double M0 = 10.0;
    double m0 = 5.0;
    double l0 = 1.0;
    CartPole cartpole = CartPole(M0, m0, l0);

    // Linearized around x = {10.0, 2.0, PI/4, 3.0}, F = {10.0}
    arma::vec x0({10.0, 2.0, ergodiclib::PI/4.0, 3.0});
    arma::vec u0({10.0});

    arma::vec xdot = cartpole.dynamics(x0, u0);
    arma::vec xdot_test({2.0, 0.21641558, 3.0, 7.0897464});

    for (int i = 0; i < 4; i++){
        REQUIRE(ergodiclib::almost_equal(xdot(i), xdot_test(i), 1e-6));
    }
}

TEST_CASE("Cartpole Integrate Step", "[CartPole]")
{
    double M0 = 10.0;
    double m0 = 5.0;
    double l0 = 1.0;
    double dt = 0.0005;
    CartPole cartpole = CartPole(M0, m0, l0);

    // Linearized around x = {10.0, 2.0, PI/4, 3.0}, F = {10.0}
    arma::vec x0({10.0, 2.0, ergodiclib::PI/4.0, 3.0});
    arma::vec u0({10.0});
    arma::vec x1 = cartpole.integrate(x0, u0, dt);

    arma::vec xdot_test({2.0, 0.21641558, 3.0, 7.0897464});
    arma::vec x1_test = x0 + dt * xdot_test;

    for (int i = 0; i < 4; i++){
        REQUIRE(ergodiclib::almost_equal(x1(i), x1_test(i), 1e-4));
    }
}

/*
TEST_CASE("Cartpole Create Trajectory", "[CartPole]")
{
    
}
*/
