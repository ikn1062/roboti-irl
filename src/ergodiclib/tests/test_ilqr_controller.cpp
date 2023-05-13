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
#include <ergodiclib/controller.hpp>

using namespace ergodiclib;

TEST_CASE("iLQR Controller Objective Function", "[Controller]")
{
    // Linearized around x = {0.0, 0.0, 0.0, 0.0}
    arma::vec x0({0.0, 0.0, 0.0, 0.0});
    arma::vec u0({0.0});
    double M0 = 10.0;
    double m0 = 5.0;
    double l0 = 1.0;
    double dt = 0.5;
    double t0 = 0.0;
    double tf = 1.0;
    CartPole cartpole = CartPole(x0, u0, dt, t0, tf, M0, m0, l0);
    
    arma::mat Q(4, 4, arma::fill::eye);
    Q(0,0) = 0.1;
    Q(1,1) = 0.1;
    Q(2,2) = 10;
    Q(3,3) = 1;

    arma::mat R(1, 1, arma::fill::eye);
    R(0, 0) = 1;

    arma::mat P(4, 4, arma::fill::eye);
    P = 100 * P;

    arma::mat r(4, 1, arma::fill::zeros);

    double alpha = 0.40;
    double beta = 0.85;
    double eps = 0.01;

    std::pair<arma::mat, arma::mat> trajectory = cartpole.createTrajectory();
    arma::mat X = trajectory.first;
    arma::mat U = trajectory.second;

    ilqrController controller = ilqrController(cartpole, x0, Q, R, P, r, dt, t0, tf, alpha, beta, eps);
    double obj_cost = controller.objectiveJ(X, U, P);
    REQUIRE(almost_equal(obj_cost, 0.0, 1e-6));
        
    arma::vec x0_new({0.0, 0.0, PI, 0.0});
    arma::mat U_new(U.n_rows, U.n_cols, arma::fill::zeros);
    arma::mat X_new = cartpole.createTrajectory(x0_new, U_new);

    // FINISH THIS TEST BASED OFF XNEW
    double obj_cost_new = controller.objectiveJ(X_new, U_new, P);
    double new_cost_test = pow(ergodiclib::PI, 2) * 2.5 + pow(ergodiclib::PI, 2) * 50;
    REQUIRE(almost_equal(obj_cost_new, new_cost_test, 1e-6));
}

TEST_CASE("iLQR Controller Calculate bT", "[Controller]")
{
    // Linearized around x = {0.0, 0.0, PI, 0.0}
    arma::vec x0({0.0, 0.0, PI, 0.0});
    arma::vec u0({10.0});
    double M0 = 10.0;
    double m0 = 5.0;
    double l0 = 1.0;
    double dt = 0.5;
    double t0 = 0.0;
    double tf = 1.0;
    CartPole cartpole = CartPole(x0, u0, dt, t0, tf, M0, m0, l0);
    
    arma::mat Q(4, 4, arma::fill::eye);
    Q(0,0) = 0.1;
    Q(1,1) = 0.1;
    Q(2,2) = 10;
    Q(3,3) = 1;

    arma::mat R(1, 1, arma::fill::eye);
    R(0, 0) = 1;

    arma::mat P(4, 4, arma::fill::eye);
    P = 100 * P;

    arma::mat r(4, 1, arma::fill::zeros);

    double alpha = 0.40;
    double beta = 0.85;
    double eps = 0.01;

    std::pair<arma::mat, arma::mat> trajectory = cartpole.createTrajectory();
    arma::mat X = trajectory.first;
    arma::mat U = trajectory.second;

    ilqrController controller = ilqrController(cartpole, x0, Q, R, P, r, dt, t0, tf, alpha, beta, eps);
    arma::mat bT = controller.calculate_bT(U);

    arma::mat bT_test(2, 1, arma::fill::ones);
    bT_test = 10.0 * bT_test;
    for (unsigned int i = 0; i < bT.n_rows; i++) {
        REQUIRE(almost_equal(bT(i), bT_test(i), 1e-6));
    }
}

TEST_CASE("iLQR Controller Calculate aT", "[Controller]")
{
    // Linearized around x = {0.0, 0.0, PI, 0.0}
    arma::vec x0({0.0, 0.0, PI, 0.0});
    arma::vec u0({10.0});
    double M0 = 10.0;
    double m0 = 5.0;
    double l0 = 1.0;
    double dt = 0.5;
    double t0 = 0.0;
    double tf = 1.0;
    CartPole cartpole = CartPole(x0, u0, dt, t0, tf, M0, m0, l0);
    
    arma::mat Q(4, 4, arma::fill::eye);
    Q(0,0) = 0.1;
    Q(1,1) = 0.1;
    Q(2,2) = 10;
    Q(3,3) = 1;

    arma::mat R(1, 1, arma::fill::eye);
    R(0, 0) = 1;

    arma::mat P(4, 4, arma::fill::eye);
    P = 100 * P;

    arma::mat r(4, 1, arma::fill::zeros);

    double alpha = 0.40;
    double beta = 0.85;
    double eps = 0.01;

    std::pair<arma::mat, arma::mat> trajectory = cartpole.createTrajectory();
    arma::mat X = trajectory.first;
    arma::mat U = trajectory.second;

    ilqrController controller = ilqrController(cartpole, x0, Q, R, P, r, dt, t0, tf, alpha, beta, eps);
    arma::mat aT = controller.calculate_aT(X);

    arma::mat aT_test(2, 4, arma::fill::ones);
    aT_test.row(0) = {0.0, 0.0, 31.4159, 0.0};
    aT_test.row(1) = {0.0112, 0.0398, 30.5506, -0.1931};

    for (unsigned int i = 0; i < aT.n_rows; i++) {
        for (unsigned int j = 0; j < aT.n_cols; j++) {
            REQUIRE(almost_equal(aT(i, j), aT_test(i, j), 1e-4));
        }
    }
}

TEST_CASE("iLQR Controller Calculate List P/r", "[Controller]")
{
    // Linearized around x = {0.0, 0.0, PI, 0.0}
    arma::vec x0({0.0, 0.0, PI, 0.0});
    arma::vec u0({10.0});
    double M0 = 10.0;
    double m0 = 5.0;
    double l0 = 1.0;
    double dt = 0.5;
    double t0 = 0.0;
    double tf = 1.0;
    CartPole cartpole = CartPole(x0, u0, dt, t0, tf, M0, m0, l0);
    
    arma::mat Q(4, 4, arma::fill::eye);
    Q(0,0) = 0.1;
    Q(1,1) = 0.1;
    Q(2,2) = 10;
    Q(3,3) = 1;

    arma::mat R(1, 1, arma::fill::eye);
    R(0, 0) = 1;

    arma::mat P(4, 4, arma::fill::eye);
    P = 100 * P;

    arma::mat r(4, 1, arma::fill::zeros);

    double alpha = 0.40;
    double beta = 0.85;
    double eps = 0.01;

    std::pair<arma::mat, arma::mat> trajectory = cartpole.createTrajectory();
    arma::mat X = trajectory.first;
    arma::mat U = trajectory.second;

    ilqrController controller = ilqrController(cartpole, x0, Q, R, P, r, dt, t0, tf, alpha, beta, eps);
    arma::mat aT = controller.calculate_aT(X);
    arma::mat bT = controller.calculate_bT(U);

    std::pair<std::vector<arma::mat>, std::vector<arma::mat>> listPr = controller.calculatePr(X, U, aT, bT);
    std::vector<arma::mat> Plist = listPr.first;
    std::vector<arma::mat> rlist = listPr.second;

    arma::mat Ptest_1(4, 4, arma::fill::zeros);
    Ptest_1.row(0) = {0.0, 0.0, 0.0, 0.0};
    Ptest_1.row(1) = {0.0, 0.0, 0.0, 0.0};
    Ptest_1.row(2) = {0.0, 0.0, 0.0, 0.0};
    Ptest_1.row(3) = {0.0, 0.0, 0.0, 0.0};

    arma::mat rtest_1(4, 1, arma::fill::zeros);
    rtest_1.row(0) = {0.0, 0.0, 0.0, 0.0};

    for (unsigned int i = 0; i < Plist[1].n_rows; i++) {
        for (unsigned int j = 0; j < Plist[1].n_cols; j++) {
            REQUIRE(almost_equal(Ptest_1(i, j), Plist[1](i, j), 1e-6));
        }
    }

    for (unsigned int i = 0; i < rlist[1].n_rows; i++) {
        for (unsigned int j = 0; j < rlist[1].n_cols; j++) {
            REQUIRE(almost_equal(rtest_1(i, j), rlist[1](i, j), 1e-6));
        }
    }
}

TEST_CASE("iLQR Controller Calculate List Zeta", "[Controller]")
{
    // Linearized around x = {0.0, 0.0, PI, 0.0}
    arma::vec x0({0.0, 0.0, PI, 0.0});
    arma::vec u0({10.0});
    double M0 = 10.0;
    double m0 = 5.0;
    double l0 = 1.0;
    double dt = 0.5;
    double t0 = 0.0;
    double tf = 1.0;
    CartPole cartpole = CartPole(x0, u0, dt, t0, tf, M0, m0, l0);
    
    arma::mat Q(4, 4, arma::fill::eye);
    Q(0,0) = 0.1;
    Q(1,1) = 0.1;
    Q(2,2) = 10;
    Q(3,3) = 1;

    arma::mat R(1, 1, arma::fill::eye);
    R(0, 0) = 1;

    arma::mat P(4, 4, arma::fill::eye);
    P = 100 * P;

    arma::mat r(4, 1, arma::fill::zeros);

    double alpha = 0.40;
    double beta = 0.85;
    double eps = 0.01;

    std::pair<arma::mat, arma::mat> trajectory = cartpole.createTrajectory();
    arma::mat X = trajectory.first;
    arma::mat U = trajectory.second;

    ilqrController controller = ilqrController(cartpole, x0, Q, R, P, r, dt, t0, tf, alpha, beta, eps);
    arma::mat aT = controller.calculate_aT(X);
    arma::mat bT = controller.calculate_bT(U);

    std::pair<arma::mat, arma::mat> descentDirection = controller.calculateZeta(X, U);
    arma::mat zeta = descentDirection.first;
    arma::mat vega = descentDirection.second;

    arma::mat zeta_test_T(zeta.n_cols, zeta.n_rows, arma::fill::zeros);
    zeta_test_T.row(0) = {0.0, 0.0, 0.0, 0.0};
    zeta_test_T.row(1) = {0.0, 0.0, 0.0, 0.0};

    arma::mat vega_test_T(vega.n_cols, vega.n_rows, arma::fill::zeros);
    vega_test_T.row(0) = {0.0};
    vega_test_T.row(1) = {0.0};

    arma::mat zeta_test = zeta_test_T.t();
    arma::mat vega_test = vega_test_T.t();

    for (unsigned int i = 0; i < zeta.n_rows; i++) {
        for (unsigned int j = 0; j < zeta.n_cols; j++) {
            REQUIRE(almost_equal(zeta_test(i, j), zeta(i, j), 1e-6));
        }
    }

    for (unsigned int i = 0; i < vega.n_rows; i++) {
        for (unsigned int j = 0; j < vega.n_cols; j++) {
            REQUIRE(almost_equal(vega(i, j), vega(i, j), 1e-6));
        }
    }
}