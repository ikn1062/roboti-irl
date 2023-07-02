#ifndef BASE_CONTROLLER_INCLUDE_GUARD_HPP
#define BASE_CONTROLLER_INCLUDE_GUARD_HPP

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <limits>

#include <ergodiclib/num_utils.hpp>
#include <ergodiclib/model.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

#define UNUSED(x) (void)(x)

namespace ergodiclib
{
template<class ModelTemplate>
class BaseController
{
public:
  /// \brief Base Constructor for Base Controller Class
  BaseController()
  {}

  /// \brief Constructor for iLQR Controller
  /// \param model_in Model input following Concept Template
  /// \param Q Q Matrix (Trajectory Penalty)
  /// \param R R Matrix (Control Penalty)
  /// \param P P Matrix (Final Trajectory Penalty)
  /// \param r r Matrix (Final Control Penalty)
  /// \param max_iter_in Max iteration for control descent
  /// \param a Alpha - Controller multiplier
  /// \param b Beta - Controller multiplier for armijo line search
  /// \param e Epsilon - Convergence Value for Objective Function
  BaseController(
    ModelTemplate model_in, arma::mat Q, arma::mat R, arma::mat P, arma::mat r,
    unsigned int max_iter_in, double a, double b, double e)
  : model(model_in),
    Q_mat(Q),
    R_mat(R),
    P_mat(P),
    r_mat(r),
    max_iter(max_iter_in),
    alpha(a),
    beta(b),
    eps(e)
  {
    x0 = model_in.x0;
    dt = model_in.dt;
    num_iter = (int) ((model_in.tf - model_in.t0) / dt);
  }

  /// \brief Begins iLQR controller
  /// \return Optimal Control and Trajectory as a pair <Trajectory, Controls>
  std::pair<arma::mat, arma::mat> iLQR()
  {
    // Create variables for iLQR loop
    std::pair<arma::mat, arma::mat> trajectory, descentDirection;
    arma::mat X, U, zeta, vega, aT, bT;
    double DJ, J, J_new, gamma;
    unsigned int i, n;

    // Get an initial trajectory
    trajectory = model.createTrajectory();
    X = trajectory.first;
    U = trajectory.second;

    J = objectiveJ(X, U);
    DJ = 1000.0;
    i = 0;

    // The Controller can use DJ or J in the while loop in comparison to eps
    while (std::abs(DJ) > eps && i < max_iter) {
      aT = calculate_aT(X);
      bT = calculate_bT(U);
      descentDirection = calculateZeta(X, U, aT, bT);
      zeta = descentDirection.first;
      vega = descentDirection.second;

      DJ = calculateDJ(descentDirection, aT, bT);
      J = objectiveJ(X, U);
      J_new = std::numeric_limits<double>::max();
      gamma = beta;
      n = 1;

      while (J_new > J + alpha * gamma * DJ && n < 2) {
        U = U + gamma * vega;
        X = model.createTrajectory(x0, U);
        J_new = objectiveJ(X, U);
        n += 1;
        gamma = pow(beta, n);
      }

      trajectory = {X, U};
      i += 1;

      // std::cout << "i: " << i << std::endl;
      // std::cout << "DJ: " << std::abs(DJ) << std::endl;
      // std::cout << "J: " << std::abs(J_new) << std::endl;
      // (X.col(X.n_cols - 1)).print("End X: ");

    }
    // // Used to save files
    // std::string x_file = "erg_trajectory";
    // arma::mat XT = X.t();
    // XT.save(x_file, arma::csv_ascii);
    // arma::mat UT = U.t();
    // std::string u_file = "erg_control";
    // UT.save(u_file, arma::csv_ascii);

    return trajectory;
  }

  /// \brief Begins model predictive controller
  /// @param x0 Initial State vector at time t=0 for controls
  /// @param u0 Initial Control vector at time t=0
  /// @param num_steps Number of time steps given model_dt
  /// @param max_iterations Number of max iterations for gradient descent loop
  /// @return Controller and State Trajectories over time horizon
  std::pair<arma::mat, arma::mat> ModelPredictiveControl(
    const arma::vec & x0, const arma::vec & u0, const unsigned int & num_steps,
    const unsigned int & max_iterations)
  {
    std::pair<arma::mat, arma::mat> trajectory, descentDirection;
    arma::mat X, U, X_new, U_new, zeta, vega, aT, bT;
    double DJ, J, J_new, gamma;
    unsigned int i, n;

    // Create Trajectory
    // std::cout << "Create Trajectory" << std::endl;
    U = arma::mat(u0.n_elem, num_steps, arma::fill::zeros);
    U.each_col() = u0;
    X = model.createTrajectory(x0, U, num_steps);

    // Get Cost of the trajectory
    // std::cout << "Cost of Trajectory" << std::endl;
    J = objectiveJ(X, U);

    DJ = 1000.0;
    i = 0;
    while (std::abs(DJ) > eps && i < max_iterations) {
      aT = calculate_aT(X);
      bT = calculate_bT(U);
      descentDirection = calculateZeta(X, U, aT, bT);
      zeta = descentDirection.first;
      vega = descentDirection.second;

      DJ = calculateDJ(descentDirection, aT, bT);
      J = objectiveJ(X, U);
      J_new = std::numeric_limits<double>::max();
      n = 1;
      gamma = beta;

      // Take a look at the algorithm for Armijo line search to check if this is correct
      while (J_new > J + alpha * gamma * DJ && n < 5) {
        U = U + gamma * vega;
        X = model.createTrajectory(x0, U);
        J_new = objectiveJ(X, U);
        n += 1;
        gamma = pow(beta, n);
      }

      // std::cout << "i: " << i << std::endl;
      // std::cout << "DJ: " << std::abs(DJ) << std::endl;
      // std::cout << "J: " << std::abs(J_new) << std::endl;
      // (X.col(X.n_cols - 1)).print("End X: ");

      trajectory = {X, U};
      i += 1;
    }

    return trajectory;
  }

protected:
  /// \brief Calculates zeta and vega matrix for controller
  /// \param Xt State trajectory over time Horizon
  /// \param Ut Control over time horizon
  /// \param aT aT Matrix
  /// \param bT bT Matrix
  /// \return Returns zeta and vega matrix for controller
  std::pair<arma::mat, arma::mat> calculateZeta(
    arma::mat & Xt, const arma::mat & Ut,
    const arma::mat & aT, const arma::mat & bT) const
  {
    std::pair<std::vector<arma::mat>, std::vector<arma::mat>> listPr = calculatePr(Xt, Ut, aT, bT);
    std::vector<arma::mat> Plist = listPr.first;
    std::vector<arma::mat> rlist = listPr.second;

    arma::mat zeta(Xt.n_rows, Xt.n_cols, arma::fill::zeros);
    arma::mat vega(Ut.n_rows, Ut.n_cols, arma::fill::zeros);

    arma::mat A = model.getA(Xt.col(0), Ut.col(0));
    arma::mat B = model.getB(Xt.col(0), Ut.col(0));

    // ERGODIC CONTROLLER: z(Xt.n_rows, arma::fill::zeros);
    // CONTROLLER:         -Plist[0] * rlist[0];
    // Fix: Use a virtual intialize z() function I guess
    // The only difference between ergodic controller and a regular controller is this function
    arma::vec z = get_z(Plist, rlist);
    //arma::vec z(Xt.n_rows, arma::fill::zeros);         

    arma::vec v = -R_mat.i() * B.t() * Plist[0] * z - R_mat.i() * B.t() * rlist[0] - R_mat.i() *
      bT.row(0).t();
    zeta.col(0) = z;
    vega.col(0) = v;

    arma::vec zdot;
    for (unsigned int i = 1; i < Xt.n_cols; i++) {
      A = model.getA(Xt.col(i), Ut.col(i));
      B = model.getB(Xt.col(i), Ut.col(i));

      zdot = A * z + B * v;
      z = z + dt * zdot;
      v = -R_mat.i() * B.t() * Plist[i] * z - R_mat.i() * B.t() * rlist[i] - R_mat.i() *
        bT.row(i).t();                                                                                              // + R_mat.i() * B.t() * Plist[i] * z

      zeta.col(i) = z;
      vega.col(i) = v;
    }

    std::pair<arma::mat, arma::mat> descDir = {zeta, vega};
    // std::cout << "calc Zeta Complete" << std::endl;
    return descDir;
  }

  /// \brief Calculates P and r lists for solving ricatti equations
  /// \param Xt State trajectory over time Horizon
  /// \param Ut Control over time horizon
  /// \param aT aT Matrix
  /// \param bT bT Matrix
  /// \return P and r lists over time horizon to calculate zeta
  std::pair<std::vector<arma::mat>, std::vector<arma::mat>> calculatePr(
    arma::mat Xt, arma::mat Ut,
    const arma::mat & aT,
    const arma::mat & bT) const
  {
    arma::mat P = P_mat;
    arma::mat r = r_mat;
    std::vector<arma::mat> Plist(Xt.n_cols, P);
    std::vector<arma::mat> rlist(Xt.n_cols, r);
    Plist[Xt.n_cols - 1] = P;
    rlist[Xt.n_cols - 1] = r;

    arma::mat Rinv = R_mat.i();

    int idx;
    arma::mat A, B, Pdot, rdot;
    for (unsigned int i = 0; i < Xt.n_cols - 2; i++) {
      idx = Xt.n_cols - 2 - i;

      A = model.getA(Xt.col(idx), Ut.col(idx));
      B = model.getB(Xt.col(idx), Ut.col(idx));

      Pdot = -P * A - A.t() * P + P * B * R_mat.i() * B.t() * P - Q_mat;           // + P * AT
      rdot = -(A - B * R_mat.i() * B.t() * P).t() * r - aT.row(idx).t() + P * B * R_mat.i() *
        bT.row(
        idx).t();

      P = P - dt * Pdot;
      r = r - dt * rdot;

      Plist[idx] = P;
      rlist[idx] = r;
    }

    std::pair<std::vector<arma::mat>, std::vector<arma::mat>> list_pair = {Plist, rlist};
    return list_pair;
  }

  /// \brief Calculates the objective function given Trajectory and Control [VIRTUAL]
  /// \param Xt State trajectory over time Horizon
  /// \param Ut Control over time horizon
  /// \return Objective value
  virtual double objectiveJ(const arma::mat & Xt, const arma::mat & Ut) const
  {
    UNUSED(Xt);
    UNUSED(Ut);
    return 0.0;
  }

  /// \brief Calculates absolute value of descent direction [VIRTUAL]
  /// \param zeta_pair zeta and vega matrix for controller
  /// \param at aT Matrix
  /// \param bt bT Matrix
  /// @return Descent direction as an double value
  virtual double calculateDJ(
    std::pair<arma::mat, arma::mat> const & zeta_pair,
    const arma::mat & aT, const arma::mat & bT)
  {
    UNUSED(aT);
    UNUSED(bT);
    UNUSED(zeta_pair);
    return 0.0;
  }

  /// \brief Calculates aT matrix [VIRTUAL]
  /// \param Xt State trajectory over time Horizon
  /// \return Retuns aT matrix
  virtual arma::mat calculate_aT(const arma::mat & Xt) const
  {
    return Xt;
  }

  /// \brief Calculates bT matrix [VIRTUAL]
  /// \param Ut Control over time horizon
  /// \return Retuns bT matrix
  virtual arma::mat calculate_bT(const arma::mat & Ut) const
  {
    return Ut;
  }

  /// \brief Gets the first iteration of the zeta matrix
  /// \param Plist P matrix over time trajectory
  /// \param rlist r matrix over time trajectory
  /// \return z matrix 
  virtual arma::vec get_z(const std::vector<arma::mat> & Plist, const std::vector<arma::mat> & rlist) const
  {
    UNUSED(Plist);
    UNUSED(rlist);
    arma::vec z(0, arma::fill::zeros);
    return z;
  }

  /// \brief Model following Concept Template
  ModelTemplate model;

  /// \brief Initial State vector at time t=0
  arma::vec x0;

  /// \brief Q Matrix (Trajectory Penalty)
  arma::mat Q_mat;

  /// \brief R Matrix (Control Penalty)
  arma::mat R_mat;

  /// \brief P Matrix (Final Trajectory Penalty)
  arma::mat P_mat;

  /// \brief r Matrix (Final Control Penalty)
  arma::mat r_mat;

  /// \brief Difference in time steps from model system
  double dt;

  /// \brief Number of iterations for time horizon
  unsigned int num_iter;

  /// \brief Max iteration for control descent
  unsigned int max_iter;

  /// \brief Alpha - Controller multiplier
  double alpha;

  /// \brief Beta - Controller multiplier for armijo line search
  double beta;

  /// \brief Epsilon - Convergence Value for Objective Function
  double eps;

};
}

#endif
