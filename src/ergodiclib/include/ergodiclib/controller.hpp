#ifndef CONTROLLER_INCLUDE_GUARD_HPP
#define CONTROLLER_INCLUDE_GUARD_HPP
/// \file
/// \brief Contains a Controller class for an iLQR controller

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <ergodiclib/num_utils.hpp>
#include <ergodiclib/model.hpp>
#include <ergodiclib/cartpole.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

namespace ergodiclib
{
/// \brief iLQR Controller for a given dynamic system
/// \tparam ModelTemplate Template for Dynamic Model
template<class ModelTemplate>
class ilqrController
{
public:
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
  ilqrController(
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

  /// \brief Begins iLQR controller - returns none
  void iLQR();

private:
  /// \brief Calculates the objective function given Trajectory and Control
  /// \param Xt State trajectory over time Horizon
  /// \param Ut Control over time horizon
  /// \return Objective value
  double objectiveJ(const arma::mat & Xt, const arma::mat & Ut) const;

  /// \brief Calculates the objective value of the trajectory
  /// \param Xt State trajectory over time Horizon
  /// \param Ut Control over time horizon
  /// \return Objective value for trajectory
  double trajectoryJ(const arma::mat & Xt, const arma::mat & Ut) const;

  /// \brief Calculates zeta and vega matrix for controller
  /// \param Xt State trajectory over time Horizon
  /// \param Ut Control over time horizon
  /// \return Returns zeta and vega matrix for controller
  std::pair<arma::mat, arma::mat> calculateZeta(const arma::mat & Xt, const arma::mat & Ut) const;

  /// \brief Calculates P and r lists for solving ricatti equations
  /// \param Xt State trajectory over time Horizon
  /// \param Ut Control over time horizon
  /// \param aT aT Matrix
  /// \param bT bT Matrix
  /// \return P and r lists over time horizon to calculate zeta
  std::pair<std::vector<arma::mat>, std::vector<arma::mat>> calculatePr(
    const arma::mat & Xt,
    const arma::mat & Ut,
    const arma::mat & aT,
    const arma::mat & bT) const;

  /// \brief Calculates aT matrix
  /// \param Xt State trajectory over time Horizon
  /// \return Retuns aT matrix
  arma::mat calculate_aT(const arma::mat & Xt) const;

  /// \brief Calculates bT matrix
  /// \param Ut Control over time horizon
  /// \return Retuns bT matrix
  arma::mat calculate_bT(const arma::mat & Ut) const;

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

template<class ModelTemplate>
void ilqrController<ModelTemplate>::iLQR()
{
  std::pair<arma::mat, arma::mat> trajectory, descentDirection;
  arma::mat X, U, X_new, U_new, zeta, vega;
  double J, J_new, gamma;
  std::cout << "create trajec" << std::endl;
  trajectory = model.createTrajectory();
  X = trajectory.first;
  U = trajectory.second;
  J = objectiveJ(X, U);

  (X.col(X.n_cols - 1)).print("End X0: ");
  std::cout << "obj: " << J << std::endl;

  int n = 0;
  unsigned int i = 0;
  gamma = beta; // This is not correct 
  std::cout << "Start loop" << std::endl;
  std::cout << "abs_J" << abs(J) << std::endl;
  std::cout << "eps" << eps << std::endl;
  while (abs(J) > eps && i < max_iter) {
    std::cout << "calc zeta" << std::endl;
    descentDirection = calculateZeta(X, U);
    zeta = descentDirection.first;
    vega = descentDirection.second;

    n = 0;
    J_new = J + J;
    while (J_new > J + alpha * gamma * trajectoryJ(X, U) && n < 10) { // get rid of n < 10, fix trajectoryJ(X, U) to descent dir
      U_new = U + gamma * vega;
      X_new = model.createTrajectory(x0, U_new);

      J_new = objectiveJ(X_new, U_new);

      n += 1;
      gamma = pow(beta, n);

      X = X_new;
      U = U_new;

      //std::cout << "J_new (Desc Dir): " << abs(J_new) << std::endl;
    }
    J = J_new;
    trajectory = {X, U};

    i += 1;

    (X.col(X.n_cols - 1)).print("End X: ");
    std::cout << "i: " << i << std::endl;
    std::cout << "J: " << abs(J) << std::endl;
  }
  std::string x_file = "trajectory_out";
  arma::mat XT = X.t();
  XT.save(x_file, arma::csv_ascii);
  arma::mat UT = U.t();
  std::string u_file = "control_out";
  UT.save(u_file, arma::csv_ascii);
}

template<class ModelTemplate>
double ilqrController<ModelTemplate>::objectiveJ(const arma::mat & Xt, const arma::mat & Ut) const
{
  int X_cols = Xt.n_cols - 1;
  arma::vec x_tf = Xt.col(X_cols);
  arma::mat finalcost_mat = x_tf.t() * P_mat * x_tf;
  double final_cost = finalcost_mat(0, 0);

  double trajectory_cost = trajectoryJ(Xt, Ut);
  double cost = 0.5 * (final_cost + trajectory_cost);

  std::cout << "final cost: " << final_cost << std::endl;
  std::cout << "trajectory cost: " << trajectory_cost << std::endl;

  return cost;
}

template<class ModelTemplate>
double ilqrController<ModelTemplate>::trajectoryJ(const arma::mat & Xt, const arma::mat & Ut) const
{
  arma::vec trajecJ(Xt.n_cols, arma::fill::zeros);

  arma::vec Xt_i, Ut_i;
  arma::mat cost;
  for (unsigned int i = 0; i < Xt.n_cols; i++) {
    Xt_i = Xt.col(i);
    Ut_i = Ut.col(i);
    cost = Xt_i.t() * Q_mat * Xt_i + Ut_i.t() * R_mat * Ut_i;
    trajecJ(i) = cost(0, 0);
  }

  double trajec_cost = ergodiclib::integralTrapz(trajecJ, dt);
  return trajec_cost;
}

template<class ModelTemplate>
std::pair<arma::mat, arma::mat> ilqrController<ModelTemplate>::calculateZeta(
  const arma::mat & Xt,
  const arma::mat & Ut)
const
{
  arma::mat aT = calculate_aT(Xt);
  arma::mat bT = calculate_bT(Ut);

  std::pair<std::vector<arma::mat>, std::vector<arma::mat>> listPr = calculatePr(Xt, Ut, aT, bT);
  std::vector<arma::mat> Plist = listPr.first;
  std::vector<arma::mat> rlist = listPr.second;

  arma::mat zeta(Xt.n_rows, Xt.n_cols, arma::fill::zeros);
  arma::mat vega(Ut.n_rows, Ut.n_cols, arma::fill::zeros);

  arma::mat A = model.getA(Xt.col(0), Ut.col(0));
  arma::mat B = model.getB(Xt.col(0), Ut.col(0));
  arma::vec z = -Plist[0] * rlist[0];

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
      bT.row(i).t();

    zeta.col(i) = z;
    vega.col(i) = v;
  }

  std::pair<arma::mat, arma::mat> descDir = {zeta, vega};
  return descDir;
}

template<class ModelTemplate>
std::pair<std::vector<arma::mat>,
  std::vector<arma::mat>> ilqrController<ModelTemplate>::calculatePr(
  const arma::mat & Xt,
  const arma::mat & Ut,
  const arma::mat & aT,
  const arma::mat & bT) const
{
  std::cout << "calc PR" << std::endl;
  arma::mat P = P_mat;
  arma::mat r = r_mat;
  std::vector<arma::mat> Plist(Xt.n_cols, P);
  std::vector<arma::mat> rlist(Xt.n_cols, r);
  Plist[Xt.n_cols - 1] = P;
  rlist[Xt.n_cols - 1] = r;

  arma::mat Rinv = R_mat.i();

  unsigned int idx;
  arma::mat A, B, Pdot, rdot;
  for (unsigned int i = 0; i < Xt.n_cols - 2; i++) {
    idx = Xt.n_cols - 2 - i;

    A = model.getA(Xt.col(idx), Ut.col(idx));
    B = model.getB(Xt.col(idx), Ut.col(idx));

    Pdot = -P * A - A.t() * P + P * B * R_mat.i() * B.t() * P - Q_mat;
    rdot = -(A - B * R_mat.i() * B.t() * P).t() * r - aT.row(idx).t() + P * B * R_mat.i() * bT.row(
      idx).t();

    P = P - dt * Pdot;
    r = r - dt * rdot;

    Plist[idx] = P;
    rlist[idx] = r;
    //A.print("A: ");
    //B.print("B: ");
    //P.print("P: ");
    //r.print("r: ");
  }

  std::pair<std::vector<arma::mat>, std::vector<arma::mat>> list_pair = {Plist, rlist};
  return list_pair;
}

template<class ModelTemplate>
arma::mat ilqrController<ModelTemplate>::calculate_aT(const arma::mat & Xt) const
{
  arma::mat aT(Xt.n_cols, Xt.n_rows, arma::fill::zeros);
  for (unsigned int i = 0; i < Xt.n_cols; i++) {
    aT.row(i) = Xt.col(i).t() * Q_mat;
  }
  return aT;
}

template<class ModelTemplate>
arma::mat ilqrController<ModelTemplate>::calculate_bT(const arma::mat & Ut) const
{
  arma::mat bT(Ut.n_cols, Ut.n_rows, arma::fill::zeros);
  for (unsigned int i = 0; i < Ut.n_cols; i++) {
    bT.row(i) = Ut.col(i).t() * R_mat;
  }
  return bT;
}
}

#endif
