#ifndef ERG_CON_INCLUDE_GUARD_HPP
#define ERG_CON_INCLUDE_GUARD_HPP
/// \file
/// \brief


#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <ergodiclib/ergodic_measure.hpp>
#include <ergodiclib/num_utils.hpp>
#include <ergodiclib/fourier_basis.hpp>
#include <ergodiclib/model.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

namespace ergodiclib
{
template<class ModelTemplate>
class ergController
{
public:
  ergController(
    ErgodicMeasure& ergodicMes, fourierBasis& basis, ModelTemplate model_agent,
    double q_val, arma::mat Q, arma::mat R, arma::mat P, arma::mat r,
    int max_iter_in, double a, double b, double e)
  : ergodicMeasure(ergodicMes),
    Basis(basis),
    model(model_agent),
    q(q_val),
    Q_mat(Q),
    R_mat(R),
    P_mat(P),
    r_mat(r),
    max_iter(max_iter_in),
    alpha(a),
    beta(b),
    eps(e)
  {
    x0 = model_agent.x0;
    dt = model_agent.dt;
    tf = model_agent.tf;
    n_iter = (int) ((model_agent.tf - model_agent.t0) / dt);
   }

  void iLQR();

private:
  double calculateDJ(std::pair<arma::mat, arma::mat> const &zeta_pair, const arma::mat & at, const arma::mat & bt);
 
  double objectiveJ(const arma::mat & Xt, const arma::mat & Ut);

  std::pair<arma::mat, arma::mat> calculateZeta(const arma::mat & Xt, const arma::mat & Ut, const arma::mat & aT, const arma::mat & bT) const;

  std::pair<std::vector<arma::mat>, std::vector<arma::mat>> calculatePr(arma::mat Xt, arma::mat Ut, const arma::mat & aT, const arma::mat & bT) const;

  arma::mat calculate_aT(const arma::mat & x_mat);

  arma::mat calculate_bT(const arma::mat & Ut);

  ErgodicMeasure& ergodicMeasure;
  fourierBasis& Basis;
  ModelTemplate model;
  double q;

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

  /// \brief Final Time
  double tf;

  /// \brief Number of iterations for time horizon
  unsigned int n_iter;

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
void ergController<ModelTemplate>::iLQR()
{
  // Create variables for iLQR loop
  std::pair<arma::mat, arma::mat> trajectory, descentDirection;
  arma::mat X, U, zeta, vega, aT, bT;
  double DJ, J, J_new, gamma;
  unsigned int i, n;

  // Get an initial trajectory
  std::cout << "create trajec" << std::endl;
  trajectory = model.createTrajectory();
  X = trajectory.first;
  U = trajectory.second;
  DJ = 1000.0;

  i = 0;
  while (std::abs(DJ) > eps && i < max_iter) {
    aT = calculate_aT(X);
    //aT.print("aT");
    bT = calculate_bT(U);
    //bT.print("bT");
    descentDirection = calculateZeta(X, U, aT, bT);
    zeta = descentDirection.first;
    vega = descentDirection.second;

    std::cout << "Calc DJ" << std::endl;
    DJ = calculateDJ(descentDirection, aT, bT);
    std::cout << "Calc J" << std::endl;
    J = objectiveJ(X, U);
    J_new = 2e31; 
    gamma = beta;

    n = 1;
    std::cout << "Armijo, J: " << J << std::endl;
    while (J_new > J + alpha * gamma * DJ && n < 2) {
      U = U + gamma * vega;
      X = model.createTrajectory(x0, U);
      J_new = objectiveJ(X, U);
      gamma = pow(beta, n+1);
      n += 1;
      std::cout << "n: " << n << ", J: " << std::abs(J_new) << std::endl;
    }
    trajectory = {X, U};
    i += 1;

    std::cout << "i: " << i << std::endl;
    std::cout << "DJ: " << std::abs(DJ) << std::endl;
    std::cout << "J: " << std::abs(J) << std::endl;
    if (std::isnan(std::abs(DJ))) {
      DJ = 1000.0;
    }
    (X.col(X.n_cols - 1)).print("End X: ");
  }
  std::string x_file = "erg_trajectory_out";
  arma::mat XT = X.t();
  XT.save(x_file, arma::csv_ascii);
  arma::mat UT = U.t();
  std::string u_file = "erg_control_out";
  UT.save(u_file, arma::csv_ascii);
}


template<class ModelTemplate>
double ergController<ModelTemplate>::calculateDJ(std::pair<arma::mat, arma::mat> const &zeta_pair, const arma::mat & aT, const arma::mat & bT)
{
  arma::vec DJ(n_iter, 1, arma::fill::zeros);
  arma::mat zeta = zeta_pair.first;
  arma::mat vega = zeta_pair.second;

  // Construct DJ Vector
  for (unsigned int i = 0; i < n_iter; i++) {
    DJ.row(i) = aT.row(i) * zeta.col(i) + bT.row(i) * vega.col(i);
  }

  // integrate and return J
  double DJ_integral = integralTrapz(DJ, dt);
  return DJ_integral;
}

template<class ModelTemplate>
double ergController<ModelTemplate>::objectiveJ(const arma::mat & Xt, const arma::mat & Ut) 
{
  const std::vector<std::vector<int>> K_series = Basis.get_K_series();
  arma::mat lambda = ergodicMeasure.get_LambdaK();
  arma::mat phi = ergodicMeasure.get_PhiK();

  double ergodicCost = 0.0;

  double ck;
  for (unsigned int i = 0; i < K_series.size(); i++) {
      ck = ergodicMeasure.calculateCk(Xt, K_series[i], i);
      ergodicCost += lambda[i] * pow(ck - phi[i], 2);
  }
  ergodicCost = q * ergodicCost;

  arma::vec trajecJ(Ut.n_cols, arma::fill::zeros);
  arma::vec Ut_i;
  arma::mat controlCost;
  for (unsigned int i = 0; i < Ut.n_cols; i++) {
    Ut_i = Ut.col(i);
    controlCost = Ut_i.t() * R_mat * Ut_i;
    trajecJ(i) = controlCost(0, 0);
  }
  double trajec_cost = ergodiclib::integralTrapz(trajecJ, dt);

  double final_cost = trajec_cost + ergodicCost;
  return final_cost;
}

template<class ModelTemplate>
std::pair<arma::mat, arma::mat> ergController<ModelTemplate>::calculateZeta(
  const arma::mat & Xt,
  const arma::mat & Ut,
  const arma::mat & aT,
  const arma::mat & bT)
const
{
  std::cout << "calc Zeta" << std::endl;
  std::pair<std::vector<arma::mat>, std::vector<arma::mat>> listPr = calculatePr(Xt, Ut, aT, bT);
  std::vector<arma::mat> Plist = listPr.first;
  std::vector<arma::mat> rlist = listPr.second;

  arma::mat zeta(Xt.n_rows, Xt.n_cols, arma::fill::zeros);
  arma::mat vega(Ut.n_rows, Ut.n_cols, arma::fill::zeros);

  arma::mat A = model.getA(Xt.col(0), Ut.col(0));
  arma::mat B = model.getB(Xt.col(0), Ut.col(0));
  arma::vec z(Xt.n_rows, arma::fill::zeros);

  arma::vec v = -R_mat.i() * B.t() * Plist[0] * z - R_mat.i() * B.t() * rlist[0] - R_mat.i() * bT.row(0).t(); 
  zeta.col(0) = z;
  vega.col(0) = v;

  arma::vec zdot;
  for (unsigned int i = 1; i < Xt.n_cols; i++) {
    A = model.getA(Xt.col(i), Ut.col(i));
    B = model.getB(Xt.col(i), Ut.col(i));

    zdot = A * z + B * v;
    z = z + dt * zdot;
    v = -R_mat.i() * B.t() * Plist[i] * z - R_mat.i() * B.t() * rlist[i] - R_mat.i() * bT.row(i).t(); // + R_mat.i() * B.t() * Plist[i] * z

    zeta.col(i) = z;
    vega.col(i) = v;
  }

  std::pair<arma::mat, arma::mat> descDir = {zeta, vega};
  std::cout << "calc Zeta Complete" << std::endl;
  return descDir;
}


template<class ModelTemplate>
std::pair<std::vector<arma::mat>, std::vector<arma::mat>> ergController<ModelTemplate>::calculatePr(
  arma::mat Xt, arma::mat Ut, const arma::mat & aT, const arma::mat & bT) const
{
  std::cout << "calc PR" << std::endl;
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

    Pdot = -P * A - A.t() * P + P * B * R_mat.i() * B.t() * P - Q_mat; // + P * AT
    rdot = -(A - B * R_mat.i() * B.t() * P).t() * r - aT.row(idx).t() + P * B * R_mat.i() * bT.row(idx).t();

    P = P - dt * Pdot;
    r = r - dt * rdot;

    Plist[idx] = P;
    rlist[idx] = r;
  }

  std::pair<std::vector<arma::mat>, std::vector<arma::mat>> list_pair = {Plist, rlist};
  std::cout << "calc PR Complete" << std::endl;
  return list_pair;
}

template<class ModelTemplate>
arma::mat ergController<ModelTemplate>::calculate_aT(const arma::mat &Xt)
{
  arma::mat a_mat(Xt.n_cols, Xt.n_rows, arma::fill::zeros);

  const std::vector<std::vector<int>> K_series = Basis.get_K_series();
  arma::mat lambda = ergodicMeasure.get_LambdaK();
  arma::mat phi = ergodicMeasure.get_PhiK();

  arma::rowvec ak_mat(Xt.n_rows, arma::fill::zeros);
  arma::rowvec dfk(Xt.n_rows, arma::fill::zeros);
  for (unsigned int i = 0; i < K_series.size(); i++) {
    double ck = ergodicMeasure.calculateCk(Xt, K_series[i], i);

    for (unsigned int t = 0; t < Xt.n_cols; t++) {
      dfk = Basis.calculateDFk(Xt.col(t), K_series[i], i);
      ak_mat = lambda[i] * (2 * (ck - phi[i]) * ((1 / tf) * dfk));
      a_mat.row(t) = a_mat.row(t) + ak_mat;
    }
  }

  a_mat = q * a_mat;
  return a_mat;
}

template<class ModelTemplate>
arma::mat ergController<ModelTemplate>::calculate_bT(const arma::mat & Ut)
{
  arma::mat bT(Ut.n_cols, Ut.n_rows, arma::fill::zeros);
  for (unsigned int i = 0; i < Ut.n_cols; i++) {
    bT.row(i) = Ut.col(i).t() * R_mat;
  }
  return bT;
}

}


#endif
