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
    ErgodicMeasure ergodicMes, fourierBasis basis, ModelTemplate model_agent,
    double q_val, arma::mat R_mat, arma::mat Q_mat, double t0_val, double tf_val,
    double dt_val, double eps_val, double beta_val)
  : ergodicMeasure(ergodicMes),
    Basis(basis),
    model(model_agent),
    q(q_val),
    R(R_mat),
    Q(Q_mat),
    t0(t0_val),
    tf(tf_val),
    dt(dt_val),
    eps(eps_val),
    beta(beta_val)
  {
    n_iter = (int) ((tf - t0) / dt);
  }
  double DJ(std::pair<arma::mat, arma::mat> zeta_pair, const arma::mat & at, const arma::mat & bt);

  int gradient_descent(arma::vec x0);

private:
  std::pair<arma::mat, arma::mat> ergController<ModelTemplate>::calculateZeta(const arma::mat & Xt, const arma::mat & Ut) const;

  std::pair<std::vector<arma::mat>, std::vector<arma::mat>> calculatePr(arma::mat Xt, arma::mat Ut, const arma::mat & aT, const arma::mat & bT) const;

  arma::mat calculate_aT(const arma::mat & x_mat);

  arma::mat calculate_bT(const arma::mat & Ut);

  ErgodicMeasure ergodicMeasure;
  fourierBasis Basis;
  ModelTemplate model;
  double q;
  arma::mat P_mat;
  arma::mat r_mat;
  arma::mat R_mat;
  arma::mat Q;
  double t0;
  double tf;
  double dt;
  double eps;
  double beta;
  int n_iter;
};

template<class ModelTemplate>
int ergController<ModelTemplate>::gradient_descent(arma::vec x0)
{
  std::pair<arma::mat, arma::mat> xtut = model.createTrajectory();
  arma::mat xt = xtut.first;
  arma::mat ut = xtut.second;
  model.setx0(x0);

  double dj = 2e31;

  arma::mat at, bt;
  std::pair<arma::mat, arma::mat> zeta_pair;
  std::pair<std::vector<arma::mat>, std::vector<arma::mat>> listPr;
  while (abs(dj) > eps) {
    at = calculate_aT(xt);
    bt = calculate_bT(ut);

    listPr = calculatePr(xt, ut, at, bt);

    zeta_pair = descentDirection(xt, ut, listPr.first, listPr.second, bt);

    dj = DJ(zeta_pair, at, bt);

    arma::mat vega = zeta_pair.second;
    ut = ut + beta * vega;
    xt = model.createTrajectory(x0, ut);
  }

  return 1;
}template<class ModelTemplate>
int ergController<ModelTemplate>::gradient_descent(arma::vec x0)
{
  std::pair<arma::mat, arma::mat> xtut = model.createTrajectory();
  arma::mat xt = xtut.first;
  arma::mat ut = xtut.second;
  model.setx0(x0);

  double dj = 2e31;

  arma::mat at, bt;
  std::pair<arma::mat, arma::mat> zeta_pair;
  std::pair<std::vector<arma::mat>, std::vector<arma::mat>> listPr;
  while (abs(dj) > eps) {
    at = calculate_aT(xt);
    bt = calculate_bT(ut);

    listPr = calculatePr(xt, ut, at, bt);

    zeta_pair = descentDirection(xt, ut, listPr.first, listPr.second, bt);

    dj = DJ(zeta_pair, at, bt);

    arma::mat vega = zeta_pair.second;
    ut = ut + beta * vega;
    xt = model.createTrajectory(x0, ut);
  }

  return 1;
}

template<class ModelTemplate>
double ergController<ModelTemplate>::DJ(std::pair<arma::mat, arma::mat> zeta_pair, const arma::mat & at, const arma::mat & bt)
{
  arma::vec J(n_iter, 1, arma::fill::zeros);
  arma::mat zeta = zeta_pair.first;
  arma::mat vega = zeta_pair.second;

  arma::mat a_T, b_T;

  for (int i = 0; i < n_iter; i++) {
    a_T = at.row(i).t();
    b_T = bt.row(i).t();

    J.row(i) = a_T * zeta.row(i) + b_T * vega.row(i);
  }

  // integrate and return J
  double J_integral = integralTrapz(J, dt);
  return J_integral;
}

template<class ModelTemplate>
double ergController<ModelTemplate>::DJ(std::pair<arma::mat, arma::mat> zeta_pair, const arma::mat & at, const arma::mat & bt)
{
  arma::vec J(n_iter, 1, arma::fill::zeros);
  arma::mat zeta = zeta_pair.first;
  arma::mat vega = zeta_pair.second;

  arma::mat a_T, b_T;

  for (int i = 0; i < n_iter; i++) {
    a_T = at.row(i).t();
    b_T = bt.row(i).t();

    J.row(i) = a_T * zeta.row(i) + b_T * vega.row(i);
  }

  // integrate and return J
  double J_integral = integralTrapz(J, dt);
  return J_integral;
}

template<class ModelTemplate>
std::pair<arma::mat, arma::mat> ergController<ModelTemplate>::calculateZeta(
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
    v = -R_mat.i() * B.t() * Plist[i] * z - R_mat.i() * B.t() * rlist[i] - R_mat.i() * bT.row(i).t();

    zeta.col(i) = z;
    vega.col(i) = v;
  }

  std::pair<arma::mat, arma::mat> descDir = {zeta, vega};
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

  arma::mat A, B, Pdot, rdot;
  for (unsigned int i = 0; i < Xt.n_cols - 2; i++) {
    idx = Xt.n_cols - 2 - i;

    A = model.getA(Xt.col(idx), Ut.col(idx));
    B = model.getB(Xt.col(idx), Ut.col(idx));

    Pdot = -P * A - A.t() * P + P * B * R_mat.i() * B.t() * P - Q_mat;
    rdot = -(A - B * R_mat.i() * B.t() * P).t() * r - aT.row(idx).t() + P * B * R_mat.i() * bT.row(idx).t();

    P = P - dt * Pdot;
    r = r - dt * rdot;

    Plist[idx] = P;
    rlist[idx] = r;
  }

  std::pair<std::vector<arma::mat>, std::vector<arma::mat>> list_pair = {Plist, rlist};
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
  arma::vec dfk;
  for (unsigned int i = 0; i < K_series.size(); i++) {
    double ck = ergodicMeasure.calculateCk(Xt, K_series[i], i);

    for (unsigned int t = 0; t < Xt.n_cols; t++) {
      dfk = Basis.calculateDFk(Xt.col(t), K_series[i], i);
      ak_mat = lambda[i] * (2 * (ck - phi[i]) * ((1 / tf) * dfk));
      a_mat.row(t) += ak_mat;
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
    bT.row(i) = Ut.col(i).t() * R;
  }
  return bT;
}

}


#endif
