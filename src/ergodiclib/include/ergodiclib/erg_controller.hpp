#ifndef ERG_CON_INCLUDE_GUARD_HPP
#define ERG_CON_INCLUDE_GUARD_HPP
/// \file
/// \brief Ergodic COntroller


#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <limits>

#include <ergodiclib/base_controller.hpp>
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
/// \brief Ergodic Controller for a given dynamic system
/// \tparam ModelTemplate Template for Dynamic Model
template<class ModelTemplate>
class ergController : public BaseController<ModelTemplate>
{
public:
  ergController()
  : BaseController<ModelTemplate>::BaseController()
  {}

  /// \brief Constructor for Ergodic Controller
  /// \param ergodicMes Ergodic Measurement Class
  /// \param basis Ergodic Basis Class
  /// \param model_agent Model agent following Concept Template
  /// \param q_val q value (Ergodic trajectory penalty)
  /// \param Q Q Matrix (Trajectory Penalty)
  /// \param R R Matrix (Control Penalty)
  /// \param P P Matrix (Final Trajectory Penalty)
  /// \param r r Matrix (Final Control Penalty)
  /// \param max_iter_in Max iteration for control descent
  /// \param a Alpha - Controller multiplier
  /// \param b Beta - Controller multiplier for armijo line search
  /// \param e Epsilon - Convergence Value for Objective Function
  ergController(
    ErgodicMeasure & ergodicMes, fourierBasis & basis, ModelTemplate model_agent,
    double q_val, arma::mat Q, arma::mat R, arma::mat P, arma::mat r,
    int max_iter_in, double a, double b, double e)
  : ergodicMeasure(ergodicMes),
    Basis(basis),
    BaseController<ModelTemplate>::BaseController(model_agent, Q, R, P, r, max_iter_in, a, b, e),
    q(q_val)
  {
    tf = model_agent.tf;
  }

  /// \brief Constructor for Ergodic Controller
  /// \param demonstrations Vector of Demonstrations - Trajectories
  /// \param demo_posneg Demonstration weights - length of demonstrations
  /// \param demo_weights Demonstration weights - length of demonstrations
  /// \param K_coeff Size of Series Coefficient
  /// \param L_dim Size of boundaries for dimensions
  /// \param dt_demo time difference between each state in the trajectory
  /// \param L_dim Length of Dimensions
  /// \param num_dim Number of Dimensions
  /// \param K Fourier Series Coefficient
  /// \param model_agent Model agent following Concept Template
  /// \param q_val q value (Ergodic trajectory penalty)
  /// \param Q Q Matrix (Trajectory Penalty)
  /// \param R R Matrix (Control Penalty)
  /// \param P P Matrix (Final Trajectory Penalty)
  /// \param r r Matrix (Final Control Penalty)
  /// \param max_iter_in Max iteration for control descent
  /// \param a Alpha - Controller multiplier
  /// \param b Beta - Controller multiplier for armijo line search
  /// \param e Epsilon - Convergence Value for Objective Function
  ergController(
    std::vector<arma::mat> demonstrations, std::vector<int> demo_posneg,
    std::vector<double> demo_weights,
    std::vector<std::pair<double, double>> L_dim, int num_dim, int K,
    ModelTemplate model_agent, double q_val, arma::mat Q,
    arma::mat R, arma::mat P, arma::mat r, int max_iter_in,
    double a, double b, double e)
  : BaseController<ModelTemplate>::BaseController(model_agent, Q, R, P, r, max_iter_in, a, b, e),
    q(q_val)
  {
    Basis = fourierBasis(L_dim, num_dim, K);
    ergodicMeasure =
      ErgodicMeasure(demonstrations, demo_posneg, demo_weights, model_agent.dt, Basis);
    tf = model_agent.tf;
  }

private:
  /// \brief Calculates absolute value of descent direction
  /// \param zeta_pair zeta and vega matrix for controller
  /// \param at aT Matrix
  /// \param bt bT Matrix
  /// @return Descent direction as an double value
  virtual double calculateDJ(
    std::pair<arma::mat, arma::mat> const & zeta_pair, const arma::mat & at,
    const arma::mat & bt);

  /// \brief Calculates the objective function given Trajectory and Control
  /// \param Xt State trajectory over time Horizon
  /// \param Ut Control over time horizon
  /// \return Objective value
  virtual double objectiveJ(const arma::mat & Xt, const arma::mat & Ut);

  /// \brief Calculates aT matrix
  /// \param Xt State trajectory over time Horizon
  /// \return Retuns aT matrix
  virtual arma::mat calculate_aT(const arma::mat & Xt) const;

  /// \brief Calculates bT matrix
  /// \param Ut Control over time horizon
  /// \return Retuns bT matrix
  virtual arma::mat calculate_bT(const arma::mat & Ut) const;

  /// \param ergodicMes Ergodic Measurement Class
  ErgodicMeasure & ergodicMeasure;

  /// \param basis Ergodic Basis Class
  fourierBasis & Basis;

  /// \param q_val q value (Ergodic trajectory penalty)
  double q;

  /// \brief Final Time
  double tf;
};

template<class ModelTemplate>
double ergController<ModelTemplate>::calculateDJ(
  std::pair<arma::mat, arma::mat> const & zeta_pair,
  const arma::mat & aT, const arma::mat & bT)
{
  const unsigned int num_iter = aT.n_rows;
  arma::vec DJ(num_iter, 1, arma::fill::zeros);
  arma::mat zeta = zeta_pair.first;
  arma::mat vega = zeta_pair.second;

  // Construct DJ Vector
  for (unsigned int i = 0; i < num_iter; i++) {
    DJ.row(i) = aT.row(i) * zeta.col(i) + bT.row(i) * vega.col(i);
  }

  // integrate and return J
  double DJ_integral = integralTrapz(DJ, BaseController<ModelTemplate>::dt);
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
    controlCost = Ut_i.t() * BaseController<ModelTemplate>::R_mat * Ut_i;
    trajecJ(i) = controlCost(0, 0);
  }
  double trajec_cost = ergodiclib::integralTrapz(trajecJ, BaseController<ModelTemplate>::dt);

  std::cout << "ergodicCost: " << ergodicCost << ", ctrlCost: " << trajec_cost << std::endl;
  double final_cost = trajec_cost + ergodicCost;
  return final_cost;
}

template<class ModelTemplate>
arma::mat ergController<ModelTemplate>::calculate_aT(const arma::mat & Xt) const
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
arma::mat ergController<ModelTemplate>::calculate_bT(const arma::mat & Ut) const
{
  arma::mat bT(Ut.n_cols, Ut.n_rows, arma::fill::zeros);
  for (unsigned int i = 0; i < Ut.n_cols; i++) {
    bT.row(i) = Ut.col(i).t() * BaseController<ModelTemplate>::R_mat;
  }
  return bT;
}

}


#endif
