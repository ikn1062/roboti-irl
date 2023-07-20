#ifndef SIMPLE_CONTROLLER_INCLUDE_GUARD_HPP
#define SIMPLE_CONTROLLER_INCLUDE_GUARD_HPP

/// \file
/// \brief Contains a Simple Controller class for an iLQR controller

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <limits>

#include <ergodiclib/num_utils.hpp>
#include <ergodiclib/model.hpp>
#include <ergodiclib/base_controller.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

namespace ergodiclib
{
/// \brief Simple iLQR Controller for a given dynamic system
/// \tparam ModelTemplate Template for Dynamic Model
template<class ModelTemplate>
class SimpleController : public BaseController<ModelTemplate>
{
public:
  /// \brief Base Constructor for Simple Controller Class
  SimpleController()
  : BaseController<ModelTemplate>::BaseController()
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
  SimpleController(
    ModelTemplate model_in, arma::mat Q, arma::mat R, arma::mat P, arma::mat r,
    unsigned int max_iter_in, double a, double b, double e)
  : BaseController<ModelTemplate>::BaseController(model_in, Q, R, P, r, max_iter_in, a, b, e)
  {}

private:
  /// \brief Calculates the objective function given Trajectory and Control
  /// \param Xt State trajectory over time Horizon
  /// \param Ut Control over time horizon
  /// \return Objective value
  virtual double objectiveJ(const arma::mat & Xt, const arma::mat & Ut) const;

  /// \brief Calculates the objective value of the trajectory
  /// \param Xt State trajectory over time Horizon
  /// \param Ut Control over time horizon
  /// \return Objective value for trajectory
  double trajectoryJ(const arma::mat & Xt, const arma::mat & Ut) const;

  /// \brief Calculates absolute value of descent direction
  /// \param zeta_pair zeta and vega matrix for controller
  /// \param at aT Matrix
  /// \param bt bT Matrix
  /// @return Descent direction as an double value
  virtual double calculateDJ(
    std::pair<arma::mat, arma::mat> const & zeta_pair, const arma::mat & at,
    const arma::mat & bt);

  /// \brief Calculates aT matrix
  /// \param Xt State trajectory over time Horizon
  /// \return Retuns aT matrix
  virtual arma::mat calculate_aT(const arma::mat & Xt) const;

  /// \brief Calculates bT matrix
  /// \param Ut Control over time horizon
  /// \return Retuns bT matrix
  virtual arma::mat calculate_bT(const arma::mat & Ut) const;

  /// \brief Gets the first iteration of the zeta matrix
  /// \param Plist P matrix over time trajectory
  /// \param rlist r matrix over time trajectory
  /// \return z matrix 
  virtual arma::vec get_z(const std::vector<arma::mat> & Plist, const std::vector<arma::mat> & rlist) const;
};

template<class ModelTemplate>
double SimpleController<ModelTemplate>::objectiveJ(const arma::mat & Xt, const arma::mat & Ut) const
{
  int X_cols = Xt.n_cols - 1;
  arma::vec x_tf = Xt.col(X_cols);
  arma::mat finalcost_mat = x_tf.t() * BaseController<ModelTemplate>::P_mat * x_tf;
  double final_cost = finalcost_mat(0, 0);

  double trajectory_cost = trajectoryJ(Xt, Ut);
  double cost = 0.5 * (final_cost + trajectory_cost);

  //std::cout << "final cost: " << final_cost << std::endl;
  //std::cout << "trajectory cost: " << trajectory_cost << std::endl;

  return cost;
}

template<class ModelTemplate>
double SimpleController<ModelTemplate>::trajectoryJ(
  const arma::mat & Xt,
  const arma::mat & Ut) const
{
  arma::vec trajecJ(Xt.n_cols, arma::fill::zeros);

  arma::vec Xt_i, Ut_i;
  arma::mat cost;
  for (unsigned int i = 0; i < Xt.n_cols; i++) {
    Xt_i = Xt.col(i);
    Ut_i = Ut.col(i);
    cost = Xt_i.t() * BaseController<ModelTemplate>::Q_mat * Xt_i + Ut_i.t() *
      BaseController<ModelTemplate>::R_mat * Ut_i;
    trajecJ(i) = cost(0, 0);
  }

  double trajec_cost = ergodiclib::integralTrapz(trajecJ, BaseController<ModelTemplate>::dt);
  return trajec_cost;
}

template<class ModelTemplate>
double SimpleController<ModelTemplate>::calculateDJ(
  std::pair<arma::mat, arma::mat> const & zeta_pair,
  const arma::mat & aT, const arma::mat & bT)
{
  const unsigned int num_it = aT.n_rows;
  arma::vec DJ(num_it, 1, arma::fill::zeros);
  arma::mat zeta = zeta_pair.first;
  arma::mat vega = zeta_pair.second;

  // Construct DJ Vector
  for (unsigned int i = 0; i < num_it; i++) {
    DJ.row(i) = aT.row(i) * zeta.col(i) + bT.row(i) * vega.col(i);
  }

  // integrate and return J
  double DJ_integral = integralTrapz(DJ, BaseController<ModelTemplate>::dt);
  return DJ_integral;
}

template<class ModelTemplate>
arma::mat SimpleController<ModelTemplate>::calculate_aT(const arma::mat & Xt) const
{
  arma::mat aT(Xt.n_cols, Xt.n_rows, arma::fill::zeros);
  for (unsigned int i = 0; i < Xt.n_cols; i++) {
    aT.row(i) = Xt.col(i).t() * BaseController<ModelTemplate>::Q_mat;
  }
  return aT;
}

template<class ModelTemplate>
arma::mat SimpleController<ModelTemplate>::calculate_bT(const arma::mat & Ut) const
{
  arma::mat bT(Ut.n_cols, Ut.n_rows, arma::fill::zeros);
  for (unsigned int i = 0; i < Ut.n_cols; i++) {
    bT.row(i) = Ut.col(i).t() * BaseController<ModelTemplate>::R_mat;
  }
  return bT;
}

template<class ModelTemplate>
arma::vec SimpleController<ModelTemplate>::get_z(const std::vector<arma::mat> & Plist, const std::vector<arma::mat> & rlist) const
{
  arma::vec z = -Plist[0] * rlist[0];
  return z;
}

}

#endif
