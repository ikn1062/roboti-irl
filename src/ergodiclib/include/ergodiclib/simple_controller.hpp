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
    template<class ModelTemplate>
    class SimpleController :  public BaseController<ModelTemplate>
    {
        ilqrController()
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

        /// \brief Calculates absolute value of descent direction
        /// \param zeta_pair zeta and vega matrix for controller
        /// \param at aT Matrix
        /// \param bt bT Matrix
        /// @return Descent direction as an double value
        double calculateDJ(
            std::pair<arma::mat, arma::mat> const & zeta_pair, const arma::mat & at,
            const arma::mat & bt);

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
    double ilqrController<ModelTemplate>::calculateDJ(
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
    double DJ_integral = integralTrapz(DJ, dt);
    return DJ_integral;
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