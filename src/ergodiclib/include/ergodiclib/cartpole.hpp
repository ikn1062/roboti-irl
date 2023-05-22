#ifndef CARTPOLE_INCLUDE_GUARD_HPP
#define CARTPOLE_INCLUDE_GUARD_HPP
/// \file
/// \brief Definition for Cartpole Model

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <ergodiclib/model.hpp>
#include <ergodiclib/num_utils.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif


#define UNUSED(x) (void)(x)

namespace ergodiclib
{
/// \brief CartPole Dynamics model class 
class CartPole
{
    public:
        /// \brief Position vector of Cartpole Model at t=0
        arma::vec x0;

        /// \brief Control vector of Cartpole Model at t=0
        arma::vec u0;

        /// \brief Time difference for model dynamics (< 0.05)
        double dt;

        /// \brief Initial Time Step
        double t0;

        /// \brief Final Time Step
        double tf;

        /// \brief Basic Class Constructor for the CartPole Model
        CartPole() : 
        x0({0.0, 0.0, ergodiclib::PI, 0.0}),
        u0({0.0}),
        dt(0.0005),
        t0(0.0),
        tf(10.0),
        M(10.0),
        m(5.0),
        g(9.81),
        l(2.0)
        {
            n_iter = (int) ((tf - t0)/ dt);
        };

        /// \brief Class Constructor for the CartPole Model
        /// \param x0_in Position vector of Cartpole Model at t=0 
        /// \param u0_in Control vector of Cartpole Model at t=0
        /// \param dt_in Time difference for model dynamics (< 0.05)
        /// \param t0_in Initial Time Step
        /// \param tf_in Final Time Step
        /// \param cart_mass Mass of Cart 
        /// \param pole_mass Mass of pole
        /// \param pole_len Length of pole
        CartPole(arma::vec x0_in, arma::vec u0_in, double dt_in, double t0_in, double tf_in, double cart_mass, double pole_mass, double pole_len) : 
        x0(x0_in),
        u0(u0_in),
        dt(dt_in),
        t0(t0_in),
        tf(tf_in),
        M(cart_mass),
        m(pole_mass),
        g(9.81),
        l(pole_len)
        {
            n_iter = (int) ((tf - t0)/ dt);
        };

        /// \brief Class Constructor for the CartPole Model
        /// \param cart_mass Mass of Cart 
        /// \param pole_mass Mass of pole
        /// \param pole_len Length of pole
        CartPole(double cart_mass, double pole_mass, double pole_len) : 
        x0({0.0, 0.0, ergodiclib::PI, 0.0}),
        u0({0.0}),
        dt(0.0005),
        t0(0.0),
        tf(10.0),
        M(cart_mass),
        m(pole_mass),
        g(9.81),
        l(pole_len)
        {
            n_iter = (int) ((tf - t0)/ dt);
        };
        
        /// \brief Get A (Dfx) Matrix from Model initialized at (xt, ut)
        /// \param xt Position Vector state at time t
        /// \param ut Control Vector state at time t
        /// \return A (Dfx) Matrix 
        arma::mat getA(const arma::vec& xt, const arma::vec& ut) const
        {
            return calculateA(xt, ut);
        };

        /// \brief Get B (Dfu) Matrix from Model initialized at (xt, ut)
        /// \param xt Position Vector state at time t
        /// \param ut Control Vector state at time t
        /// \return B (Dfu) Matrix 
        arma::mat getB(const arma::vec& xt, const arma::vec& ut) const
        {
            return calculateB(xt, ut);
        };

        /// \brief Creates an intial trajectory of model with x0 and u0
        /// \return State Position and Control Trajectories over time horizon
        virtual std::pair<arma::mat, arma::mat> createTrajectory() const
        {
            arma::mat x_traj(x0.n_elem, n_iter, arma::fill::zeros);
            arma::mat u_traj(u0.n_elem, n_iter, arma::fill::zeros);

            x_traj.col(0) = x0;
            u_traj.col(0) = u0; 

            arma::mat x_new;
            for (int i = 1; i < n_iter; i++) {
                x_new = integrate(x_traj.col(i-1), u0);
                x_new(2) = ergodiclib::normalizeAngle(x_new(2));
                x_traj.col(i) = x_new;
                u_traj.col(i) = u0;
            }

            std::pair<arma::mat, arma::mat> pair_trajec = {x_traj, u_traj};
            return pair_trajec;
        };

        /// \brief Creates a trajectory given initial position vector x0 and control over time horizon
        /// \param x0_input Position vector of Cartpole Model at t=0
        /// \param ut_mat Control Matrix over time horizon
        /// \return State Position Trajectory over time horizon
        virtual arma::mat createTrajectory(const arma::vec& x0_input, const arma::mat& ut_mat) const
        {
            arma::mat x_traj(x0.n_elem, n_iter, arma::fill::zeros);
            x_traj.col(0) = x0_input;

            arma::mat x_new;
            for (int i = 1; i < n_iter; i++) {
                x_new = integrate(x_traj.col(i-1), ut_mat.col(i-1));
                x_new(2) = ergodiclib::normalizeAngle(x_new(2));
                x_traj.col(i) = x_new;
            }

            return x_traj;
        };

    private:
        /// \brief Mass of cart 
        double M;

        /// \brief Mass of pole
        double m;
        double g;
        double l;
        double n_iter;

        /// \brief Calculates A (Dfx) Matrix from Model initialized at (xt, ut)
        /// \param xt Position Vector state at time t
        /// \param ut Control Vector state at time t
        /// \return A (Dfx) Matrix 
        virtual arma::mat calculateA(const arma::vec& xt, const arma::vec& ut) const
        {
            arma::mat A(4, 4, arma::fill::zeros);
            double t = xt(2);
            double dt = xt(3);
            double f = ut(0);

            double sint = sin(t);
            double cost = cos(t);
            double sin2t = pow(sint, 2);
            double cos2t = pow(cost, 2);

            double d2x_t_a = (m * g * (cos2t - sin2t) - l * m * pow(dt, 2) * cost) / (M + m * (1 - cos2t));
            double d2x_t_b = (2 * m * sint * cost * (f + g*m*sint*cost - l*m*pow(dt, 2)*sint)) / pow((M + m * (1 - cos2t)), 2);
            double d2x_t = d2x_t_a - d2x_t_b;
            double d2x_dt = (-2 * l * m * dt * sin(t)) / (M + m * (1 - cos2t)); // adding a minus sign
            
            double d2t_t_a = (-f * sint + g * (m + M) * cost + l * m * pow(dt, 2) * (sin2t - cos2t)) / (l * (M + m * (1 - cos2t)));
            double d2t_t_b = (2 * m * sint * cost * (f*cost + (m+M)*g*sint - l*m*pow(dt,2)*sint*cost)) / (l * pow(M + m * (1 - cos2t), 2));
            double d2t_t = d2t_t_a - d2t_t_b;
            double d2t_dt = (-2 * m * dt * sint * cost) / (M + m * (1 - cos2t)); // adding a minus sign

            A(0, 1) = 1.0;
            A(1, 2) = d2x_t;
            A(1, 3) = d2x_dt;
            A(2, 3) = 1.0;
            A(3, 2) = d2t_t;
            A(3, 3) = d2t_dt;

            return A;
        };

        /// \brief Calculates B (Dfu) Matrix from Model initialized at (xt, ut)
        /// \param xt Position Vector state at time t
        /// \param ut Control Vector state at time t
        /// \return B (Dfu) Matrix 
        virtual arma::mat calculateB(const arma::vec& xt, const arma::vec& ut) const
        {
            UNUSED(ut);
            arma::mat B(4, 1, arma::fill::zeros);
            double t = xt(2);

            double cost = cos(t);
            double cos2t = pow(cost, 2); 

            B(1, 0) = 1.0 / (M + m * (1 - cos2t));
            B(3, 0) = cost / (l * (M + m * (1 - cos2t)));

            return B;
        };

        /// \brief Returns difference in state vector
        /// \param x_vec State Vector state at a given time
        /// \param u_vec Control Vector state at a given time
        /// \return Change in state vector over time x_dot
        virtual arma::vec dynamics(const arma::vec& x_vec, const arma::vec& u_vec) const
        {
            //double x = x_vec(0);
            double dx = x_vec(1);
            double t = x_vec(2);
            double dt = x_vec(3);

            double f = u_vec(0);

            double sint = sin(t);
            double cost = cos(t);
            double cos2t = pow(cost, 2);

            arma::vec xdot(4, 1, arma::fill::zeros);
            xdot(0) = dx;
            xdot(1) = (-m*l*sint*pow(dt,2) + f + m*g*cost*sint) / (M + m * (1 - cos2t));
            xdot(2) = dt;
            xdot(3) = (-m*l*cost*sint*pow(dt,2) + f*cost + (M+m)*g*sint) / (l * (M + m * (1 - cos2t)));

            return xdot;
        }

        /// \brief Integrates the state vector by one time step (dt) using rk4
        /// \param x_vec State Vector state at a given time
        /// \param u_vec Control Vector state at a given time
        /// \return New state vector after one time step
        arma::vec integrate(arma::vec x_vec, const arma::vec& u_vec) const
        {
            arma::vec k1 = dynamics(x_vec, u_vec);
            arma::vec k2 = dynamics(x_vec + 0.5 * dt * k1, u_vec);
            arma::vec k3 = dynamics(x_vec + 0.5 * dt * k2, u_vec);
            arma::vec k4 = dynamics(x_vec + dt * k3, u_vec);

            arma::vec k_sum = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            arma::vec res = x_vec + k_sum;

            return res;
        }
};

static_assert(ModelConcept<CartPole>);

}


#endif