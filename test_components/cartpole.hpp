#ifndef CARTPOLE_INCLUDE_GUARD_HPP
#define CARTPOLE_INCLUDE_GUARD_HPP

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "model.hpp"
#include "num_utils.hpp"

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif


#define UNUSED(x) (void)(x)

/// @brief 
class CartPole : public Model
{
    public:
        CartPole() : 
        A_mat(4, 4, arma::fill::zeros),
        B_mat(4, 1, arma::fill::zeros),
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

        CartPole(arma::vec x0_in, arma::vec u0_in, double dt_in, double t0_in, double tf_in, double cart_mass, double pole_mass, double pole_len) : 
        A_mat(4, 4, arma::fill::zeros),
        B_mat(4, 1, arma::fill::zeros),
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

        CartPole(double cart_mass, double pole_mass, double pole_len) : 
        A_mat(4, 4, arma::fill::zeros),
        B_mat(4, 1, arma::fill::zeros),
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

        virtual std::pair<arma::mat, arma::mat> createTrajectory()
        {
            arma::mat x_traj(x0.n_elem, n_iter, arma::fill::zeros);
            arma::mat u_traj(u0.n_elem, n_iter, arma::fill::zeros);

            x_traj.col(0) = x0;
            u_traj.col(0) = u0; 

            arma::mat x_new;
            for (int i = 1; i < n_iter; i++) {
                x_new = integrate(x_traj.col(i-1), u0, dt);
                x_new(2) = ergodiclib::normalizeAngle(x_new(2));
                x_traj.col(i) = x_new;
                u_traj.col(i) = u0;
            }

            std::pair<arma::mat, arma::mat> pair_trajec = {x_traj, u_traj};
            return pair_trajec;
        };

        virtual arma::mat createTrajectory(const arma::vec& x0_input, const arma::mat& ut_input) 
        {
            arma::mat x_traj(x0.n_elem, n_iter, arma::fill::zeros);
            x_traj.col(0) = x0_input;

            arma::mat x_new;
            for (int i = 1; i < n_iter; i++) {
                x_new = integrate(x_traj.col(i-1), ut_input.col(i-1), dt);
                x_new(2) = ergodiclib::normalizeAngle(x_new(2));
                x_traj.col(i) = x_new;
            }

            return x_traj;
        };

    private:
        virtual arma::mat calculateA(const arma::vec& xt, const arma::vec& ut)
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

            A_mat = A;
            return A;
        };

        virtual arma::mat calculateB(const arma::vec& xt, const arma::vec& ut) 
        {
            UNUSED(ut);
            arma::mat B(4, 1, arma::fill::zeros);
            double t = xt(2);

            double cost = cos(t);
            double cos2t = pow(cost, 2); 

            B(1, 0) = 1.0 / (M + m * (1 - cos2t));
            B(3, 0) = cost / (l * (M + m * (1 - cos2t)));

            B_mat = B;
            return B;
        };

        virtual arma::vec dynamics(const arma::vec& x_vec, const arma::vec& u_vec)
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

        arma::mat A_mat;
        arma::mat B_mat;
        arma::vec x0;
        arma::vec u0;
        double dt;
        double t0;
        double tf;
        double M;
        double m;
        double g;
        double l;
        double n_iter;
};


static_assert(ModelConcept<CartPole>);

#endif