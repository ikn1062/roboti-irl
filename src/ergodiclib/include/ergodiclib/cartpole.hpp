#ifndef CARTPOLE_INCLUDE_GUARD_HPP
#define CARTPOLE_INCLUDE_GUARD_HPP

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

/*
CAN I OPTIMIZE THIS WITH CONST EXPR?
*/

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
        dt(0.1),
        t0(0.0),
        tf(15.0),
        M(10.0),
        m(5.0),
        g(9.81),
        l(2.0)
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
                x_new = integrate(x_traj.col(i-1), u0);
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
                x_new = integrate(x_traj.col(i-1), ut_input.col(i-1));
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
            double d2x_dt = (2 * l * m * dt * sin(t)) / (M + m * (1 - cos2t));
            
            double d2t_t_a = (-f * sint + g * (m + M) * cost + l * m * pow(dt, 2) * (sin2t - cos2t)) / (l * (M + m * (1 - cos2t)));
            double d2t_t_b = (2 * m * sint * cost * (f*cost + (m+M)*g*sint - l*m*pow(dt,2)*sint*cost)) / pow(l * (M + m * (1 - cos2t)), 2);
            double d2t_t = d2t_t_a + d2t_t_b;
            double d2t_dt = (2 * m * dt * sint * cost) / (M + m * (1 - cos2t)); 

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


#endif