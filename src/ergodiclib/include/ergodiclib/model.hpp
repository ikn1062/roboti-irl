#ifndef MODEL_INCLUDE_GUARD_HPP
#define MODEL_INCLUDE_GUARD_HPP

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include<armadillo>

#define UNUSED(x) (void)(x)

class Model
{
    public:
        arma::mat getA() 
        {return A_mat;};

        arma::mat getB() 
        {return B_mat;};

        void setx0(arma::vec x0_input)
        {x0 = x0_input;};

        virtual std::pair<arma::mat, arma::mat> createTrajectory() 
        {
            return {arma::mat(1, 1, arma::fill::zeros), arma::mat(1, 1, arma::fill::zeros)};
        };

        virtual arma::mat createTrajectory(arma::vec x0_input, arma::mat ut_input) 
        {
            UNUSED(x0_input); 
            UNUSED(ut_input); 
            return arma::mat(1, 1, arma::fill::zeros);
        };

    private:
        arma::vec integrate(const arma::vec& x_vec, const arma::vec& u_vec)
        {
            arma::vec k1 = dynamics(x_vec, u_vec);
            arma::vec k2 = dynamics(x_vec + 0.5 * dt * k1, u_vec);
            arma::vec k3 = dynamics(x_vec + 0.5 * dt * k2, u_vec);
            arma::vec k4 = dynamics(x_vec + dt * k3, u_vec);
            return x_vec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }

        arma::vec dynamics(const arma::vec& x_vec, const arma::vec& u_vec)
        {
            arma::mat A = calculateA(x_vec, u_vec);
            arma::mat B = calculateB(x_vec, u_vec); 
            return A * x_vec + B * u_vec;
        }

        virtual arma::mat calculateA(arma::vec xt, arma::vec ut)
        {
            UNUSED(xt); 
            UNUSED(ut); 
            return arma::mat(1, 1, arma::fill::zeros);
        };

        virtual arma::mat calculateB(arma::vec xt, arma::vec ut)
        {
            UNUSED(xt); 
            UNUSED(ut); 
            return arma::mat(1, 1, arma::fill::zeros);
        };

        arma::mat A_mat;
        arma::mat B_mat;
        arma::vec x0;
        arma::mat u0;
        double dt;
        double t0;
        double tf;

        friend class CartPole;
};


#endif