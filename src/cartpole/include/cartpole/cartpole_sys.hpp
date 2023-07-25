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

  /// \brief Number of iterations
  int n_iter;


  /// \brief Basic Class Constructor for the CartPole Model
  CartPole()
  : x0({0.0, 0.0, ergodiclib::PI, 0.0}),
    u0({0.0}),
    dt(0.0005),
    t0(0.0),
    tf(10.0),
    M(10.0),
    m(5.0),
    g(9.81),
    l(2.0)
  {
    n_iter = (int) ((tf - t0) / dt);
  }

  /// \brief Class Constructor for the CartPole Model
  /// \param x0_in Position vector of Cartpole Model at t=0
  /// \param u0_in Control vector of Cartpole Model at t=0
  /// \param dt_in Time difference for model dynamics (< 0.05)
  /// \param t0_in Initial Time Step
  /// \param tf_in Final Time Step
  /// \param cart_mass Mass of Cart
  /// \param pole_mass Mass of pole
  /// \param pole_len Length of pole
  CartPole(
    arma::vec x0_in, arma::vec u0_in, double dt_in, double t0_in, double tf_in,
    double cart_mass, double pole_mass, double pole_len)
  : x0(x0_in),
    u0(u0_in),
    dt(dt_in),
    t0(t0_in),
    tf(tf_in),
    M(cart_mass),
    m(pole_mass),
    g(9.81),
    l(pole_len)
  {
    n_iter = (int) ((tf - t0) / dt);
  }

  /// \brief Class Constructor for the CartPole Model
  /// \param cart_mass Mass of Cart
  /// \param pole_mass Mass of pole
  /// \param pole_len Length of pole
  CartPole(double cart_mass, double pole_mass, double pole_len)
  : x0({0.0, 0.0, ergodiclib::PI, 0.0}),
    u0({0.0}),
    dt(0.0005),
    t0(0.0),
    tf(10.0),
    M(cart_mass),
    m(pole_mass),
    g(9.81),
    l(pole_len)
  {
    n_iter = (int) ((tf - t0) / dt);
  }

  /// \brief Get A (Dfx) Matrix from Model initialized at (xt, ut)
  /// \param xt Position Vector state at time t
  /// \param ut Control Vector state at time t
  /// \return A (Dfx) Matrix
  arma::mat getA(const arma::vec & xt, const arma::vec & ut) const
  {
    return calculateA(xt, ut);
  }

  /// \brief Get B (Dfu) Matrix from Model initialized at (xt, ut)
  /// \param xt Position Vector state at time t
  /// \param ut Control Vector state at time t
  /// \return B (Dfu) Matrix
  arma::mat getB(const arma::vec & xt, const arma::vec & ut) const
  {
    return calculateB(xt, ut);
  }

  /// \brief Returns difference in state vector
  /// \param x_vec State Vector state at a given time
  /// \param u_vec Control Vector state at a given time
  /// \return Change in state vector over time x_dot
  arma::vec dynamics(const arma::vec & x_vec, const arma::vec & u_vec) const
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
    // xdot(1) = (-m * l * sint * pow(dt, 2) + f + m * g * cost * sint) / (M + m * (1 - cos2t));
    xdot(1) = (f + m*l*(pow(dt,2) * sint - dt * cost)) / (m + M);
    xdot(2) = dt;
    // xdot(3) = (-m * l * cost * sint * pow(dt, 2) + f * cost + (M + m) * g * sint) / (l * (M + m * (1 - cos2t)));
    xdot(3) = (g*sint-((f + l * pow(dt, 2) * sint * m)/(M+m))*cost)/(l*(4/3-(cos2t * m/(m+M))));

    return xdot;
  }

  arma::mat resolveState(const arma::mat & x_vec) const
  {
    arma::mat x_new = x_vec;
    x_new(2) = ergodiclib::normalizeAngle(x_vec(2));
    return x_new;
  }

private:
  /// \brief Mass of cart
  double M;

  /// \brief Mass of pole
  double m;

  /// \brief Gravity
  double g;

  /// \brief Length of pole
  double l;

  /// \brief Calculates A (Dfx) Matrix from Model initialized at (xt, ut)
  /// \param xt Position Vector state at time t
  /// \param ut Control Vector state at time t
  /// \return A (Dfx) Matrix
  arma::mat calculateA(const arma::vec & xt, const arma::vec & ut) const
  {
    arma::mat A(4, 4, arma::fill::zeros);
    double t = xt(2);
    double dt = xt(3);
    double f = ut(0);

    double dt2 = pow(dt, 2);
    double sint = sin(t);
    double cost = cos(t);
    double sin2t = pow(sint, 2);
    double cos2t = pow(cost, 2);

    // double d2x_t_a = (m * g * (cos2t - sin2t) - l * m * pow(dt, 2) * cost) / (M + m * (1 - cos2t));
    // double d2x_t_b =
    //   (2 * m * sint * cost *
    //   (f + g * m * sint * cost - l * m * pow(dt, 2) * sint)) / pow((M + m * (1 - cos2t)), 2);
    // double d2x_t = d2x_t_a - d2x_t_b;
    double d2x_t_a_denom = l*((m*cos2t)/(M + m) - 4/3);
    double d2x_t_a = ((cost*(g*cost + (sin(t)*(l*m*sint*dt2 + f))/(M + m) - (dt2*l*m*cos2t)/(M + m))) - (sin(t)*(g*sin(t) - (cos(t)*(l*m*sin(t)*dt2 + f))/(M + m)))) / d2x_t_a_denom; 
    double d2x_t_b = (2*m*cos2t*sint*(g*sint- (cost*(l*m*sint*dt2 + f))/(M+m))) / (l*(M+m)*pow((m*cos2t)/(M+m) - 4/3, 2));

    double d2x_t = (l * m * (dt2 * cost + d2x_t_a + d2x_t_b)) / (M + m);
    double d2x_dt = -(l * m * 8 * dt * sint) / (3 * m * cos2t - 4 * (m + M));

    // double d2t_t_a =
    //   (-f * sint + g * (m + M) * cost + l * m *
    //   pow(dt, 2) * (sin2t - cos2t)) / (l * (M + m * (1 - cos2t)));
    // double d2t_t_b =
    //   (2 * m * sint * cost *
    //   (f * cost + (m + M) * g * sint - l * m *
    //   pow(dt, 2) * sint * cost)) / (l * pow(M + m * (1 - cos2t), 2));
    // double d2t_t = d2t_t_a - d2t_t_b;
    double d2t_t_a = (3 * (g * cost * (M + m) + f * sint + l * dt2 * m * (sin2t - cos2t)))/(l * (3 * m * cos2t - 4 * (M + m)));
    double d2t_t_b = -((9 * m * sin(2*t) * (g * sin(t) * (M + m) - f * cost - l * dt2 * m * cost * sint))/(l * pow((3 * m * cos2t - 4 * (M + m)), 2)));
    double d2t_t = d2t_t_a - d2t_t_b;
    double d2t_dt = (dt * m * sin(2*t) * 3) / (3 * m * cos2t - 4 * (M + m)); 

    A(0, 1) = 1.0;
    A(1, 2) = d2x_t;
    A(1, 3) = d2x_dt;
    A(2, 3) = 1.0;
    A(3, 2) = d2t_t;
    A(3, 3) = d2t_dt;

    return A;
  }

  /// \brief Calculates B (Dfu) Matrix from Model initialized at (xt, ut)
  /// \param xt Position Vector state at time t
  /// \param ut Control Vector state at time t
  /// \return B (Dfu) Matrix
  arma::mat calculateB(const arma::vec & xt, const arma::vec & ut) const
  {
    UNUSED(ut);
    arma::mat B(4, 1, arma::fill::zeros);
    double t = xt(2);

    double cost = cos(t);
    double cos2t = pow(cost, 2);

    // B(1, 0) = 1.0 / (M + m * (1 - cos2t));
    // B(3, 0) = cost / (l * (M + m * (1 - cos2t)));

    B(1, 0) = - 4.0 / (3 * m * cos2t - 4.0 * (M + m));
    B(3, 0) = (3.0 * cost) / (l * (3 * m * cos2t - 4 * (M + m)));

    return B;
  }

  // /// \brief Integrates the state vector by one time step (dt) using rk4
  // /// \param x_vec State Vector state at a given time
  // /// \param u_vec Control Vector state at a given time
  // /// \return New state vector after one time step
  // arma::vec integrate(arma::vec x_vec, const arma::vec & u_vec) const
  // {
  //   arma::vec k1 = dynamics(x_vec, u_vec);
  //   arma::vec k2 = dynamics(x_vec + 0.5 * dt * k1, u_vec);
  //   arma::vec k3 = dynamics(x_vec + 0.5 * dt * k2, u_vec);
  //   arma::vec k4 = dynamics(x_vec + dt * k3, u_vec);

  //   arma::vec k_sum = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  //   arma::vec res = x_vec + k_sum;

  //   return res;
  // }
};

static_assert(ModelConcept<CartPole>);

}


#endif
