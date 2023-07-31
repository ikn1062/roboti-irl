#ifndef NUM_UTIL_INCLUDE_GUARD_HPP
#define NUM_UTIL_INCLUDE_GUARD_HPP
/// \file
/// \brief Contains number utility functions for library

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <ergodiclib/model.hpp>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

namespace ergodiclib
{
/// \brief Double form of PI
constexpr double PI = 3.14159265358979323846;

/// \brief Finds the integral of a trajectory using the trapezoidal rule, using constant dx
/// \param y_trajec Trajectory to perform integration over
/// \param dx difference in x for each step of the trajectory
/// \return Value of integral
double integralTrapz(const arma::vec & y_trajec, const double & dx);

/// \brief Normalizes angle between -pi and pi
/// \param rad Input angle
/// \return Normalized Angle
inline double normalizeAngle(const double & rad)
{
  double new_rad;
  new_rad = fmod(rad, 2 * PI);
  new_rad = fmod(new_rad + 2 * PI, 2 * PI);
  if (new_rad > PI) {
    new_rad -= 2 * PI;
  }
  return new_rad;
}

/// \brief approximately compare two floating-point numbers using
///        an absolute comparison
/// \param d1 - a number to compare
/// \param d2 - a second number to compare
/// \param epsilon - absolute threshold required for equality
/// \return true if abs(d1 - d2) < epsilon
/// NOTE: implement this in the header file
/// constexpr means that the function can be computed at compile time
/// if given a compile-time constant as input
inline constexpr bool almost_equal(const double d1, const double d2, double epsilon = 1.0e-12)
{
  return d1 == d2 || (abs(d1 - d2) <= epsilon);
}

/// \brief convert degrees to radians
/// \param deg - angle in degrees
/// \returns radians
inline constexpr double deg2rad(double deg)
{
  return deg * (PI / 180.0);
}

/// \brief convert radians to degrees
/// \param rad - angle in radians
/// \returns the angle in degrees
inline constexpr double rad2deg(double rad)
{
  return rad * (180.0 / PI);
}

/// \brief Gets the L2 norm of a given vector
/// \tparam T Type of vector
/// \param v Vector of integers
/// \return L2 Norm Value
template<typename T>
double l2_norm(const std::vector<T> & v)
{
  double sum = 0.0;
  for (unsigned int i = 0; i < v.size(); i++) {
    sum += pow(v[i], 2);
  }
  return sum;
}

/// \brief Creates an intial trajectory of model with x0 and u0
/// \param model Model Class used to calculate dynamics
/// \param dt Change in time
/// \return State Position and Control Trajectories over time horizon
template<class ModelTemplate>
std::pair<arma::mat, arma::mat> createTrajectory(ModelTemplate model, const double & dt)
{
  arma::mat x_traj(model.x0.n_elem, model.n_iter, arma::fill::zeros);
  arma::mat u_traj(model.u0.n_elem, model.n_iter, arma::fill::zeros);

  x_traj.col(0) = model.x0;
  u_traj.col(0) = model.u0;

  arma::mat x_new;
  for (int i = 1; i < model.n_iter; i++) {
    x_new = integrate(model, x_traj.col(i - 1), model.u0, dt);
    x_new = model.resolveState(x_new);
    //x_new(2) = ergodiclib::normalizeAngle(x_new(2));
    x_traj.col(i) = x_new;
    u_traj.col(i) = model.u0;
  }

  std::pair<arma::mat, arma::mat> pair_trajec = {x_traj, u_traj};
  return pair_trajec;
}

/// \brief Creates a trajectory given initial position vector x0 and control over time horizon
/// \param model Model Class used to calculate dynamics
/// \param x0_input Position vector of Cartpole Model at t=0
/// \param ut_mat Control Matrix over time horizon
/// \param dt Change in time
/// \return State Position Trajectory over time horizon
template<class ModelTemplate>
arma::mat createTrajectory(
  ModelTemplate model, const arma::vec & x0_input,
  const arma::mat & ut_mat, const double & dt)
{
  const double num_iter = ut_mat.n_cols;
  arma::mat x_traj(model.x0.n_elem, num_iter, arma::fill::zeros);
  x_traj.col(0) = x0_input;

  arma::mat x_new;
  for (int i = 1; i < num_iter; i++) {
    x_new = integrate(model, x_traj.col(i - 1), ut_mat.col(i - 1), dt);
    x_new = model.resolveState(x_new);
    //x_new(2) = ergodiclib::normalizeAngle(x_new(2));
    x_traj.col(i) = x_new;
  }

  return x_traj;
}

/// \brief Creates a trajectory given initial position vector x0 and control over time horizon
/// \param model Model Class used to calculate dynamics
/// \param x0_input Position vector of Cartpole Model at t=0
/// \param ut_mat Control Matrix over time horizon
/// \param num_iter Number of time steps
/// \param dt Change in time
/// \return State Position Trajectory over time horizon
template<class ModelTemplate>
arma::mat createTrajectory(
  ModelTemplate model, const arma::vec & x0_input, const arma::mat & ut_mat,
  const unsigned int & num_iter, const double & dt)
{
  arma::mat x_traj(model.x0.n_elem, num_iter, arma::fill::zeros);
  x_traj.col(0) = x0_input;

  arma::mat x_new;
  for (unsigned int i = 1; i < num_iter; i++) {
    x_new = integrate(model, x_traj.col(i - 1), ut_mat.col(i - 1), dt);
    x_new = model.resolveState(x_new);
    // x_new(2) = ergodiclib::normalizeAngle(x_new(2));
    x_traj.col(i) = x_new;
  }

  return x_traj;
}

/// \brief Integrates the state vector by one time step (dt) using rk4
/// \param model Model Class used to calculate dynamics
/// \param x_vec State Vector state at a given time
/// \param u_vec Control Vector state at a given time
/// \param dt Change in time
/// \return New state vector after one time step
template<class ModelTemplate>
arma::vec integrate(
  ModelTemplate model, arma::vec x_vec, const arma::vec & u_vec,
  const double & dt)
{
  arma::vec k1 = model.dynamics(x_vec, u_vec);
  arma::vec k2 = model.dynamics(x_vec + 0.5 * dt * k1, u_vec);
  arma::vec k3 = model.dynamics(x_vec + 0.5 * dt * k2, u_vec);
  arma::vec k4 = model.dynamics(x_vec + dt * k3, u_vec);

  arma::vec k_sum = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  arma::vec res = x_vec + k_sum;

  return res;
}

}

#endif
