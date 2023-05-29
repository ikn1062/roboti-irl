#ifndef NUM_UTIL_INCLUDE_GUARD_HPP
#define NUM_UTIL_INCLUDE_GUARD_HPP
/// \file
/// \brief Contains number utility functions for library

#include <iosfwd>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

namespace ergodiclib
{
constexpr double PI = 3.14159265358979323846;

/// \brief Finds the integral of a trajectory using the trapezoidal rule, using constant dx
/// \param y_trajec Trajectory to perform integration over
/// \param dx difference in x for each step of the trajectory
/// \return Value of integral
double integralTrapz(const arma::vec & y_trajec, const double & dx);

/// \brief Normalizes angle between -pi and pi
/// \param rad Input angle
/// \return Normalized Angle
double normalizeAngle(const double & rad);

/// \brief approximately compare two floating-point numbers using
///        an absolute comparison
/// \param d1 - a number to compare
/// \param d2 - a second number to compare
/// \param epsilon - absolute threshold required for equality
/// \return true if abs(d1 - d2) < epsilon
/// NOTE: implement this in the header file
/// constexpr means that the function can be computed at compile time
/// if given a compile-time constant as input
constexpr bool almost_equal(double d1, double d2, double epsilon = 1.0e-12)
{
  return d1 == d2 || (abs(d1 - d2) <= epsilon);
}

/// \brief convert degrees to radians
/// \param deg - angle in degrees
/// \returns radians
constexpr double deg2rad(double deg)
{
  return deg * (PI / 180.0);
}

/// \brief convert radians to degrees
/// \param rad - angle in radians
/// \returns the angle in degrees
constexpr double rad2deg(double rad)
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

}

#endif
