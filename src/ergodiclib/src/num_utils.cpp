#include <ergodiclib/num_utils.hpp>

namespace ergodiclib
{
double integralTrapz(const arma::vec & y_trajec, const double & dx)
{
  int len_trajec = y_trajec.n_elem - 1;
  double sum = (y_trajec(0) + y_trajec(len_trajec)) / 2.0;
  for (int i = 1; i < len_trajec; i++) {
    sum += y_trajec(i);
  }
  sum *= dx;
  return sum;
}

double normalizeAngle(const double & rad)
{
  double new_rad;
  new_rad = fmod(rad, 2 * PI);
  new_rad = fmod(new_rad + 2 * PI, 2 * PI);
  if (new_rad > PI) {
    new_rad -= 2 * PI;
  }
  return new_rad;
}

}
