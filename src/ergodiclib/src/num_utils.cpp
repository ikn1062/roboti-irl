#include <ergodiclib/num_utils.hpp>

namespace ergodiclib
{
double integralTrapz(const arma::vec& y_trajec, double dx)
{
  int len_trajec = y_trajec.n_elem - 1;
  double sum = (y_trajec(0) + y_trajec(len_trajec)) / 2.0;
  for (int i = 1; i < len_trajec; i++) {
    sum += y_trajec(i);
  }
  sum *= dx;
  return sum;
}

double l2_norm(const std::vector<int> & v)
{
  double sum = 0.0;
  for (unsigned int i = 0; i < v.size(); i++) {
    sum += pow(v[i], 2);
  }
  return sqrt(sum);
}

double l2_norm(const std::vector<double> & v)
{
  double sum = 0.0;
  for (unsigned int i = 0; i < v.size(); i++) {
    sum += pow(v[i], 2);
  }
  return sum;
}
}
