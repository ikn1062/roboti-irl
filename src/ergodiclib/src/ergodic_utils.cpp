#include <ergodiclib/ergodic_utils.hpp>

namespace ergodiclib
{
std::vector<std::vector<int>> create_K_series(int K, int n_dim)
{
  std::vector<int> input_k;
  for (int k = 0; k < K + 1; k++) {
    input_k.push_back(k);
  }
  std::vector<int> permutation(n_dim);
  return create_K_helper(input_k, permutation, n_dim, 0);
}

std::vector<std::vector<int>> create_K_helper(
  std::vector<int> K_num, std::vector<int> permutation,
  int n_dim, int idx)
{
  std::vector<std::vector<int>> res;
  std::vector<std::vector<int>> k_series;
  for (unsigned int i = 0; i < K_num.size(); i++) {
    permutation[idx] = K_num[i];
    if (idx == n_dim - 1) {
      res.push_back(permutation);
    } else {
      k_series = create_K_helper(K_num, permutation, n_dim, idx + 1);
      res.reserve(res.size() + std::distance(k_series.begin(), k_series.end()));
      res.insert(res.end(), k_series.begin(), k_series.end());
    }
  }
  return res;
}

double integralTrapz(std::vector<double> y_trajec, double dx)
{
  int len_trajec = y_trajec.size() - 1;
  double sum = (y_trajec[0] + y_trajec[len_trajec]) / 2.0;
  for (int i = 1; i < len_trajec; i++) {
    sum += y_trajec[i];
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
  return sqrt(sum);
}
}
