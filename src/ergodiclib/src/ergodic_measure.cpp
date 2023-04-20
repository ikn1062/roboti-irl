#include <ergodiclib/ergodic_measure.hpp>

namespace ergodiclib
{
ErgodicMeasure::ErgodicMeasure(
  std::vector<std::vector<std::vector<double>>> demonstrations,
  std::vector<int> demo_weights, int K_coeff,
  std::vector<std::pair<double, double>> L_dim, double dt_demo)
: D_mat(demonstrations),
  E_vec(demo_weights),
  K(K_coeff),
  L(L_dim),
  dt(dt_demo),
  n_dim(demonstrations[0][0].size()),
  m_demo(demonstrations.size())
{
  if (demonstrations.size() != E_vec.size()) {
    throw std::invalid_argument("Length of Demonstration unequal to length of demonstration weights");
  }
  for (int i = 0; i < m_demo; i++) {
    weight_vec.push_back(1 / m_demo);
  }
  K_series = create_K_series(K_coeff, n_dim);

  PhiK_vec.resize(K_series.size());
  hK_vec.resize(K_series.size());
  lambdaK_vec.resize(K_series.size());
}

std::vector<double> ErgodicMeasure::get_PhiK()
{
  return PhiK_vec;
}

std::vector<double> ErgodicMeasure::get_LambdaK()
{
  return lambdaK_vec;
}

std::vector<double> ErgodicMeasure::get_hK()
{
  return hK_vec;
}

void ErgodicMeasure::calcErgodic()
{
  PhiK_vec = calculatePhik();
  lambdaK_vec = calculateLambdaK();
}

std::vector<double> ErgodicMeasure::calculatePhik()
{
  double PhiK_val = 0;

  for (unsigned int i = 0; i < K_series.size(); i++) {
    PhiK_val = 0;
    for (int j = 0; j < m_demo; j++) {
      PhiK_val += E_vec[j] * weight_vec[j] * calculateCk(D_mat[j], K_series[i], i);
    }
    PhiK_vec[i] = PhiK_val;
  }
  return PhiK_vec;
}

double ErgodicMeasure::calculateCk(
  const std::vector<std::vector<double>> & x_trajectory,
  const std::vector<int> & K_vec, int k_idx)
{
  int x_len = x_trajectory.size();
  double trajec_time = x_len * dt;
  std::vector<double> Fk_vec(x_len);

  for (int i = 0; i < x_len; i++) {
    Fk_vec[i] = calculateFk(x_trajectory[i], K_vec, k_idx);
  }
  double Ck = (1 / trajec_time) * integralTrapz(Fk_vec, dt);
  return Ck;
}

double ErgodicMeasure::calculateFk(
  const std::vector<double> & x_i_trajectory,
  const std::vector<int> & K_vec, int k_idx)
{
  double hk = calculateHk(K_vec, k_idx);
  double fourier_basis = 1.0;
  double upper, lower;
  for (unsigned int i = 0; i < x_i_trajectory.size(); i++) {
    upper = K_vec[i] * PI * x_i_trajectory[i];
    lower = L[i].first - L[i].second;
    fourier_basis *= cos(upper / lower);
  }
  double Fk = (1 / hk) * fourier_basis;
  return Fk;
}

double ErgodicMeasure::calculateHk(const std::vector<int> & K_vec, int k_idx)
{
  double l0, l1, ki;

  double hk = 1.0;
  for (int i = 0; i < n_dim; i++) {
    l0 = L[i].first;
    l1 = L[i].second;
    if (K_vec[i] == 0) {
      hk *= (l1 - l0);
      continue;
    }
    ki = (K_vec[i] * PI) / l1;
    hk *= (2 * ki * (l1 - l0) - sin(2 * ki * l0) + sin(2 * ki * l1)) / (4 * ki);
  }
  hk = sqrt(hk);
  hK_vec[k_idx] = hk;
  return hk;
}

std::vector<double> ErgodicMeasure::calculateLambdaK()
{
  double lambda_k;
  double s = (n_dim + 1) / 2.0;
  for (unsigned int i = 0; i < K_series.size(); i++) {
    // FIX UTILITY to not return sqrt
    lambda_k = 1 / pow(1 + pow(l2_norm(K_series[i]), 2), s);
    lambdaK_vec[i] = lambda_k;
  }
  return lambdaK_vec;
}
}
