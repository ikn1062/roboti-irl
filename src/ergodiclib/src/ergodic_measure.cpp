#include <ergodiclib/ergodic_measure.hpp>

namespace ergodiclib
{
ErgodicMeasure::ErgodicMeasure(
  std::vector<std::vector<std::vector<double>>> demonstrations,
  std::vector<int> demo_weights, int K_coeff,
  std::vector<std::pair<double, double>> L_dim, double dt_demo, fourierBasis basis)
: D_mat(demonstrations),
  E_vec(demo_weights),
  dt(dt_demo),
  n_dim(demonstrations[0][0].size()),
  m_demo(demonstrations.size()),
  Basis(basis)
{
  if (demonstrations.size() != E_vec.size()) {
    throw std::invalid_argument("Length of Demonstration unequal to length of demonstration weights");
  }
  for (int i = 0; i < m_demo; i++) {
    weight_vec.push_back(1 / m_demo);
  }
  K_series = basis.get_K_series();

  PhiK_vec.resize(K_series.size());
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
    Fk_vec[i] = Basis.calculateFk(x_trajectory[i], K_vec, k_idx);
  }
  double Ck = (1 / trajec_time) * integralTrapz(Fk_vec, dt);
  return Ck;
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
