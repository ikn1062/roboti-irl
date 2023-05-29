#include <ergodiclib/ergodic_measure.hpp>

namespace ergodiclib
{
ErgodicMeasure::ErgodicMeasure(
  std::vector<arma::mat> demonstrations,
  std::vector<int> demo_weights,
  double dt_demo, fourierBasis & basis)
: D_mat(demonstrations),
  Basis(basis),
  E_vec(demo_weights),
  dt(dt_demo),
  n_dim(demonstrations[0].n_rows),
  m_demo(demonstrations.size())
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

arma::vec ErgodicMeasure::get_PhiK() const
{
  return PhiK_vec;
}

arma::vec ErgodicMeasure::get_LambdaK() const
{
  return lambdaK_vec;
}

void ErgodicMeasure::calcErgodic()
{
  PhiK_vec = calculatePhik();
  lambdaK_vec = calculateLambdaK();
}

arma::vec ErgodicMeasure::calculatePhik()
{
  double PhiK_val = 0.0;

  for (unsigned int i = 0; i < K_series.size(); i++) {
    PhiK_val = 0.0;
    for (int j = 0; j < m_demo; j++) {
      PhiK_val += E_vec[j] * weight_vec[j] * calculateCk(D_mat[j], K_series[i], i);
    }
    PhiK_vec(i) = PhiK_val;
  }
  return PhiK_vec;
}

double ErgodicMeasure::calculateCk(
  const arma::mat & x_trajectory,
  const std::vector<int> & K_vec, int k_idx)
{
  int x_len = x_trajectory.n_cols;
  double trajec_time = x_len * dt;
  arma::vec Fk_vec(x_len, arma::fill::zeros);

  for (int i = 0; i < x_len; i++) {
    Fk_vec(i) = Basis.calculateFk(x_trajectory.col(i), K_vec, k_idx);
  }
  double Ck = (1 / trajec_time) * integralTrapz(Fk_vec, dt);
  return Ck;
}

arma::vec ErgodicMeasure::calculateLambdaK()
{
  double lambda_k;
  double s = (n_dim + 1) / 2.0;
  for (unsigned int i = 0; i < K_series.size(); i++) {
    lambda_k = 1 / pow(1 + l2_norm(K_series[i]), s);
    lambdaK_vec(i) = lambda_k;
  }
  return lambdaK_vec;
}


}
