#include <ergodiclib/ergodic_measure.hpp>

#ifndef THREADS
#define THREADS 0;
#else 
#include <thread>
#endif

namespace ergodiclib
{
ErgodicMeasure::ErgodicMeasure(
  std::vector<arma::mat> demonstrations,
  std::vector<int> demo_posneg, std::vector<double> demo_weights,
  double dt_demo, std::vector<std::pair<double, double>> dimensionLengths, int nDim, int K)
: D_mat(demonstrations),
  Basis(new fourierBasis(dimensionLengths, nDim, K)),
  E_vec(demo_posneg),
  dt(dt_demo),
  n_dim(demonstrations[0].n_rows),
  m_demo(demonstrations.size()),
  weight_vec(demo_weights),
  K_series(Basis->get_K_series())
{
  sizeK = K_series.size();
  if (demonstrations.size() != E_vec.size() && demonstrations.size() != weight_vec.size()) {
    throw std::invalid_argument("Length of Demonstration unequal to length of demonstration weights");
  }
  if (!almost_equal(std::reduce(weight_vec.begin(), weight_vec.end()), 1.0)) {
    throw std::invalid_argument("Sum of Weight Vector should be equal to 1.0");
  }
  PhiK_vec.resize(sizeK);
  lambdaK_vec.resize(sizeK);
}

arma::vec const & ErgodicMeasure::get_PhiK() const
{
  return PhiK_vec;
}

arma::vec const & ErgodicMeasure::get_LambdaK() const
{
  return lambdaK_vec;
}

void ErgodicMeasure::calcErgodic()
{
#if !THREADS
  calculatePhik();
  calculateLambdaK();
#else 
  std::thread phikThread(&ErgodicMeasure::calculatePhik, this);
  std::thread lambdaThread(&ErgodicMeasure::calculateLambdaK, this);
  if (phikThread.joinable()) phikThread.join();
  if (lambdaThread.joinable()) lambdaThread.join();
#endif
}

void ErgodicMeasure::calculatePhik()
{
  double PhiK_val = 0.0;

  for (unsigned int i = 0; i < sizeK; i++) {
    PhiK_val = 0.0;
    for (int j = 0; j < m_demo; j++) {
      PhiK_val += E_vec[j] * weight_vec[j] * calculateCk(D_mat[j], i);
    }
    PhiK_vec(i) = PhiK_val;
  }

  return;
}

double ErgodicMeasure::calculateCk(
  const arma::mat & x_trajectory, const int k_idx)
{
  int x_len = x_trajectory.n_cols;
  double trajec_time = x_len * dt;
  arma::vec Fk_vec(x_len, arma::fill::zeros);

  for (int i = 0; i < x_len; i++) {
    Fk_vec(i) = Basis->calculateFk(x_trajectory.col(i), k_idx);
  }
  double Ck = (1 / trajec_time) * integralTrapz(Fk_vec, dt);
  return Ck;
}

void ErgodicMeasure::calculateLambdaK()
{
  double lambda_k;
  double s = (n_dim + 1) / 2.0;
  for (unsigned int i = 0; i < sizeK; i++) {
    lambda_k = 1 / pow(1 + l2_norm(K_series[i]), s);
    lambdaK_vec(i) = lambda_k;
  }

  return;
}

arma::rowvec ErgodicMeasure::calculateFourierDFk(const arma::colvec & xTrajectory, const int Kidx) const
{
  return Basis->calculateDFk(xTrajectory, Kidx);
}

}
