#include <ergodiclib/fourier_basis.hpp>

#ifndef THREADS
#define THREADS 0;
#else 
#include <thread>
#include <mutex>
#include <chrono>
std::mutex lockHk;
#endif

namespace ergodiclib
{
fourierBasis::fourierBasis(std::vector<std::pair<double, double>> L_dim, int num_dim, int K)
: n_dim(num_dim),
  L(L_dim)
{
  K_series = create_K_series(K);
  hK_vec.resize(K_series.size());
  calculateHk();
}


std::vector<double> fourierBasis::get_hK() const
{
  return hK_vec;
}

std::vector<std::vector<int>> fourierBasis::get_K_series() const
{
  return K_series;
}

double fourierBasis::calculateFk(
  const arma::vec & xTrajectory,
  const std::vector<int> & Kvec, const int Kidx) const
{
  double hk = hK_vec[Kidx];
  double fourier_basis = 1.0;
  double upper, lower;

  for (unsigned int i = 0; i < xTrajectory.n_elem; i++) {
    upper = Kvec[i] * PI * xTrajectory(i);
    lower = L[i].first - L[i].second;
    fourier_basis *= std::cos(upper / lower);
  }

  double Fk = (1 / hk) * fourier_basis;
  return Fk;
}

arma::rowvec fourierBasis::calculateDFk(
  const arma::colvec & xTrajectory,
  const std::vector<int> & Kvec, const int Kidx) const
{
  arma::rowvec dfk(n_dim, arma::fill::ones);

  arma::rowvec normState(n_dim, arma::fill::zeros);
  arma::rowvec kDerivative(n_dim, arma::fill::zeros);

  double hk = hK_vec[Kidx];
  double dkj = 1;

  // Uses a variation of the Product Except Self algorithm
  for (unsigned int i = 0; i < n_dim; i++) {
    normState(i) = (Kvec[i] * PI) / (L[i].first - L[i].second);
    kDerivative(i) = std::cos(normState(i) * xTrajectory(i));
  }

  for (unsigned int i = 1; i < n_dim; i++){
    dfk(i) = dfk(i-1) * kDerivative[i-1];
  }
  for (int i = n_dim - 2; i >= 0; i--) {
    dkj *= kDerivative(i+1);
    dfk(i) *= dkj;
  }

  for (unsigned int i = 0; i < n_dim; i++) {
    dfk(i) *= (1 / hk) * (-1.0 * normState(i)) * std::sin(normState(i) * xTrajectory(i));
  }

  return dfk;
}

#if THREADS
// static void integralHk(const unsigned int integral_iter, const double ki, const double dx, double& hk)
void integralHk(const unsigned int integral_iter, const double ki, const double dx, double& hk)
{
  arma::vec integralVector(integral_iter, arma::fill::zeros);
  double hkdim, coski;

  for (unsigned int i = 0; i < integral_iter; i++) {
    coski = std::cos(ki * (i * dx));
    integralVector(i) = coski * coski;
  }

  hkdim = integralTrapz(integralVector, dx);

  lockHk.lock();
  hk *= hkdim;
  lockHk.unlock();

  return;
}
#endif

void fourierBasis::calculateHk()
{
  std::vector<int> K_vec;
  std::vector<double> normState(n_dim, 0);
  arma::vec integral;
  double hk;
  double dx = 0.0001;
  unsigned int integral_iter;

  for (unsigned int i = 0; i < K_series.size(); i++) {
    K_vec = K_series[i];
    hk = 1.0;

    for (unsigned int j = 0; j < n_dim; j++) {
      normState[j] = (K_vec[j] * PI) / (L[j].second - L[j].first);
    }

#if !THREADS
    double coski, hktemp;
    for (unsigned int j = 0; j < n_dim; j++) {
      integral_iter = (unsigned int) ((L[j].second - L[j].first) / dx);
      integral = arma::vec(integral_iter, arma::fill::zeros);
      for (unsigned int k = 0; k < integral_iter; k++) {
        coski = std::cos(normState[j] * (k * dx));
        integral(k) = coski * coski;
      }
      hktemp = integralTrapz(integral, dx);
      hk *= hktemp;
    }
#else
    std::vector<std::thread> threadVec(n_dim);

    for (unsigned int k = 0; k < n_dim; k++) {
      integral_iter = (unsigned int) ((L[k].second - L[k].first) / dx);
      //threadVec.push_back(std::thread(integralHk, integral_iter, normState[k], dx, std::ref(hk)));
      threadVec.push_back(std::thread(integralHk, integral_iter, normState[k], dx, std::ref(hk)));
    }

    for (std::thread & t : threadVec) {
      if (t.joinable()) t.join();
    }
#endif
    hk = sqrt(hk);
    hK_vec[i] = hk;
  }

  return;
}


// /// \brief Reccursive helper for fourier series coefficients
// /// \return List of Fourier Series Coefficients
static void createKhelper(std::vector<int>& inputK, std::vector<int>& Kvec, std::vector<std::vector<int>>& Kseries)
{
  if (Kvec.size() == inputK.size()) {
    Kseries.push_back(Kvec);
    return;
  }
  for (auto & k : inputK) {
    Kvec.push_back(k);
    createKhelper(inputK, Kvec, Kseries);
    Kvec.pop_back();
  }
  return;
}


std::vector<std::vector<int>> fourierBasis::create_K_series(const int K)
{
  std::vector<std::vector<int>> Kseries;
  std::vector<int> inputK, Kvec;
  for (int k = 0; k < K + 1; k++) {
    inputK.push_back(k);
  }
  createKhelper(inputK, Kvec, Kseries);
  return Kseries;
}
}
