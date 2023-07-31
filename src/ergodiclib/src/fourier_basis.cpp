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
fourierBasis::fourierBasis(std::vector<std::pair<double, double>> dimensionLengths, int nDim, int K)
: _nDim(nDim),
  _lengthDims(dimensionLengths)
{
  size_t ksize = std::pow(K+1, nDim);
  _kSeries.resize(ksize);
  create_K_series(K);
  _hkVec.resize(ksize);
  calculateHk();
}

std::vector<std::vector<int>> const & fourierBasis::get_K_series() const
{
  return _kSeries;
}

double fourierBasis::calculateFk(const arma::vec & xTrajectory, const int Kidx) const
{
  double hk = _hkVec[Kidx];
  double fourier_basis = 1.0;
  double upper, lower;

  for (unsigned int i = 0; i < xTrajectory.n_elem; i++) {
    upper = _kSeries[Kidx][i] * PI * xTrajectory(i);
    lower = _lengthDims[i].first - _lengthDims[i].second;
    fourier_basis *= std::cos(upper / lower);
  }

  double Fk = (1 / hk) * fourier_basis;
  return Fk;
}

arma::rowvec fourierBasis::calculateDFk(
  const arma::colvec & xTrajectory, const int Kidx) const
{
  arma::rowvec dfk(_nDim, arma::fill::ones);

  arma::rowvec normState(_nDim, arma::fill::zeros);
  arma::rowvec kDerivative(_nDim, arma::fill::zeros);

  double hk = _hkVec[Kidx];
  double dkj = 1;

  // Uses a variation of the Product Except Self algorithm
  for (unsigned int i = 0; i < _nDim; i++) {
    normState(i) = (_kSeries[Kidx][i] * PI) / (_lengthDims[i].first - _lengthDims[i].second);
    kDerivative(i) = std::cos(normState(i) * xTrajectory(i));
  }

  for (unsigned int i = 1; i < _nDim; i++){
    dfk(i) = dfk(i-1) * kDerivative[i-1];
  }

  for (int i = _nDim - 2; i >= 0; i--) {
    dkj *= kDerivative(i+1);
    dfk(i) *= dkj;
  }

  for (unsigned int i = 0; i < _nDim; i++) {
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
  std::vector<double> normState(_nDim, 0);
  arma::vec integral;
  double hk;
  double dx = 0.0001;
  unsigned int integral_iter;
  
  for (unsigned int i = 0; i < _kSeries.size(); i++) {
    K_vec = _kSeries[i];
    hk = 1.0;

    for (unsigned int j = 0; j < _nDim; j++) {
      normState[j] = (K_vec[j] * PI) / (_lengthDims[j].second - _lengthDims[j].first);
    }

#if !THREADS
    double coski, hktemp;
    for (unsigned int j = 0; j < _nDim; j++) {
      integral_iter = (unsigned int) ((_lengthDims[j].second - _lengthDims[j].first) / dx);
      integral = arma::vec(integral_iter, arma::fill::zeros);
      for (unsigned int k = 0; k < integral_iter; k++) {
        coski = std::cos(normState[j] * (k * dx));
        integral(k) = coski * coski;
      }
      hktemp = integralTrapz(integral, dx);
      hk *= hktemp;
    }
#else
    std::vector<std::thread> threadVec(_nDim);

    for (unsigned int k = 0; k < _nDim; k++) {
      integral_iter = (unsigned int) ((_lengthDims[k].second - _lengthDims[k].first) / dx);
      //threadVec.push_back(std::thread(integralHk, integral_iter, normState[k], dx, std::ref(hk)));
      threadVec.push_back(std::thread(integralHk, integral_iter, normState[k], dx, std::ref(hk)));
    }

    for (std::thread & t : threadVec) {
      if (t.joinable()) t.join();
    }
#endif
    hk = sqrt(hk);
    _hkVec[i] = hk;
  }

  return;
}


// /// \brief Reccursive helper for fourier series coefficients
// /// \return List of Fourier Series Coefficients
static void createKhelper(std::vector<int>& inputK, std::vector<int>& Kvec, std::vector<std::vector<int>>& Kseries, const unsigned int& n, unsigned int& idx)
{
  if (Kvec.size() == n) {
    Kseries[idx] = Kvec;
    idx++;
    return;
  }
  for (auto & k : inputK) {
    Kvec.push_back(k);
    createKhelper(inputK, Kvec, Kseries, n, idx);
    Kvec.pop_back();
  }
  return;
}


void fourierBasis::create_K_series(const int K)
{
  std::vector<int> inputK, Kvec;
  unsigned int idx = 0;
  for (int k = 0; k < K + 1; k++) {
    inputK.push_back(k);
  }
  createKhelper(inputK, Kvec, _kSeries, _nDim, idx);
  return;
}
}
