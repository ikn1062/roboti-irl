#include <ergodiclib/fourier_basis.hpp>


/// \brief Reccursive helper for fourier series coefficients
/// \param K_num Size of series coefficients
/// \param permutation Current Permutation in sequence
/// \param n_dim Size of dimension for demonstrations
/// \param idx Current idx of sequence in permutation
/// \return List of Fourier Series Coefficients
static std::vector<std::vector<int>> create_K_helper(
  const std::vector<int> K_num, std::vector<int> permutation,
  const int n_dim, int idx)
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


namespace ergodiclib
{
fourierBasis::fourierBasis(std::vector<std::pair<double, double>> L_dim, int num_dim, int K)
: n_dim(num_dim),
  L(L_dim)
{
  K_series = create_K_series(K, num_dim);
  hK_vec.resize(K_series.size());
  for (unsigned int i = 0; i < K_series.size(); i++) {
    hK_vec[i] = calculateHk(K_series[i], i);
  }
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
  const arma::vec & x_i_trajectory,
  const std::vector<int> & K_vec, const int & k_idx)
{
  double hk = hK_vec[k_idx];
  double fourier_basis = 1.0;
  double upper, lower;

  for (unsigned int i = 0; i < x_i_trajectory.n_elem; i++) {
    upper = K_vec[i] * PI * x_i_trajectory(i);
    lower = L[i].first - L[i].second;
    fourier_basis *= cos(upper / lower);
  }

  double Fk = (1 / hk) * fourier_basis;
  return Fk;
}

double fourierBasis::calculateHk(const std::vector<int> & K_vec, const int & k_idx)
{
  double l0, l1, ki;

  double hk = 1.0;
  double dx = 0.0001; // 
  unsigned int integral_iter = 0;
  for (unsigned int i = 0; i < n_dim; i++) {
    l0 = L[i].first;
    l1 = L[i].second;
    ki = (K_vec[i] * PI) / (l1 - l0);

    // Calculating from integral:
    // if (K_vec[i] == 0) {
    //   hk *= (l1 - l0);
    //   continue;
    // }
    // hk *= (2 * ki * (l1 - l0) - sin(2 * ki * l0) + sin(2 * ki * l1)) / (4 * ki);

    integral_iter = (unsigned int) ((l1 - l0) / dx);
    arma::vec integral(integral_iter, arma::fill::zeros);
    for (unsigned int i = 0; i < integral_iter; i++) {
      integral(i) = pow(cos(ki * (i * dx)), 2);
    }
    hk *= integralTrapz(integral, dx);
  }

  hk = sqrt(hk);
  return hk;
}

arma::rowvec fourierBasis::calculateDFk(
  const arma::colvec & xt,
  const std::vector<int> & K_vec, const int &k_idx) const
{
  arma::rowvec dfk(xt.n_rows, arma::fill::zeros);
  // std::cout << "hk_values" << std::endl;
  // for (unsigned int i = 0; i < hK_vec.size(); i++) {
  //   std::cout << hK_vec[i] << std::endl;
  // }


  double hk = hK_vec[k_idx];
  double ki, kj, dfk_val;
  for (unsigned int i = 0; i < n_dim; i++) {
    dfk_val = 1.0;
    ki = (K_vec[i] * PI) / (L[i].first - L[i].second);
    dfk_val *= (1 / hk) * (-1.0 * ki) * sin(ki * xt(i));

    for (unsigned int j = 0; j < n_dim; j++) {
      if (i != j) {
        kj = (K_vec[j] * PI) / (L[j].first - L[j].second);
        dfk_val *= cos(kj * xt(j));
      }
    }

    dfk(i) = dfk_val;
  }
  //dfk.print("dfk: ");
  return dfk;
}

std::vector<std::vector<int>> fourierBasis::create_K_series(const int & K, const int & n_dim)
{
  std::vector<int> input_k;
  for (int k = 0; k < K + 1; k++) {
    input_k.push_back(k);
  }
  std::vector<int> permutation(n_dim);
  return create_K_helper(input_k, permutation, n_dim, 0);
}
}
