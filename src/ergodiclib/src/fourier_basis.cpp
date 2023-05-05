
#include <ergodiclib/fourier_basis.hpp>

namespace ergodiclib
{
    fourierBasis::fourierBasis(std::vector<std::pair<double, double>> L_dim, int num_dim, int K) : 
    L(L_dim),
    n_dim(num_dim)
    {
        K_series = create_K_series(K, num_dim);
        hK_vec.resize(K_series.size());
    }


    std::vector<double> fourierBasis::get_hK()
    {
        return hK_vec;
    }

    std::vector<std::vector<int>> fourierBasis::get_K_series()
    {
        return K_series;
    }

    double fourierBasis::calculateFk(
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

    double fourierBasis::calculateHk(const std::vector<int> & K_vec, int k_idx)
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

    arma::mat fourierBasis::calculateDFk(const arma::mat& xi_vec, const std::vector<int>& K_vec)
    {
        arma::mat dfk(xi_vec.n_rows, xi_vec.n_cols, arma::fill::zeros);

        double ki = 0.0;
        for (int i = 0; i < n_dim; i++) {
            ki = (K_vec[i] * PI) / (L[i].first - L[i].second);
            dfk(i) = (1 / hK_vec[i]) * (-1.0 * ki) * cos(ki * xi_vec(i)) * sin(ki * xi_vec(i));
        }

        return dfk;
    }

    std::vector<std::vector<int>> fourierBasis::create_K_series(int K, int n_dim)
    {
        std::vector<int> input_k;
        for (int k = 0; k < K + 1; k++) {
            input_k.push_back(k);
        }
        std::vector<int> permutation(n_dim);
        return create_K_helper(input_k, permutation, n_dim, 0);
    }

    std::vector<std::vector<int>> fourierBasis::create_K_helper(
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

}

