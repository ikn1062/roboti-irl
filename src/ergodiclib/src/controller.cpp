#include <ergodiclib/controller.hpp>
#include <ergodiclib/ergodic_measure.hpp>

namespace ergodiclib
{
   arma::mat iLQRController::calc_b(const arma::mat& u_mat, const arma::mat& R_mat) 
   {
      // Change to entire trajectory?
      arma::mat b_mat(u_mat.n_cols, u_mat.n_rows, arma::fill::zeros);
      b_mat = u_mat.t() * R_mat;
      return b_mat;
   }

   arma::mat iLQRController::calc_a(const arma::mat& x_mat, const std::vector<std::vector<int> > K_series, const double q)
   {
      // CALL ERGODICMEASURE OBJECT HERE - should be initialized
      arma::mat a_mat(x_mat.n_rows, x_mat.n_cols, arma::fill::zeros);
      arma::mat ak_mat(1, x_mat.n_cols, arma::fill::zeros); // should be 1 for time in row dim

      std::vector<double> lambda = ErgodicMeasure::get_LambdaK();
      std::vector<double> phi = ErgodicMeasure::get_PhiK();
      std::vector<std::vector<double> > fix_ck_arma;

      double t_final = 0.0;
      for (int t = 0; t < x_mat.n_rows; t++) {
         for (int i = 0; i < K_series.size(); i++) {
            double ck = ErgodicMeasure::calculateCk(fix_ck_arma, K_series, i);
            double dfk = ErgodicMeasure::calculateDFk;

            ak_mat = lambda[i] * (2 * (ck - phi[i]) * ((1/t_final) * dfk));

            a_mat.row(t) = ak_mat;
            ak_mat.fill(0.0);
         }
      }

      a_mat = q * a_mat;
      return a_mat;
   }

   arma::mat iLQRController::integrate(const arma::mat& x_mat, const arma::mat& u_mat)
   {
      double dt;
      arma::mat k1 = dynamics(x_mat, u_mat);
      arma::mat k2 = dynamics(x_mat + 0.5 * dt * k1, u_mat);
      arma::mat k3 = dynamics(x_mat + 0.5 * dt * k2, u_mat);
      arma::mat k4 = dynamics(x_mat + dt * k3, u_mat);
      return x_mat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
   }

   arma::mat iLQRController::dynamics(const arma::mat& x_mat, const arma::mat& u_mat) 
   {
      arma::mat A, B;
      return A * x_mat + B * u_mat;
   }

   void iLQRController::calculatePr(const arma::mat& at, const arma::mat& bt)
   {
      int timespan;
      double dt;
      arma::mat A, B, Q, R;
      std::vector<arma::mat> listP, listr;
      arma::mat Rinv = R.i();

      arma::mat P, r, a, b, Pdot, rdot;
      
      for (int i = 1; i < timespan; i++) {
         // A and B here have to be solved from the model
         P = listP[i-1];
         r = listr[i-1];

         a = at.row(i).t();
         b = bt.row(i).t();

         Pdot = P * B * Rinv * B.t() * P - Q - P * A - A.t() * P;
         // - np.transpose(A - B @ Rinv @ np.transpose(B) @ P) @ r - a + P @ B @ Rinv @ b
         arma::mat rdot_int = A - B * Rinv * B.t() * P;
         rdot = - 1.0 * rdot_int.t() * r - a + P * B * Rinv * b;

         listP[i] = dt * Pdot + P;
         listr[i] = dt * rdot + r;
      }

   }

   std::pair<arma::mat, arma::mat> iLQRController::descentDirection(arma::mat xt, arma::mat ut, std::vector<arma::mat> listP, std::vector<arma::mat> listr, arma::mat bt)
   {
      int timespan;
      double dt;

      arma::mat A, B, R;
      arma::mat zeta(xt.n_rows, xt.n_cols, arma::fill::zeros);
      arma::mat vega(ut.n_rows, ut.n_cols, arma::fill::zeros);
      arma::mat Rinv = R.i();

      arma::mat P, r, b;
      arma::mat zdot;
      for (int i = 1; i < timespan; i++) {
         P = listP[i];
         r = listr[i];
         b = bt.row(i);

         vega.row(i) = -1.0 * Rinv * B.t() * P * zeta.row(i-1) - Rinv * B.t() * r - Rinv * b;
         zdot = A * zeta.row(i-1) + B * vega.row(i);
         zeta.row(i) = zeta.row(i-1) + zdot * dt;
      }

      std::pair<arma::mat, arma::mat> zeta_pair = {zeta, vega};
      return zeta_pair;
   }

   double iLQRController::DJ(std::pair<arma::mat, arma::mat> zeta_pair, const arma::mat& at, const arma::mat& bt) 
   {
      int timespan;
      arma::mat J(timespan, 1, arma::fill::zeros);
      arma::mat zeta = zeta_pair.first;
      arma::mat vega = zeta_pair.second;

      arma::mat a_T, b_T;

      for (int i = 0; i < timespan; i++) {
         a_T = at.row(i).t();
         b_T = bt.row(i).t();

         J.row(i) = a_T * zeta.row(i) + b_T * vega.row(i);
      }

      // integrate and return J
      return 0.0;
   }

   arma::mat iLQRController::make_trajec(arma::mat x0, arma::mat ut)
   {
      int timespan;
      arma::mat x_traj(timespan, x0.n_elem, arma::fill::zeros);
      x_traj.row(0) = x0;
      
      arma::mat x_new;
      for (int i = 1; i < timespan; i++) {
         x_new = integrate(x_traj.row(i-1), ut.row(i-1));
         x_traj.row(i) = x_new;
      }
      
      return x_traj;
   }

}