#include <ergodiclib/controller.hpp>

namespace ergodiclib
{
   iLQRController::iLQRController(ErgodicMeasure ergodicMes, fourierBasis basis, double q_val, arma::mat R_mat, arma::mat Q_mat, double t0_val, double tf_val, double dt_val):
   ergodicMeasure(ergodicMes),
   Basis(basis),
   q(q_val),
   R(R_mat),
   Q(Q_mat),
   t0(t0),
   tf(tf),
   dt(dt)
   {
      n_iter = (int) ((tf - t0)/ dt);
   }

   arma::mat iLQRController::calc_b(const arma::mat& u_mat) 
   {
      arma::mat b_mat(u_mat.n_cols, u_mat.n_rows, arma::fill::zeros); // Transposed
      for (int i = 0; i < u_mat.n_cols; i++) {
         b_mat.col(i) = u_mat.col(i).t() * R;
      }
      return b_mat;
   }

   arma::mat iLQRController::calc_a(const arma::mat& x_mat)
   {
      arma::mat a_mat(x_mat.n_cols, x_mat.n_rows, arma::fill::zeros);
      arma::rowvec ak_mat(x_mat.n_rows, arma::fill::zeros); // should be 1 for time in row dim

      const std::vector<std::vector<int> > K_series = Basis.get_K_series();
      arma::mat lambda = ergodicMeasure.get_LambdaK();
      arma::mat phi = ergodicMeasure.get_PhiK();
      
      arma::vec dfk; 
      for (int i = 0; i < K_series.size(); i++) {
         double ck = ergodicMeasure.calculateCk(x_mat, K_series[i], i);

         for (int t = 0; t < x_mat.n_cols; t++) {
            dfk = Basis.calculateDFk(x_mat.col(t), K_series[i]);
            ak_mat = lambda[i] * (2 * (ck - phi[i]) * ((1/tf) * dfk));
            a_mat.row(t) = ak_mat;
         }
      }

      a_mat = q * a_mat;
      return a_mat;
   }

   arma::vec iLQRController::integrate(const arma::vec& x_mat, const arma::vec& u_mat)
   {
      arma::vec k1 = dynamics(x_mat, u_mat);
      arma::vec k2 = dynamics(x_mat + 0.5 * dt * k1, u_mat);
      arma::vec k3 = dynamics(x_mat + 0.5 * dt * k2, u_mat);
      arma::vec k4 = dynamics(x_mat + dt * k3, u_mat);
      return x_mat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
   }

   arma::vec iLQRController::dynamics(const arma::vec& x_mat, const arma::vec& u_mat) 
   {
      arma::mat A, B;
      return A * x_mat + B * u_mat;
   }

   std::pair<std::vector<arma::mat>, std::vector<arma::mat>> iLQRController::calculatePr(const arma::mat& at_mat, const arma::mat& bt_mat)
   {
      arma::mat A, B;
      std::vector<arma::mat> listP, listr;
      arma::mat Rinv = R.i();

      arma::mat P, r, at, bt, Pdot, rdot;
      
      for (int i = 1; i < n_iter; i++) {
         // A and B here have to be solved from the model
         P = listP[i-1];
         r = listr[i-1];

         at = at_mat.row(i).t();
         bt = bt_mat.row(i).t();

         Pdot = P * B * Rinv * B.t() * P - Q - P * A - A.t() * P;

         arma::mat rdot_int = A - B * Rinv * B.t() * P;
         rdot = - 1.0 * rdot_int.t() * r - at + P * B * Rinv * bt;

         listP[i] = dt * Pdot + P;
         listr[i] = dt * rdot + r;
      }

      std::pair<std::vector<arma::mat>, std::vector<arma::mat>> list_pair = {listP, listr};
      return list_pair; 
   }

   std::pair<arma::mat, arma::mat> iLQRController::descentDirection(arma::mat xt, arma::mat ut, std::vector<arma::mat> listP, std::vector<arma::mat> listr, arma::mat bt)
   {
      arma::mat A, B;
      arma::mat zeta(xt.n_rows, xt.n_cols, arma::fill::zeros);
      arma::mat vega(ut.n_rows, ut.n_cols, arma::fill::zeros);
      arma::mat Rinv = R.i();

      arma::mat P, r, b;
      arma::vec zdot;
      for (int i = 1; i < n_iter; i++) {
         P = listP[i];
         r = listr[i];
         b = bt.row(i);

         vega.col(i) = -1.0 * Rinv * B.t() * P * zeta.col(i-1) - Rinv * B.t() * r - Rinv * b;
         zdot = A * zeta.col(i-1) + B * vega.col(i);
         zeta.col(i) = zeta.col(i-1) + zdot * dt;
      }

      std::pair<arma::mat, arma::mat> zeta_pair = {zeta, vega};
      return zeta_pair;
   }

   double iLQRController::DJ(std::pair<arma::mat, arma::mat> zeta_pair, const arma::mat& at, const arma::mat& bt) 
   {
      int timespan;
      arma::vec J(timespan, 1, arma::fill::zeros);
      arma::mat zeta = zeta_pair.first;
      arma::mat vega = zeta_pair.second;

      arma::mat a_T, b_T;

      for (int i = 0; i < timespan; i++) {
         a_T = at.row(i).t();
         b_T = bt.row(i).t();

         J.row(i) = a_T * zeta.row(i) + b_T * vega.row(i);
      }

      // integrate and return J
      double J_integral = integralTrapz(J, dt);
      return J_integral;
   }

   arma::mat iLQRController::make_trajec(arma::vec x0, arma::mat ut)
   {
      arma::mat x_traj(x0.n_elem, n_iter, arma::fill::zeros);
      x_traj.col(0) = x0;
      
      arma::mat x_new;
      for (int i = 1; i < n_iter; i++) {
         x_new = integrate(x_traj.col(i-1), ut.col(i-1));
         x_traj.col(i) = x_new;
      }
      
      return x_traj;
   }

}