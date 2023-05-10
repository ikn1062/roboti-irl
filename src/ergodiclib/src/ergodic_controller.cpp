#include <ergodiclib/ergodic_controller.hpp>

namespace ergodiclib
{
   // ergController::ergController(ErgodicMeasure ergodicMes, fourierBasis basis, Model model_agent, double q_val, arma::mat R_mat, arma::mat Q_mat, double t0_val, double tf_val, double dt_val, double eps_val, double beta_val) : 
   // ergodicMeasure(ergodicMes), 
   // Basis(basis),
   // model(model_agent),
   // q(q_val),
   // R(R_mat),
   // Q(Q_mat),
   // t0(t0_val),
   // tf(tf_val),
   // dt(dt_val),
   // eps(eps_val),
   // beta(beta_val)
   // {
   //       n_iter = (int) ((tf - t0)/ dt);
   // };

   arma::mat ergController::calc_b(const arma::mat& u_mat) 
   {
      arma::mat b_mat(u_mat.n_cols, u_mat.n_rows, arma::fill::zeros); // Transposed
      for (unsigned int i = 0; i < u_mat.n_cols; i++) {
         b_mat.col(i) = u_mat.col(i).t() * R;
      }
      return b_mat;
   }

   arma::mat ergController::calc_a(const arma::mat& x_mat)
   {
      arma::mat a_mat(x_mat.n_cols, x_mat.n_rows, arma::fill::zeros);
      arma::rowvec ak_mat(x_mat.n_rows, arma::fill::zeros); // should be 1 for time in row dim

      const std::vector<std::vector<int> > K_series = Basis.get_K_series();
      arma::mat lambda = ergodicMeasure.get_LambdaK();
      arma::mat phi = ergodicMeasure.get_PhiK();
      
      arma::vec dfk; 
      for (unsigned int i = 0; i < K_series.size(); i++) {
         double ck = ergodicMeasure.calculateCk(x_mat, K_series[i], i);

         for (unsigned int t = 0; t < x_mat.n_cols; t++) {
            dfk = Basis.calculateDFk(x_mat.col(t), K_series[i]);
            ak_mat = lambda[i] * (2 * (ck - phi[i]) * ((1/tf) * dfk));
            a_mat.row(t) = ak_mat;
         }
      }

      a_mat = q * a_mat;
      return a_mat;
   }

   std::pair<std::vector<arma::mat>, std::vector<arma::mat>> ergController::calculatePr(arma::mat xt, arma::mat ut, const arma::mat& at_mat, const arma::mat& bt_mat)
   {
      arma::mat A = model.getA(xt, ut);
      arma::mat B = model.getB(xt, ut);

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

   std::pair<arma::mat, arma::mat> ergController::descentDirection(arma::mat xt, arma::mat ut, std::vector<arma::mat> listP, std::vector<arma::mat> listr, arma::mat bt)
   {
      arma::mat A = model.getA(xt, ut);
      arma::mat B = model.getB(xt, ut);

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

   double ergController::DJ(std::pair<arma::mat, arma::mat> zeta_pair, const arma::mat& at, const arma::mat& bt) 
   {
      arma::vec J(n_iter, 1, arma::fill::zeros);
      arma::mat zeta = zeta_pair.first;
      arma::mat vega = zeta_pair.second;

      arma::mat a_T, b_T;

      for (int i = 0; i < n_iter; i++) {
         a_T = at.row(i).t();
         b_T = bt.row(i).t();

         J.row(i) = a_T * zeta.row(i) + b_T * vega.row(i);
      }

      // integrate and return J
      double J_integral = integralTrapz(J, dt);
      return J_integral;
   }

   int ergController::gradient_descent(arma::vec x0) 
   {
      std::pair<arma::mat, arma::mat> xtut = model.createTrajectory();
      arma::mat xt = xtut.first;
      arma::mat ut = xtut.second; 
      model.setx0(x0);
      
      double dj = 2e31;

      arma::mat at, bt;
      std::pair<arma::mat, arma::mat> zeta_pair;
      std::pair<std::vector<arma::mat>, std::vector<arma::mat>> listPr;
      while (abs(dj) > eps) {
         at = calc_a(xt);
         bt = calc_b(ut);

         listPr = calculatePr(xt, ut, at, bt);
         
         zeta_pair = descentDirection(xt, ut, listPr.first, listPr.second, bt);

         dj = DJ(zeta_pair, at, bt);

         arma::mat vega = zeta_pair.second;
         ut = ut + beta * vega;
         xt = model.createTrajectory(x0, ut);
      }

      return 1;
   }
}