#include <ergodiclib/controller.hpp>

ilqrController::ilqrController(Model model_in, arma::vec x0_in, arma::mat Q, arma::mat R, arma::mat P, arma::mat r, double dt_in, double t0_in, double tf_in, double a, double b, double e) :
model(model_in),
x0(x0_in),
Q_mat(Q),
R_mat(R),
P_mat(P),
r_mat(r),
dt(dt_in),
t0(t0_in),
tf(tf_in),
alpha(a),
beta(b),
eps(e)
{
    num_iter = (int) ((tf - t0) / dt);
}

void ilqrController::ILQR()
{
    std::pair<arma::mat, arma::mat> trajectory, descentDirection;
    arma::mat X, U, X_new, U_new, zeta, vega;
    double J, J_new, gamma;

    trajectory = model.createTrajectory();
    X = trajectory.first;
    U = trajectory.second;
    J = objectiveJ(X, U, P_mat);
    
    int n = 0;
    int i = 0;
    gamma = beta;
    while (abs(J) > eps) {
        descentDirection = calculateZeta(X, U);
        zeta = descentDirection.first;
        vega = descentDirection.second;

        n = 0;
        J_new = J;
        while (J_new > J + alpha * gamma * trajectoryJ(X, U)) {
            U_new = U + gamma * vega;
            X_new = model.createTrajectory(x0, U_new);

            J_new = objectiveJ(X_new, U_new, P_mat);

            n += 1;
            gamma = pow(beta, n);

            X = X_new;
            U = U_new;
        }

        trajectory = {X, U};
    }

    std::string x_file = "trajectory_" + i;
    std::string u_file = "control_" + i;
    X.save(x_file, arma::csv_ascii);
    U.save(u_file, arma::csv_ascii);
    i += 1;
}

double ilqrController::objectiveJ(arma::mat Xt, arma::mat Ut, arma::mat P1)
{
    int X_cols = Xt.n_cols-1;
    arma::vec x_tf = Xt.col(X_cols);
    arma::mat finalcost_mat = x_tf.t() * P1 * x_tf; 
    double final_cost = finalcost_mat(0, 0);

    double trajectory_cost = trajectoryJ(Xt, Ut);

    return final_cost + trajectory_cost;
}

double ilqrController::trajectoryJ(arma::mat Xt, arma::mat Ut)
{
    arma::vec trajecJ(Xt.n_cols, arma::fill::zeros);

    arma::vec Xt_i, Ut_i;
    arma::mat cost;
    for (unsigned int i = 0; i < Xt.n_cols; i++) {
        Xt_i = Xt.col(i);
        Ut_i = Ut.col(i);
        cost = Xt_i.t() * Q_mat * Xt_i + Ut_i.t() * R_mat * Ut_i;
        trajecJ(i) = cost(0, 0);
    }

    double trajec_cost = ergodiclib::integralTrapz(trajecJ, dt);
    return trajec_cost;
}

std::pair<arma::mat, arma::mat> ilqrController::calculateZeta(arma::mat Xt, arma::mat Ut)
{
    arma::mat aT = calculate_aT(Xt);
    arma::mat bT = calculate_bT(Ut);
    
    std::pair<std::vector<arma::mat>, std::vector<arma::mat>> listPr = calculatePr(Xt, Ut, aT, bT); 
    std::vector<arma::mat> Plist = listPr.first;
    std::vector<arma::mat> rlist = listPr.second;

    arma::mat zeta(Xt.n_rows, Xt.n_cols, arma::fill::zeros);
    arma::mat vega(Ut.n_rows, Ut.n_cols, arma::fill::zeros);
    
    arma::mat A = model.getA(Xt.col(0), Ut.col(0));
    arma::mat B = model.getB(Xt.col(0), Ut.col(0));   
    arma::vec z = - Plist[0] * rlist[0];
    arma::vec v = - R_mat.i() * B.t() * Plist[0] * z - R_mat.i() * B * rlist[0] - R_mat.t() * bT.col(0);
    zeta.col(0) = z;
    vega.col(0) = v;

    arma::vec zdot;
    for (unsigned int i = 1; i < Xt.n_cols; i++) {
        A = model.getA(Xt.col(i), Ut.col(i));
        B = model.getB(Xt.col(i), Ut.col(i));

        zdot = A * z + B * v;
        z = z + dt * zdot;
        v = - R_mat.i() * B.t() * Plist[i] * z - R_mat.i() * B * rlist[i] - R_mat.t() * bT.col(i); 

        zeta.col(i) = z;
        vega.col(i) = v; 
    } 

    std::pair<arma::mat, arma::mat> descDir = {zeta, vega};
    return descDir;
}

std::pair<std::vector<arma::mat>, std::vector<arma::mat>> ilqrController::calculatePr(arma::mat Xt, arma::mat Ut, arma::mat aT, arma::mat bT)
{
    std::vector<arma::mat> Plist, rlist;
    arma::mat P = P_mat;
    arma::mat r = r_mat;
    Plist.push_back(P);
    rlist.push_back(r);

    arma::mat A, B, Pdot, rdot;
    for (unsigned int i = 1; i < Xt.n_cols; i++) {
        A = model.getA(Xt.col(i), Ut.col(i));
        B = model.getB(Xt.col(i), Ut.col(i));

        Pdot = - P * A - A.t() * P + P * B * R_mat.i() * B.t() * P - Q_mat;
        rdot = - (A - B * R_mat.i() * B.t() * P).t() * r - aT.t() + P * B * R_mat.i() * bT.t();

        P = P + dt * Pdot;
        r = r + dt * rdot;

        Plist.push_back(P);
        rlist.push_back(r);
    }
    P_mat = P;

    std::pair<std::vector<arma::mat>, std::vector<arma::mat>> list_pair = {Plist, rlist}; 
    return list_pair;
}

arma::mat ilqrController::calculate_aT(arma::mat Xt)
{
    arma::mat aT(Xt.n_cols, Xt.n_rows, arma::fill::zeros);
    for (unsigned int i = 0; i < Xt.n_cols; i++) {
        aT.row(i) = Xt.col(i).t() * Q_mat;
    }

    return aT;
}

arma::mat ilqrController::calculate_bT(arma::mat Ut)
{
    arma::mat bT(Ut.n_cols, Ut.n_rows, arma::fill::zeros);
    for (unsigned int i = 0; i < Ut.n_cols; i++) {
        bT.row(i) = Ut.col(i).t() * R_mat;
    }

    return bT;
}

