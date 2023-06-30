/// \file
/// \brief Cartpole Controller
///
/// PARAMETERS:
///     rate (float): Rate of Main Loop
///     control_type (string) : Name of Controller Type ('file' or 'MPC')
///     control_file (string) : File path to controls given 'file' option
/// PUBLISHES:
///     /cartpole/timestep (std_msgs::msg::UInt64): Current Timestep of Simulation
///     /cartpole/cmd (std_msgs::msg::Float64): Command Publisher for cartpole sim
/// SUBSCRIBES:
///     /cartpole/joint_state (sensor_msgs::msg::JointState): Receives joint states of cart and pole
/// SERVERS:
///     /cartpole/file_control_trigger (std_srvs::srv::Trigger) : Runs the file controller for 1 loop
///     /cartpole/mpc_trigger (std_srvs::srv::Trigger) : Turns the MPC controller on


#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <ctime>
#include <thread>

#include <armadillo>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/u_int64.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_srvs/srv/trigger.hpp"

#include <ergodiclib/file_utils.hpp>
#include <ergodiclib/ergodic_measure.hpp>
#include <ergodiclib/simple_controller.hpp>
#include <ergodiclib/cartpole.hpp>

using namespace std::chrono;

class CartpoleControl : public rclcpp::Node
{
public:
  CartpoleControl()
  : Node("cartpole_ctrl")
  {
    // CARTPOLE SIMULATION CONSTRUCTOR
    // Decalre Variables
    declare_parameter("rate", 100.0);
    declare_parameter("control_type", "file");
    declare_parameter("control_file", "control_out_1.csv");

    // Get Variables
    rate_ = get_parameter("rate").as_double();
    auto duration_ = std::chrono::duration<double>(1.0 / rate_);
    auto mpc_duration_ = std::chrono::duration<double>(1.0 / 10);

    control_type = get_parameter("control_type").as_string();
    control_file = get_parameter("control_file").as_string();

    // Publishers and Subscribers
    timestep_pub_ = create_publisher<std_msgs::msg::UInt64>("cartpole/timestep", 50);
    command_pub_ = create_publisher<std_msgs::msg::Float64>("/cartpole/cmd", 10);
    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/cartpole/joint_state",
      10,
      std::bind(&CartpoleControl::jointstate_cb_, this, std::placeholders::_1));

    file_cntrl_trigger_ = create_service<std_srvs::srv::Trigger>(
      "~/file_control_trigger", std::bind(
        &CartpoleControl::fileController, this, std::placeholders::_1, std::placeholders::_2));
    MPC_trigger_ =
      create_service<std_srvs::srv::Trigger>(
      "~/mpc_trigger",
      std::bind(&CartpoleControl::MPCTrigger, this, std::placeholders::_1, std::placeholders::_2));

    // Initial Variables
    timestep_.data = 0;
    timestep_pub_->publish(timestep_);

    x0 = arma::vec({0.0, 0.0, ergodiclib::PI, 0.0});
    u0 = arma::vec({0.0});
    Q = arma::mat(4, 4, arma::fill::eye);
    R = arma::mat(1, 1, arma::fill::eye);
    P = arma::mat(4, 4, arma::fill::eye);
    r = arma::mat(4, 1, arma::fill::zeros);

    mpc_trigger = false;
    Q(0, 0) = 0.0;
    Q(1, 1) = 0.0;
    Q(2, 2) = 25.0;
    Q(3, 3) = 1.0;

    R(0, 0) = 0.01;

    P(0, 0) = 0.0001;
    P(1, 1) = 0.0001;
    P(2, 2) = 100;
    P(3, 3) = 2;

    cartpole = ergodiclib::CartPole(x0, u0, dt, t0, tf, 10.0, 5.0, 2.0);
    controller = ergodiclib::SimpleController(cartpole, Q, R, P, r, 7500, alpha, beta, eps);

    // Create main callback
    timer_ = create_wall_timer(duration_, std::bind(&CartpoleControl::cartpole_main, this));
    mpc_timer_ =
      create_wall_timer(mpc_duration_, std::bind(&CartpoleControl::ModelPredictiveControl, this));
  }

private:
  // Main loop publisher and timer
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr mpc_timer_;
  double rate_;
  int err;

  // Input for controls
  std::string control_type;
  std::string control_file;

  // Controls message
  std_msgs::msg::Float64 force_cmd;

  // MPC Control Variables
  bool mpc_trigger;
  double x_cart, v_cart, x_pole, v_pole;
  arma::vec x0;
  arma::vec u0;
  arma::mat Q;
  arma::mat R;
  arma::mat P;
  arma::mat r;

  double dt = 0.01;
  double t0 = 0.0;
  double tf = 5.0;
  double alpha = 0.40;
  double beta = 0.85;
  double eps = 0.01;
  double M = 10.0;
  double m = 5.0;
  double l = 2.0;
  unsigned int max_iter = 5000;

  ergodiclib::CartPole cartpole;
  ergodiclib::SimpleController<ergodiclib::CartPole> controller;

  std::pair<arma::mat, arma::mat> trajectories;
  arma::mat X;
  arma::mat U;

  // Create publishers and subscribers
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr command_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr file_cntrl_trigger_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr MPC_trigger_;

  // Publishes the current timestep of the simulation
  std_msgs::msg::UInt64 timestep_;
  rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr timestep_pub_;

  void cartpole_main()
  {
    timestep_.data += (1 / rate_) * 1000;       // converts to an int, note that this is in ms
    timestep_pub_->publish(timestep_);
  }

  void jointstate_cb_(const sensor_msgs::msg::JointState & cartpole_js)
  {
    std::string joint_name = cartpole_js.name[0];

    if (joint_name == "slider_to_cart") {
      x_cart = cartpole_js.position[0];
      v_cart = cartpole_js.velocity[0];
    } else {
      x_pole = ergodiclib::normalizeAngle(cartpole_js.position[0]);
      v_pole = cartpole_js.velocity[0];
    }
  }

  void fileController(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    std::vector<float> control_input;
    std::ifstream controls(control_file);

    if (!controls.is_open()) {
      RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Control file could not be open");
      response->success = false;
      response->message = "Control File could not be opened";
      return;
    }

    std::string line;
    float num;
    while (std::getline(controls, line)) {
      std::istringstream iss(line);
      std::string value;
      while (std::getline(iss, value, ',')) {
        num = std::stof(value);
        control_input.push_back(num);
      }
    }

    controls.close();

    std::cout << "control_input size: " << control_input.size() << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < control_input.size(); i++) {
      force_cmd.data = control_input[i];
      command_pub_->publish(force_cmd);
      std::this_thread::sleep_for(std::chrono::microseconds(1150));
      force_cmd.data = 0.0;
      command_pub_->publish(force_cmd);
      std::this_thread::sleep_for(std::chrono::microseconds(3650));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time: " << duration.count() << std::endl;

    response->success = true;
    response->message = "Control Passed";
    return;
  }

  void ModelPredictiveControl()
  {
    while (mpc_trigger) {
      double controls;
      arma::vec curr_pos({x_cart, v_cart, x_pole, v_pole});
      trajectories = controller.ModelPredictiveControl(curr_pos, u0, 10, 100);
      X = trajectories.first;
      U = trajectories.second;

      for (unsigned int i = 0; i < U.n_cols; i++) {
        controls = U(0, 0);
        force_cmd.data = controls;
        command_pub_->publish(force_cmd);
      }
    }
  }

  void MPCTrigger(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (mpc_trigger) {
      mpc_trigger = false;
      response->message = "Control Trigger - ON";
    } else {
      mpc_trigger = true;
      response->message = "Control Trigger - OFF";
    }

    response->success = true;
    return;
  }

};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Creating Cartpole Controller Node. ");

  rclcpp::spin(std::make_shared<CartpoleControl>());

  rclcpp::shutdown();
  return 0;
}
