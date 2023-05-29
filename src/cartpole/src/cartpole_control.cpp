/// \file
/// \brief Cartpole Controller File
///
/// PARAMETERS:
///     name (type): desc
/// PUBLISHES:
///     name (type): desc
/// SUBSCRIBES:
///     name (type): desc


#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <sstream>
#include <fstream>

#include <armadillo>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/u_int64.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/twist.hpp"

using namespace std::chrono;

class CartpoleControl : public rclcpp::Node
{
public:
  CartpoleControl()
  : Node("cartpole_ctrl")
  {
    // CARTPOLE SIMULATION CONSTRUCTOR
    // Decalre Variables
    declare_parameter("rate", 10.0);
    declare_parameter("control_type", "MPC");
    declare_parameter("control_file", "");

    // Get Variables
    rate_ = get_parameter("rate").as_double();
    auto duration_ = std::chrono::duration<double>(1.0 / rate_);

    control_type = get_parameter("control_type").as_string();
    control_file = get_parameter("control_file").as_string();

    // Publishers and Subscribers
    timestep_pub_ = create_publisher<std_msgs::msg::UInt64>("cartpole/timestep", 50);
    command_pub_ = create_publisher<std_msgs::msg::Float64>("/cartpole/cmd", 10);
    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>("/cartpole/joint_state", 10, std::bind(&CartpoleControl::jointstate_cb_, this, std::placeholders::_1));

    // Initial Variables
    timestep_.data = 0;
    timestep_pub_->publish(timestep_);

    if (control_type == "MPC") {
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
        controller = ergodiclib::ilqrController(cartpole, Q, R, P, r, 7500, alpha, beta, eps);
    }

    // Create main callback
    timer_ = create_wall_timer(duration_, std::bind(&CartpoleControl::cartpole_main, this));
  }

private:
  // Main loop publisher and timer
  rclcpp::TimerBase::SharedPtr timer_;
  double rate_;
  int err;

  // Input for controls
  std::string control_type;
  std::string control_file;

  // Controls message
  std_msgs::msg::Float64 force_cmd;

  // MPC Control Variables
  double x_cart, v_cart, x_pole, v_pole;
  arma::vec x0({0.0, 0.0, PI, 0.0});
  arma::vec u0({0.0});
  arma::mat Q(4, 4, arma::fill::eye);
  arma::mat R(1, 1, arma::fill::eye);
  arma::mat P(4, 4, arma::fill::eye);
  arma::mat r(4, 1, arma::fill::zeros);

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
  ergodiclib::ilqrController controller;

  std::pair<arma::mat, arma::mat> trajectories;
  arma::mat X;
  arma::mat U;

  // Create publishers and subscribers
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr command_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

  // Publishes the current timestep of the simulation
  std_msgs::msg::UInt64 timestep_;
  rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr timestep_pub_;

  void cartpole_main()
  {
    if (control_type == "file") {
        err = fileController();
    }

    if (control_type == "MPC") {
        err = ModelPredictiveControl();
    }

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
      x_pole = ergodiclib::normalize_angle(cartpole_js.position[0]); 
      v_pole = cartpole_js.velocity[0];
    }
  }

  int fileController()
  {
    std::vector<float> control_input;
    std::ifstream controls(control_file);

    if (!controls.is_open()) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Control file could not be open");
        return -1;
    }
    
    std::string line;
    float num;
    while (std::getline(file, line)){
        std::istringstream iss(line);
        std::string value;
        while (std::getline(iss, value, ',')) {
            num = std::stof(value);
            control_input.push_back(num); 
        }
    }

    controls.close();

    for (unsigned int i = 0; i < control_input.size() i++){
        force_cmd.data = control_input[i];
        command_pub_->publish(force_cmd);
    }

    control_type = "";
    return 0;
  }

  int ModelPredictiveControl()
  {
    double controls;
    arma::vec curr_pos({x_cart, v_cart, x_pole, v_pole});
    trajectories = controller.ModelPredictiveControl(curr_pos, u0, 50, 100);
    X = trajectories.first;
    U = trajectories.second;

    for (unsigned int i = 0; i < U.n_cols; i++) {
        controls = U(0, 0);
        force_cmd.data = controls;
        command_pub_->publish(force_cmd);
    }

    return 0;
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
