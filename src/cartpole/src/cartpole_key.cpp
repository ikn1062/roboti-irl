/// \file
/// \brief Cartpole Controller File used to command Cart in simulation and save Joint States
///
/// PARAMETERS:
///     rate (float): Rate of Main Loop
///     scale (float): Scale of force relative to teleop twist input
///     max_force (float): Max Force allowed to publish to cartpole sim
///     cart_path (string): Path to save the cart joint position outputs
///     pole_path (string): Path to save the pole joint position outputs
/// PUBLISHES:
///     /cartpole/timestep (std_msgs::msg::UInt64): Current Timestep of Simulation
///     /cartpole/cmd (std_msgs::msg::Float64): Command Publisher for cartpole sim
/// SUBSCRIBES:
///     /cmd_vel (geometry_msgs::msg::Twist): Subscribes to the Teleop Twist Keyboard
///     /cartpole/joint_state (sensor_msgs::msg::JointState): Receives joint states of cart and pole


#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/u_int64.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/twist.hpp"

using namespace std::chrono;

/// \brief Cartpole Keyboard Controller Node
class CartpoleControl : public rclcpp::Node
{
public:
  CartpoleControl()
  : Node("cartpole_key")
  {
    // CARTPOLE SIMULATION CONSTRUCTOR
    // Decalre Variables
    declare_parameter("rate", 10.0);
    declare_parameter("scale", 10.0);
    declare_parameter("max_force", 256.0);
    declare_parameter("cart_path", "./src/cartpole/demonstrations/cart_x.txt");
    declare_parameter("pole_path", "./src/cartpole/demonstrations/pole_x.txt");

    // Get Variables
    rate_ = get_parameter("rate").as_double();
    auto duration_ = std::chrono::duration<double>(1.0 / rate_);

    scale = get_parameter("scale").as_double();
    max_force = get_parameter("max_force").as_double();

    cart_path = get_parameter("cart_path").as_string();
    pole_path = get_parameter("pole_path").as_string();

    // Publishers and Subscribers
    timestep_pub_ = create_publisher<std_msgs::msg::UInt64>("cartpole/timestep", 50);
    command_pub_ = create_publisher<std_msgs::msg::Float64>("/cartpole/cmd", 10);
    keyboard_sub_ =
      create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      std::bind(&CartpoleControl::keyboard_cb, this, std::placeholders::_1));
    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/cartpole/joint_state",
      10,
      std::bind(&CartpoleControl::jointstate_cb_, this, std::placeholders::_1));

    // Initial Variables
    timestep_.data = 0;
    timestep_pub_->publish(timestep_);

    std::cout << scale << std::endl;
    std::cout << cart_path << std::endl;

    cart_output.open(cart_path);
    pole_output.open(pole_path);

    // Create main callback
    timer_ = create_wall_timer(duration_, std::bind(&CartpoleControl::cartpole_main, this));
  }

private:
  // Main loop publisher and timer
  rclcpp::TimerBase::SharedPtr timer_;
  double rate_;

  // File output
  std::string cart_path, pole_path;
  std::ofstream cart_output;
  std::ofstream pole_output;

  // Controller Parameters
  double scale;
  double max_force;
  double x_cart, v_cart, x_pole, v_pole;
  std_msgs::msg::Float64 force_cmd;

  // Create publishers and subscribers
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr command_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr keyboard_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

  // Publishes the current timestep of the simulation
  std_msgs::msg::UInt64 timestep_;
  rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr timestep_pub_;

  void cartpole_main()
  {
    timestep_.data += (1 / rate_) * 1000;       // converts to an int, note that this is in ms
    timestep_pub_->publish(timestep_);
  }

  void keyboard_cb(const geometry_msgs::msg::Twist & twist)
  {
    double force = twist.angular.z * scale;
    if (force > max_force) {
      force = max_force;
    }

    force_cmd.data = force;

    command_pub_->publish(force_cmd);
  }

  void jointstate_cb_(const sensor_msgs::msg::JointState & cartpole_js)
  {
    std::string joint_name = cartpole_js.name[0];

    if (joint_name == "slider_to_cart") {
      x_cart = cartpole_js.position[0];
      v_cart = cartpole_js.velocity[0];
      cart_output << x_cart << ", " << v_cart << "\n";
    } else {
      x_pole = normalize_angle(cartpole_js.position[0]);
      v_pole = cartpole_js.velocity[0];
      pole_output << x_pole << ", " << v_pole << "\n";
    }
  }

  double normalize_angle(const double & rad)
  {
    double const PI = 3.14159265359;
    double new_rad;

    new_rad = fmod(rad, 2 * PI);
    new_rad = fmod(new_rad + 2 * PI, 2 * PI);
    if (new_rad > PI) {
      new_rad -= 2 * PI;
    }

    new_rad = new_rad <= 0 ? (PI - abs(new_rad)) : -1 * (PI - abs(new_rad));

    return new_rad;
  }
};

/// \brief Main function to run Controller Node
/// \param argc Inputs
/// \param argv Inputs
/// \return None
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Creating Cartpole Key Controller Node. ");

  rclcpp::spin(std::make_shared<CartpoleControl>());

  rclcpp::shutdown();
  return 0;
}
