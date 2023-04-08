/// \file
/// \brief INCLUDE BRIEF HERE
///
/// PARAMETERS:
///     NAME (TYPE): DESC
/// PUBLISHES:
///     pub_var_name (MSG TYPE): DESC
/// SUBSCRIBES:
///     sub_var_name (MSG TYPE): DESC
/// SERVERS:
///     nusim/reset (Empty): Resets the turtlebot to the original position in the world frame


#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/u_int64.hpp"
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
    declare_parameter("scale", 64.0);
    declare_parameter("max_force", 256.0);

    // Get Variables
    rate_ = get_parameter("rate").as_double();
    auto duration_ = std::chrono::duration<double>(1.0 / rate_);

    scale = get_parameter("scale").as_double();
    max_force = get_parameter("max_force").as_double();

    // Publishers and Subscribers
    timestep_pub_ = create_publisher<std_msgs::msg::UInt64>("~/timestep", 50);
    command_pub_ = create_publisher<std_msgs::msg::Float64>("/cartpole/cmd", 10);
    keyboard_sub_ = create_subscription<geometry_msgs::msg::Twist>("/cmd_vel", 10, std::bind(&CartpoleControl::keyboard_cb, this, std::placeholders::_1));

    // Initial Variables
    timestep_.data = 0;
    timestep_pub_->publish(timestep_);

    // Create main callback
    timer_ = create_wall_timer(duration_, std::bind(&CartpoleControl::cartpole_main, this));
    }

private:
    // Main loop publisher and timer
    rclcpp::TimerBase::SharedPtr timer_;
    double rate_;

    // Controller Parameters
    double scale;
    double max_force;
    std_msgs::msg::Float64 force_cmd;

    // Create publishers and subscribers
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr command_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr keyboard_sub_;

    // Publishes the current timestep of the simulation
    std_msgs::msg::UInt64 timestep_;
    rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr timestep_pub_;

    void cartpole_main() 
    {
        timestep_.data += (1 / rate_) * 1000;   // converts to an int, note that this is in ms
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
};


int main(int argc, char ** argv)
    {
    rclcpp::init(argc, argv);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Creating Cartpole Controller Node. ");

    rclcpp::spin(std::make_shared<CartpoleControl>());

    rclcpp::shutdown();
    return 0;
}
