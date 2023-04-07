from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Whether to use simulation clock [true/false]'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                "robot_description":
                Command(['xacro', ' ',
                        PathJoinSubstitution([
                            FindPackageShare(package='cartpole'),
                            'urdf', 'xacro', 'cartpole_gazebo_urdf.urdf.xacro'])])}
                        ],
            arguments=[PathJoinSubstitution([
                FindPackageShare(package='cartpole'),
                'urdf', 'xacro', 'cartpole_gazebo_urdf.urdf.xacro'])]
        ),
    ])
