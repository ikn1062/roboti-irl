from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.actions import SetLaunchConfiguration
from launch.conditions import IfCondition
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
        DeclareLaunchArgument(
            name='use_rviz',
            default_value='true',
            description='Whether to launch rviz2 [true/false]'),
        DeclareLaunchArgument(
            'r_frame',
            default_value='world',
            description='Name of the fixed frame in RVIZ.'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                "robot_description":
                Command(['xacro', ' ',
                        PathJoinSubstitution([
                            FindPackageShare(package='cartpole'),
                            'urdf', 'xacro', 'cartpole_urdf.urdf.xacro'])])}
                        ],
            arguments=[PathJoinSubstitution([
                FindPackageShare(package='cartpole'),
                'urdf', 'xacro', 'cartpole_urdf.urdf.xacro'])]
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher'),
        DeclareLaunchArgument(
            'rviz_config_name',
            default_value='simulation.rviz',
            description="Set path to configuration"),
        SetLaunchConfiguration('rviz_config_file',
                               PathJoinSubstitution([
                                FindPackageShare(
                                    package='cartpole'),
                                'config',
                                LaunchConfiguration('rviz_config_name')])),
        Node(
            condition=IfCondition(LaunchConfiguration('use_rviz')),
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d',
                       PathJoinSubstitution([
                        FindPackageShare(package='cartpole'),
                        'rviz', LaunchConfiguration('rviz_config_file')]
                                            ),
                       LaunchConfiguration('r_frame')],
            on_exit=Shutdown()
        ),
    ])
