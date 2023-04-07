import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import xacro

def generate_launch_description():
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py']),
             )

    xacro_file = os.path.join(get_package_share_directory('cartpole'),
                              'urdf', 'xacro',
                              'cartpole_urdf.urdf.xacro')
    
    config_file = os.path.join(get_package_share_directory('cartpole'),
                               'config',
                               'cart_pole_controller.yaml')
    
    robot_desc = Command(['xacro', ' ',
                        PathJoinSubstitution([
                            FindPackageShare(package='cartpole'),
                            'urdf', 'xacro', 'cartpole_urdf.urdf.xacro'])])

    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    params = {'robot_description': doc.toxml()}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'cartpole'],
                        output='screen')

    load_joint_state_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen'
    )

    load_joint_trajectory_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'effort_controllers'],
        output='screen'
    )

    return LaunchDescription([
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[load_joint_state_controller],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_controller,
                on_exit=[load_joint_trajectory_controller],
            )
        ),
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[robot_desc, config_file],
            output="both"
        ),
        Node(
            package="controller_manager",
            executable="spawner",   
            arguments=["effort_controllers", "--controller-manager", "/controller_manager"],
            ),
        Node(
           package="controller_manager",
           executable="spawner",    
            arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
            )
    ])
