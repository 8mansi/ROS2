from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([

        Node(
            package='driver_assistance_rl',
            executable='sim_node',
            name='sim_node',
            output='screen'
        ),

        Node(
            package='driver_assistance_rl',
            executable='sensor_node',
            name='sensor_node',
            output='screen'
        ),

        Node(
            package='driver_assistance_rl',
            executable='control_node',
            name='control_node',
            output='screen'
        ),

        Node(
            package='driver_assistance_rl',
            executable='rl_agent_node',
            name='rl_agent_node',
            output='screen'
        ),
    ])
