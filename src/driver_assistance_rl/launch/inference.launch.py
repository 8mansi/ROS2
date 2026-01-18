from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch file for inference/testing the trained PPO agent in ROS2.
    Runs the agent without training, using the saved model.
    """
    
    return LaunchDescription([
        # Simulation node - runs PyBullet with GUI
        Node(
            package='driver_assistance_rl',
            executable='sim_node',
            name='sim_node',
            output='screen',
            emulate_tty=True
        ),

        # Sensor node - processes simulation state
        Node(
            package='driver_assistance_rl',
            executable='sensor_node',
            name='sensor_node',
            output='screen',
            emulate_tty=True
        ),

        # Control node - applies velocities to robot
        Node(
            package='driver_assistance_rl',
            executable='control_node',
            name='control_node',
            output='screen',
            emulate_tty=True
        ),

        # RL Agent node - in inference mode
        Node(
            package='driver_assistance_rl',
            executable='rl_agent_node',
            name='rl_agent_node',
            output='screen',
            emulate_tty=True
        ),
    ])
