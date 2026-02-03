from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch file for training the PPO agent in ROS2.
    Starts all necessary nodes for training:
    - sim_node: Runs PyBullet simulation with traffic
    - sensor_node: Processes sensor data
    - control_node: Applies commands to simulation
    - rl_agent_node: PPO agent (in training mode)
    - training_node: Orchestrates training loop
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

        # # Control node - applies velocities to robot
        # Node(
        #     package='driver_assistance_rl',
        #     executable='control_node',
        #     name='control_node',
        #     output='screen',
        #     emulate_tty=True
        # ),

        # RL Agent node - in training mode
        Node(
            package='driver_assistance_rl',
            executable='rl_agent_node',
            name='rl_agent_node',
            output='screen',
            emulate_tty=True,
            parameters=[{'mode': 'training'}]
        ),

        # Training coordinator node
        Node(
            package='driver_assistance_rl',
            executable='training_node',
            name='training_node',
            output='screen',
            emulate_tty=True
        ),
    ])
