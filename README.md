# Driver Assistance RL - ROS2 Implementation

This is a ROS2 adaptation of `DriverAssistanceSim.py`, a PyBullet-based reinforcement learning simulation for autonomous driving with collision avoidance.

## Overview

The system is distributed across multiple ROS2 nodes:

- **sim_node**: Runs the PyBullet physics simulation with traffic cars and sensor rays
- **sensor_node**: Processes and publishes state information
- **control_node**: Applies velocity commands to the robot
- **rl_agent_node**: Runs the PPO reinforcement learning agent
- **training_node**: Orchestrates the training loop (episodic learning)

## Architecture

### System Flow

```
PyBullet Simulation (sim_node)
    ↓
State Publishing (/sim/step)
    ↓
Sensor Processing (sensor_node)
    ↓
RL Agent (rl_agent_node) → Action Decision
    ↓
Command Publishing (/cmd_vel)
    ↓
Control Execution (control_node)
    ↓
Velocity Applied to Robot
    ↓
Training Coordination (training_node)
```

### Topics

- `/sim/step`: Published by sim_node with raw simulation state
- `/rl/state`: Published by sensor_node with processed state
- `/rl/action`: Published by rl_agent_node with action and log probability
- `/cmd_vel`: Control commands to apply robot velocities
- `/control/velocity`: Diagnostics from control_node
- `/training/episode_stats`: Training metrics from training_node

## State Representation

The state vector (9 dimensions) includes:

```python
[
    b1,              # Left beam distance (normalized)
    b2,              # Right beam distance (normalized)
    b3,              # Angled-left beam distance (normalized)
    b4,              # Angled-right beam distance (normalized)
    beam_dist,       # Front center beam distance (normalized)
    left_lane,       # Lane crossing flag (left)
    right_lane,      # Lane crossing flag (right)
    lane_offset,     # Normalized position in lane
    heading_error    # Normalized yaw angle
]
```

## Actions

The agent can take 4 discrete actions:

- **0**: Go straight (v=4.0, ω=0)
- **1**: Turn left (v=3.0, ω=6.0)
- **2**: Turn right (v=3.0, ω=-6.0)
- **3**: Slow/stop (v=1.5, ω=0)

## Building

```bash
cd ~/driver_assistance/ROS2
colcon build --symlink-install
```

## Running

### For Inference (Testing)

```bash
# Terminal 1: Source environment
source install/setup.bash
ros2 launch driver_assistance_rl inference.launch.py
```

### For Training

```bash
# Terminal 1: Source environment
source install/setup.bash
ros2 launch driver_assistance_rl train.launch.py
```

## Key Features Adapted from DriverAssistanceSim

1. **Multi-ray sensing**: 4 parallel beam sensors + central beam sensor
2. **Traffic simulation**: Random traffic cars with collision detection
3. **Lane detection**: Visual lane crossing detection via camera
4. **Reward shaping**: Complex reward function balancing safety, efficiency, and comfort
5. **PPO training**: Proximal Policy Optimization with entropy regularization

## Configuration

Edit these files to adjust parameters:

- `sim_node.py`: Simulation parameters (num_cars, lane_width, max_range, etc.)
- `ppo_agent.py`: RL hyperparameters (learning rates, clip_eps, batch_size, etc.)
- `training_node.py`: Training loop parameters (EPISODES, STEPS_PER_EPISODE, etc.)

## Training Output

After training, the following are generated:

- `ppo_driver_model.pth`: Trained model weights
- Metrics and trajectory plots (if plotting is enabled in training_node)

## Differences from Original DriverAssistanceSim.py

1. **Distributed architecture**: Simulation decoupled from RL agent
2. **ROS2 communication**: All inter-process communication via ROS2 topics/services
3. **Asynchronous processing**: Each node runs independently
4. **Scalability**: Easy to extend with additional sensors or actuators

## Troubleshooting

### PyBullet GUI not appearing
- Check that X11 forwarding is enabled (if running remotely)
- Ensure `p.connect(p.GUI)` is used (not `p.GUI_SERVER`)

### Nodes not communicating
- Verify all nodes are running: `ros2 node list`
- Check topics: `ros2 topic list`
- Debug: `ros2 topic echo /rl/state`

### Training not converging
- Adjust learning rates in `ppo_agent.py`
- Modify reward shaping in `training_node.compute_reward()`
- Increase training episodes in `training_node.py`

## Future Improvements

1. Multi-agent training (multiple robots learning simultaneously)
2. Transfer learning from pre-trained models
3. Real robot deployment
4. Additional sensor modalities (LIDAR, camera)
5. End-to-end learning with camera inputs

## References

- Original code: DriverAssistanceSim.py, PPO.py
- ROS2 Documentation: https://docs.ros.org/en/humble/
- PyBullet: https://pybullet.org/
- PPO Algorithm: Schulman et al., https://arxiv.org/abs/1707.06347
