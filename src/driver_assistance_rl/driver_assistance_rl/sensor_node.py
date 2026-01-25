#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import time

class SensorNode(Node):
    """
    Sensor node that receives state from simulation and publishes it for the RL agent.
    """
    
    def __init__(self):
        super().__init__('sensor_node')
        
        # Publisher for state
        self.pub = self.create_publisher(Float32MultiArray, '/rl/state', 10)
        
        # Subscriber to simulation step events
        self.sub = self.create_subscription(
            Float32MultiArray, '/sim/step', self.state_callback, 10
        )
        
        # Subscriber to reset requests
        self.reset_sub = self.create_subscription(
            Float32MultiArray, '/sim/reset', self.reset_callback, 10
        )
        
        # System synchronization
        self.system_ready = False
        self.startup_delay = 5.2  # seconds
        self.startup_time = time.time()
        self.first_episode = True
        
        # Post-episode reset synchronization
        self.awaiting_post_episode_reset = False
        self.post_episode_reset_time = None
        self.post_episode_reset_delay = 2.0  # seconds to wait after reset request
        
        self.get_logger().info("SensorNode initialized")
    
    def reset_callback(self, msg):
        """Handle reset requests to reinitialize state publishing"""
        self.get_logger().info("Reset signal received in sensor_node")
        if msg.data[0] == 0.0:
            # Training finished, stop reacting
            self.get_logger().info("Received stop signal. No further resets.")
            return

        self.first_episode = False
        self.awaiting_post_episode_reset = True
        self.post_episode_reset_time = time.time()
        self.system_ready = False  # Pause publishing until sync complete
        
    def state_callback(self, msg):
        """
        Receive state from simulation and republish for RL agent.
        This node acts as a passthrough that could be extended for:
        - Additional sensor processing
        - Noise injection
        - State filtering
        """
        # Handle post-episode reset synchronization
        if self.awaiting_post_episode_reset:
            elapsed = time.time() - self.post_episode_reset_time
            if elapsed < self.post_episode_reset_delay:
                return  # Silently wait
            else:
                self.awaiting_post_episode_reset = False
                self.system_ready = True
                self.get_logger().info("Sensor node re-synchronized after episode reset!")
                return  # Skip this message
        
        # Check if system is ready
        if not self.system_ready:
            elapsed = time.time() - self.startup_time
            if elapsed < self.startup_delay:
                return  # Silently wait
            else:
                self.system_ready = True
                self.get_logger().info("Sensor node ready!")
        
        # State directly from simulation: [b1, b2, b3, b4, beam_dist, left_lane, right_lane, lane_offset, heading_error]
        state = np.array(msg.data, dtype=np.float32)
        
        # Publish state for RL agent
        out_msg = Float32MultiArray()
        out_msg.data = state.tolist()
        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
