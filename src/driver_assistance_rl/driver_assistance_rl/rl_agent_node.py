#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import time
from driver_assistance_rl.ppo_agent import PPOAgent

class RLAgentNode(Node):
    """
    RL Agent node that:
    1. Receives state from sensor node
    2. Computes action using PPO agent
    3. Publishes action command to control node
    """
    
    def __init__(self, mode='inference', model_path=None):
        super().__init__('rl_agent_node')
        
        # Mode can be 'inference' or 'training'
        self.mode = mode
        self.model_path = model_path or "./ppo_driver_model.pth"
        
        # Initialize PPO agent
        self.agent = PPOAgent(
            state_dim=9,
            action_dim=4
        )
        
        # Load pretrained model if available
        try:
            self.agent.load_model(self.model_path)
            self.get_logger().info(f"Loaded model from {self.model_path}")
        except Exception as e:
            self.get_logger().warn(f"Could not load model: {e}. Starting with random weights.")
        
        # Subscriber to state
        self.sub = self.create_subscription(
            Float32MultiArray,
            '/rl/state',
            self.state_callback,
            10
        )
        
        # Publisher for commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Publisher for diagnostics (action info)
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            '/rl/action',
            10
        )
        
        # For training: store transitions
        self.last_state = None
        self.last_action = None
        self.last_logprob = None
        
        # System synchronization
        self.system_ready = False
        self.startup_delay = 5.5  # seconds to wait (0.5 second after sim_node)
        self.startup_time = time.time()
        
        self.get_logger().info(f"RLAgentNode initialized in {mode} mode")
        
    def state_callback(self, msg):
        """
        Receive state and compute action.
        State format: [b1, b2, b3, b4, beam_dist, left_lane, right_lane, lane_offset, heading_error]
        """
        # Check if system is ready before processing
        if not self.system_ready:
            elapsed = time.time() - self.startup_time
            if elapsed < self.startup_delay:
                return  # Silently wait
            else:
                self.system_ready = True
                self.get_logger().info("RL Agent ready!")
        
        state = np.array(msg.data, dtype=np.float32)
        
        # Compute action using PPO agent
        action, logprob = self.agent.act(state)
        
        # Create and publish command
        cmd = Twist()
        
        if action == 0:  # straight
            cmd.linear.x = 4.0
            cmd.angular.z = 0.0
        elif action == 1:  # left
            cmd.linear.x = 3.0
            cmd.angular.z = 6.0
        elif action == 2:  # right
            cmd.linear.x = 3.0
            cmd.angular.z = -6.0
        elif action == 3:  # slow
            cmd.linear.x = 1.5
            cmd.angular.z = 0.0
        
        self.cmd_pub.publish(cmd)
        
        # Publish action diagnostics
        action_msg = Float32MultiArray()
        action_msg.data = [float(action), logprob]
        self.action_pub.publish(action_msg)
        
        # Store for training
        if self.mode == 'training':
            self.last_state = state
            self.last_action = action
            self.last_logprob = logprob
        
        self.get_logger().debug(f"Action: {action}, LogProb: {logprob:.4f}")
        
    def remember_transition(self, reward, done):
        """
        Store transition in agent memory for training.
        Called from training node after reward computation.
        """
        if self.mode == 'training' and self.last_state is not None:
            self.agent.remember(
                self.last_state,
                self.last_action,
                self.last_logprob,
                reward,
                done
            )
    
    def train(self, force=False, entropy_coef=None):
        """Train the agent"""
        if self.mode != 'training':
            return None
        return self.agent.train(force=force, entropy_coef=entropy_coef)
    
    def save_model(self, path):
        """Save the model"""
        self.agent.save_model(path)
        self.get_logger().info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model"""
        self.agent.load_model(path)
        self.get_logger().info(f"Model loaded from {path}")


def main():
    rclpy.init()
    
    # Default to inference mode
    mode = 'inference'
    # Uncomment for training:
    mode = 'training'
    
    node = RLAgentNode(mode=mode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
