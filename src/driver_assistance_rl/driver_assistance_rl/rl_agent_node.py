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
            state_dim=10,
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
        self.first_episode = True
        
        # Post-episode reset synchronization
        self.awaiting_post_episode_reset = False
        self.post_episode_reset_time = None
        self.post_episode_reset_delay = 1.5  # seconds to wait after reset request
        
        # Subscriber to reset requests
        self.reset_sub = self.create_subscription(
            Float32MultiArray,
            '/sim/reset',
            self.reset_callback,
            10
        )
        
        # Subscriber to transition data from training node
        self.transition_sub = self.create_subscription(
            Float32MultiArray,
            '/training/transition',
            self.transition_callback,
            10
        )
        
        # Subscriber to training trigger
        self.train_trigger_sub = self.create_subscription(
            Float32MultiArray,
            '/training/do_training',
            self.train_trigger_callback,
            10
        )
        
        # Subscriber to D* path planning for diagnostics
        self.path_sub = self.create_subscription(
            Float32MultiArray,
            '/planning/dstar_path',
            self.path_callback,
            10
        )
        
        # Subscriber to SLAM lane map for diagnostics
        self.slam_sub = self.create_subscription(
            Float32MultiArray,
            '/slam/lane_map',
            self.slam_callback,
            10
        )
        
        self.get_logger().info(f"RLAgentNode initialized in {mode} mode")
        
    def reset_callback(self, msg):
        """Handle reset requests to reinitialize action publishing"""
        # self.get_logger().info("Reset signal received in rl_agent_node - keeping model intact")
        if msg.data[0] == 0.0:
            # Training finished, stop reacting
            self.get_logger().info("Received stop signal. No further resets.")
            return

        self.first_episode = False
        self.awaiting_post_episode_reset = True
        self.post_episode_reset_time = time.time()
        self.system_ready = False  # Pause action publishing until sync complete
        # Reset the last stored action/state
        self.last_state = None
        self.last_action = None
        self.last_logprob = None
        self.save_model(self.model_path)
        
    
    def transition_callback(self, msg):
        """Receive reward and done flag, store transition in agent memory"""
        if self.mode != 'training' or self.last_state is None:
            return
        
        reward = float(msg.data[0])
        done = bool(msg.data[1])
        
        # Store transition in agent memory
        self.agent.remember(
            self.last_state,
            self.last_action,
            self.last_logprob,
            reward,
            done
        )
        self.get_logger().debug(f"Transition stored: reward={reward:.2f}, done={done}")
    
    def train_trigger_callback(self, msg):
        """Receive training trigger and perform gradient updates"""
        if self.mode != 'training':
            return
        
        force = bool(msg.data[0]) if len(msg.data) > 0 else False
        
        result = self.agent.train(force=force)
        if result is not None:
            loss, actor_loss, critic_loss, entropy = result
            print("\n" + "="*60)
            self.get_logger().info(
                f"PPO Update - Loss: {loss:.4f}, Actor: {actor_loss:.4f}, "
                f"Critic: {critic_loss:.4f}, Entropy: {entropy:.4f}"
            )
            print("\n" + "="*60)
        
    def state_callback(self, msg):
        """
        Receive state and compute action.
        State format: [b1, b2, b3, b4, beam_dist, left_lane, right_lane, lane_offset, heading_error]
        """
        # Handle post-episode reset synchronization
        if self.awaiting_post_episode_reset:
            elapsed = time.time() - self.post_episode_reset_time
            if elapsed < self.post_episode_reset_delay:
                return  # Silently wait
            else:
                self.awaiting_post_episode_reset = False
                self.system_ready = True
                self.get_logger().info("RL Agent re-synchronized after episode reset!")
                return  # Skip this message
        
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
    
    def path_callback(self, msg):
        """Receive D* path planning information for diagnostics"""
        if len(msg.data) >= 5:
            robot_x, robot_y, goal_x, goal_y, obstacle_count = msg.data[:5]
            self.get_logger().debug(
                f"D* Plan: Robot at ({robot_x:.1f}, {robot_y:.1f}), "
                f"Goal at ({goal_x:.1f}, {goal_y:.1f}), Obstacles: {int(obstacle_count)}"
            )
    
    def slam_callback(self, msg):
        """Receive SLAM lane features for diagnostics"""
        if len(msg.data) >= 3:
            # SLAM publishes lane features as [a1, b1, c1, a2, b2, c2, ...]
            num_features = len(msg.data) // 3
            self.get_logger().debug(f"SLAM detected {num_features} lane features")
    
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
