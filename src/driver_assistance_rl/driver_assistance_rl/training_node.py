#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import time
import os
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist


class TrainingNode(Node):
    """
    Training node that orchestrates the full training loop.
    Similar to DriverAssistanceSim.run() but adapted for ROS2.
    """
    
    def __init__(self):
        super().__init__('training_node')
        
        # Training hyperparameters
        self.EPISODES = 500
        self.STEPS_PER_EPISODE = 500
        self.timeStep = 0.1
        self.max_range = 3
        self.lane_width = 2.0
        self.offset = 0.1
        
        # RL hyperparameters
        self.initial_entropy = 0.5
        self.final_entropy = 0.01
        
        # State tracking
        self.current_episode = 0
        self.current_step = 0
        self.episode_reward = 0
        self.episode_metrics = {
            "episode": [],
            "step": [],
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "reward": []
        }
        
        # For training mode flag
        self.training_active = False
        self.model_path = "./ppo_driver_model.pth"
        
        # Publishers for training diagnostics
        self.training_pub = self.create_publisher(
            Float32MultiArray, '/training/episode_stats', 10
        )
        
        # Publisher for simulation reset
        self.reset_pub = self.create_publisher(
            Float32MultiArray, '/sim/reset', 10
        )
        
        # Subscribers to simulation and agent
        self.state_sub = self.create_subscription(
            Float32MultiArray, '/rl/state', self.state_callback, 10
        )
        
        self.action_sub = self.create_subscription(
            Float32MultiArray, '/rl/action', self.action_callback, 10
        )
        
        # Subscriber to collision/done status
        self.done_sub = self.create_subscription(
            Float32MultiArray, '/sim/done', self.done_callback, 10
        )
        
        # Command publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer for episode management
        self.step_timer = self.create_timer(self.timeStep, self.training_step_callback)
        
        # Storage for current transition
        self.current_state = None
        self.current_action = None
        self.current_logprob = None
        self.prev_reward = 0
        self.step_count = 0
        self.done = False  # Track episode termination
        
        # System synchronization
        self.system_ready = False
        self.startup_delay = 6  # seconds to wait (1 second after sim_node)
        self.startup_time = time.time()
        self.first_state_received = False
        self.first_action_received = False
        
        # Post-episode synchronization
        self.awaiting_post_episode_sync = False
        self.post_episode_sync_time = None
        self.post_episode_sync_delay = 2  # seconds to wait between episodes
        
        self.get_logger().info("TrainingNode initialized")
        
    def state_callback(self, msg):
        """Receive state from sensor"""
        self.current_state = np.array(msg.data, dtype=np.float32)
        self.first_state_received = True
        self.training_step_callback()
        
    def action_callback(self, msg):
        """Receive action from agent"""
        self.current_action = int(msg.data[0])
        self.current_logprob = float(msg.data[1])
        self.first_action_received = True
        
    def done_callback(self, msg):
        """Receive collision/lane exit status from simulation"""
        collision = bool(msg.data[0])
        lane_exit = bool(msg.data[1])
        self.done = bool(msg.data[2])
        
        if collision:
            self.get_logger().warn("Collision detected!")
        if lane_exit:
            self.get_logger().warn("Lane exit detected!")
        
    def compute_reward(self, state, action, done):
        """
        Compute reward based on state and action.
        Adapted from DriverAssistanceSim.compute_reward()
        """
        reward = 0
        b1, b2, b3, b4 = state[0], state[1], state[2], state[3]
        beam_dist = state[4]
        left_lane = state[5]
        right_lane = state[6]
        lane_offset = abs(state[7])
        heading_error = abs(state[8])
        
        # Obstacle avoidance
        max_beam = max(b1, b2, b3, b4)
        reward += (1 - max_beam) * 3
        
        if max_beam > 0.85:
            reward -= 12
        
        if left_lane or right_lane:
            reward -= 8.0
        
        # Encourage straight driving
        reward += 2.0 * (1.0 - lane_offset)
        reward += 0.5 * (1.0 - heading_error)
        
        left_clear = 1 - max(b1, b3)
        right_clear = 1 - max(b2, b4)
        SAFE = max_beam < 0.30
        DANGER = max_beam > 0.45
        
        if action == 0:  # STRAIGHT
            if SAFE:
                reward += 1.5
            elif max_beam > 0.45:
                reward -= 4.0
        elif action in (1, 2):  # left/right
            if SAFE:
                reward -= 0.8
            else:
                if action == 1 and left_clear > right_clear + 0.15:
                    reward += 2.0
                if action == 2 and right_clear > left_clear + 0.15:
                    reward += 2.0
        elif action == 3:  # slow
            if max_beam > 0.55:
                reward += 1.0
            else:
                reward -= 0.5
        
        if done:
            reward -= 20
        
        return reward
        
    def training_step_callback(self):
        """Main training loop step"""
        # Handle post-episode synchronization
        if self.awaiting_post_episode_sync:
            elapsed = time.time() - self.post_episode_sync_time
            if elapsed < self.post_episode_sync_delay:
                self.get_logger().info(
                    f"Post-episode sync: {elapsed:.1f}/{self.post_episode_sync_delay}s"
                )
                return
            else:
                self.awaiting_post_episode_sync = False
                self.training_active = True
                self.get_logger().info(f"✓ Episode {self.current_episode} - System re-synchronized!")
                return
        
        # Check if system is ready (initial startup)
        if not self.system_ready:
            elapsed = time.time() - self.startup_time
            if elapsed < self.startup_delay:
                self.get_logger().info(
                    f"System startup: {elapsed:.1f}/{self.startup_delay}s | "
                    f"State: {self.first_state_received} | Action: {self.first_action_received}"
                )
                return
            elif not self.first_state_received or not self.first_action_received:
                self.get_logger().warn(
                    f"Waiting for all nodes... State: {self.first_state_received}, Action: {self.first_action_received}"
                )
                return
            else:
                self.system_ready = True
                self.training_active = True
                self.get_logger().info("System synchronized! Starting training")
                self.current_episode = 1
        
        if not self.training_active or self.current_state is None:
            return
        
        self.step_count += 1

        
        # Compute reward from previous transition
        if self.current_action is not None:
            # Use actual done flag from simulation
            reward = self.compute_reward(self.current_state, self.current_action, self.done)
            self.episode_reward += reward
            print(f"Step {self.step_count}, Reward: {reward:.2f}, Action: {self.current_action}, Total: {self.episode_reward:.2f}")
            
            # Store in agent memory (in actual implementation,
            # this would call agent.remember through service or action)
            
        self.step_count += 1
        if self.step_count >= self.STEPS_PER_EPISODE or self.done:
            self.end_episode()
        
    def end_episode(self):
        """End current episode and prepare for next"""
        self.current_episode += 1
        
        # Update metrics
        self.episode_metrics["episode"].append(self.current_episode)
        self.episode_metrics["step"].append(self.step_count)
        self.episode_metrics["reward"].append(self.episode_reward)
        
        # Log episode stats
        stats_msg = Float32MultiArray()
        stats_msg.data = [
            float(self.current_episode),
            float(self.step_count),
            self.episode_reward
        ]
        self.training_pub.publish(stats_msg)
        print("Episode {} completed. Steps: {}, Reward: {:.2f}".format(
            self.current_episode, self.step_count, self.episode_reward
        ))
        
        # Reset for next episode
        self.step_count = 0
        self.episode_reward = 0
        self.done = False  # Reset done flag for new episode
        
        # Publish reset request to simulation node
        reset_msg = Float32MultiArray()
        reset_msg.data = [1.0]  # Simple flag to trigger reset
        self.reset_pub.publish(reset_msg)
        self.get_logger().info(f"Episode {self.current_episode - 1} - Reset signal sent to sim_node")
        
        # Start post-episode synchronization
        self.training_active = False  # Pause training
        self.awaiting_post_episode_sync = True
        self.post_episode_sync_time = time.time()
        self.get_logger().info(f"Episode {self.current_episode - 1} → {self.current_episode}: Starting post-episode sync...")
        
        # Check if training is complete
        if self.current_episode > self.EPISODES:
            self.stop_training()
    
    def start_training(self):
        print("Starting training")
        """Start the training process"""
        self.training_active = True
        self.current_episode = 0
        self.step_count = 0
        self.episode_reward = 0
        self.get_logger().info("Training started")
        
    def stop_training(self):
        print("Stopping training")
        """Stop training and save model"""
        self.training_active = False
        self.get_logger().info("Training completed")
        
        # Save final model
        try:
            # Would need to get agent from rl_agent_node
            self.get_logger().info(f"Training metrics saved")
        except Exception as e:
            self.get_logger().error(f"Error saving model: {e}")


def main():
    rclpy.init()
    node = TrainingNode()
    
    # Don't manually start training - it will auto-start after synchronization
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
