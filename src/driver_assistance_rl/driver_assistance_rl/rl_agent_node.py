# rl_agent_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np

from driver_assistance_rl.ppo_agent import PPOAgent  
class RLAgentNode(Node):
    def __init__(self):
        super().__init__('rl_agent_node')

        self.agent = PPOAgent(
            state_dim=9,
            action_dim=4
        )

        self.sub = self.create_subscription(
            Float32MultiArray,
            '/rl/state',
            self.state_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

    def state_callback(self, msg):
        state = np.array(msg.data, dtype=np.float32)

        action, logprob = self.agent.act(state)

        cmd = Twist()

        if action == 0:        # straight
            cmd.linear.x = 4.0
        elif action == 1:      # left
            cmd.linear.x = 3.0
            cmd.angular.z = 6.0
        elif action == 2:      # right
            cmd.linear.x = 3.0
            cmd.angular.z = -6.0
        elif action == 3:      # slow
            cmd.linear.x = 1.5

        self.cmd_pub.publish(cmd)

        # Reward + remember() usually happens
        # after next state is received (temporal logic)

def main():
    rclpy.init()
    node = RLAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
