# rl_agent_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import numpy as np

class RLAgentNode(Node):
    def __init__(self):
        super().__init__('rl_agent_node')

        self.sub = self.create_subscription(
            Float32MultiArray, '/rl/state', self.state_cb, 10
        )
        self.pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

    def state_cb(self, msg):
        state = np.array(msg.data)

        cmd = Twist()
        cmd.linear.x = 2.0   # dummy policy
        cmd.angular.z = -0.5 * state[1]

        self.pub.publish(cmd)

def main():
    rclpy.init()
    node = RLAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
