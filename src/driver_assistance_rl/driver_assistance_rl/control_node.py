# control_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pybullet as p

WHEEL_JOINTS = [2, 3, 4, 5]
MAX_FORCE = 200
WHEEL_DISTANCE = 0.55

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10
        )

    def cmd_callback(self, msg):
        linear = msg.linear.x
        angular = msg.angular.z

        left = linear - angular * WHEEL_DISTANCE / 2
        right = linear + angular * WHEEL_DISTANCE / 2

        for i, j in enumerate(WHEEL_JOINTS):
            vel = left if i % 2 == 0 else right
            p.setJointMotorControl2(
                bodyIndex=0,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=MAX_FORCE
            )

def main():
    rclpy.init()
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
