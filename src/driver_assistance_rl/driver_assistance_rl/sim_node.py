# sim_node.py
import rclpy
from rclpy.node import Node
import pybullet as p
import pybullet_data
import time

class SimNode(Node):
    def __init__(self):
        super().__init__('sim_node')

        # PyBullet setup
        # p.connect(p.GUI)
        self.physicsClient = p.connect(p.GUI_SERVER)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(0.05)

        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            "husky/husky.urdf", [0, 0, 0.1]
        )

        self.timer = self.create_timer(0.05, self.step)

    def step(self):
        p.stepSimulation()

def main():
    rclpy.init()
    node = SimNode()
    rclpy.spin(node)
    node.destroy_node()
    p.disconnect()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
