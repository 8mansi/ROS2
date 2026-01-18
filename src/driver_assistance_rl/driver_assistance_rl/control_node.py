#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
import pybullet as p

class ControlNode(Node):
    """
    Control node that applies wheel velocities to the robot in the simulation.
    Implements the set_robot_wheel_velocities logic from DriverAssistanceSim.
    """
    
    def __init__(self):
        super().__init__('control_node')
        
        # Constants
        self.WHEEL_JOINTS = [2, 3, 4, 5]
        self.WHEEL_DISTANCE = 0.55
        self.MAX_FORCE = 200
        
        self.p = p
        
        # Publisher for diagnostics
        self.diag_pub = self.create_publisher(Float32MultiArray, '/control/velocity', 10)
        
        # Subscription to command velocity
        self.sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10
        )
        
        # Store robot ID
        self.robot_id = None
        self.get_logger().info("ControlNode initialized")
        
    def cmd_callback(self, msg):
        """Apply command velocities to robot wheels"""
        if self.robot_id is None:
            self.get_logger().debug("Waiting for robot ID...")
            try:
                # Robot is typically ID 1 (after plane 0)
                self.robot_id = 1
            except:
                return
                
        linear = msg.linear.x
        angular = msg.angular.z
        
        # Convert to wheel velocities
        left = linear - angular * self.WHEEL_DISTANCE / 2
        right = linear + angular * self.WHEEL_DISTANCE / 2
        
        # Apply to wheels
        for i, j in enumerate(self.WHEEL_JOINTS):
            vel = left if i % 2 == 0 else right
            try:
                self.p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=j,
                    controlMode=self.p.VELOCITY_CONTROL,
                    targetVelocity=vel,
                    force=self.MAX_FORCE
                )
            except Exception as e:
                self.get_logger().debug(f"Control error: {e}")
        
        # Publish diagnostics
        diag_msg = Float32MultiArray()
        diag_msg.data = [linear, angular, left, right]
        self.diag_pub.publish(diag_msg)


def main():
    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
