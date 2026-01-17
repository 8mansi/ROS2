import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import pybullet as p
import numpy as np
import time

class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')
        self.pub = self.create_publisher(Float32MultiArray, '/rl/state', 10)
        
        # 1. CONNECT TO SIMULATION
        # We use SHARED_MEMORY to talk to the sim_node process
        self.client = p.connect(p.SHARED_MEMORY)
        
        if self.client < 0:
            self.get_logger().error("Failed to connect to PyBullet. Ensure sim_node is running!")
        else:
            self.get_logger().info("Connected to PyBullet simulation via Shared Memory.")

        # 2. RUN AT 10Hz
        self.timer = self.create_timer(0.1, self.publish_state)

    def publish_state(self):
        # 3. SAFETY CHECK: Is the simulation ready?
        if self.client < 0 or p.getNumBodies(physicsClientId=self.client) == 0:
            self.get_logger().warn("Simulation not ready or no objects found...", once=True)
            return

        try:
            # 4. GET POSITION AND ORIENTATION
            # Body ID 0 is usually the plane, 1 is usually your car.
            # Change '1' to '0' if your car is the only object loaded.
            car_id = 1 
            pos, orn = p.getBasePositionAndOrientation(car_id, physicsClientId=self.client)
            
            # Convert orientation to Euler to get Yaw
            euler = p.getEulerFromQuaternion(orn)
            yaw = euler[2]

            # 5. PREPARE MSG
            state = np.array([
                pos[0], # X position
                pos[1], # Y position
                yaw     # Rotation (Yaw)
            ], dtype=np.float32)

            msg = Float32MultiArray()
            msg.data = state.tolist()
            self.pub.publish(msg)

        except Exception as e:
            # If the car ID doesn't exist yet, it will throw an error. 
            # We catch it so the node doesn't die.
            self.get_logger().debug(f"Waiting for car: {e}")

def main():
    rclpy.init()
    node = SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect(physicsClientId=node.client)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
