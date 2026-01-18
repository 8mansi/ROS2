#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pybullet as p
import pybullet_data
import numpy as np
import math
import time
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist


class SimNode(Node):
    """
    Full PyBullet simulation node with traffic, collision detection,
    and comprehensive state sensing for the driver assistance system.
    """
    
    def __init__(self):
        super().__init__('sim_node')
        
        # Simulation parameters
        self.timeStep = 0.1
        self.num_steps = 500
        self.max_range = 3
        self.z_offset = 0.05
        self.num_cars = 10
        self.lane_width = 2.0
        self.offset = 0.1
        self.lane_width_total = 2 * self.lane_width + self.offset
        self.ang_vel = 6
        self.wheel_joints = [2, 3, 4, 5]
        self.MAX_FORCE = 200
        self.WHEEL_DISTANCE = 0.55
        
        # PyBullet setup
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Initialize simulation
        self.reset_simulation()
        
        # ROS Subscriptions and Publishers
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10
        )
        
        # Subscriber for reset requests
        self.reset_sub = self.create_subscription(
            Float32MultiArray, '/sim/reset', self.reset_callback, 10
        )
        
        self.step_pub = self.create_publisher(
            Float32MultiArray, '/sim/step', 10
        )
        
        # Publisher for collision/done status
        self.done_pub = self.create_publisher(
            Float32MultiArray, '/sim/done', 10
        )
        
        # Simulation step timer
        self.timer = self.create_timer(self.timeStep, self.simulation_step)
        
        self.get_logger().info("SimNode initialized")
        
    def reset_simulation(self):
        """Reset the entire simulation"""
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load ground
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Initialize road
        self.initialize_road()
        
        # Initialize robot
        self.initialize_robot()
        
        # Spawn traffic cars
        self.cars = []
        self.spawn_cars(self.num_cars)
        
        self.trajectory = []
        self.danger_points = []
        
        # Step simulation once to stabilize
        p.stepSimulation()
        
        self.get_logger().info("Simulation reset complete")
        
    def initialize_road(self):
        """Create lanes and road markings"""
        self.road_id = p.loadURDF("plane.urdf")
        self.left_boundary = self.lane_width - (self.offset * 6)
        self.right_boundary = -self.lane_width + (self.offset * 6)
        
        p.changeVisualShape(self.road_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1])
        
        # Draw lane lines
        p.addUserDebugLine(
            [0, self.lane_width - self.offset, 0.01],
            [200, self.lane_width - self.offset, 0.01],
            lineColorRGB=[1, 1, 1], lineWidth=4
        )
        p.addUserDebugLine(
            [0, -self.lane_width + self.offset, 0.01],
            [200, -self.lane_width + self.offset, 0.01],
            lineColorRGB=[1, 1, 1], lineWidth=4
        )
        
        lane_length = 200
        lane_color = [0, 0, 0, 1]
        
        lane_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[lane_length / 2, self.lane_width_total / 2, 0.01],
            rgbaColor=lane_color
        )
        lane_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[lane_length / 2, self.lane_width_total / 2, 0.01]
        )
        lane_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=lane_collision,
            baseVisualShapeIndex=lane_visual,
            basePosition=[lane_length/2, 0, 0]
        )
        
    def initialize_robot(self):
        """Initialize the robot (Husky)"""
        y = np.random.uniform(
            -self.lane_width + (self.offset * 5),
            self.lane_width - (self.offset * 5)
        )
        self.robot_start_pos = [0, y, 0.05]
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        robot_path = pybullet_data.getDataPath() + "/husky/husky.urdf"
        self.robot_id = p.loadURDF(
            robot_path, self.robot_start_pos,
            self.robot_start_orientation, globalScaling=1.0
        )
        
    def spawn_cars(self, car_count):
        """Spawn traffic cars"""
        for _ in range(car_count):
            self.initialize_car(new_car=True)
            
    def initialize_car(self, new_car=False):
        """Initialize a single traffic car"""
        min_dist_between_cars = 1.0
        robot_safe_dist = 5.0
        
        placed = False
        while not placed:
            x = np.random.uniform(0, 10)
            y = np.random.uniform(
                -self.lane_width + (self.offset * 2),
                self.lane_width - (self.offset * 2)
            )
            
            # Check distance to robot
            dist_to_robot = np.sqrt(
                (x - self.robot_start_pos[0])**2 +
                (y - self.robot_start_pos[1])**2
            )
            if dist_to_robot < robot_safe_dist:
                continue
            
            # Check distance to other cars
            overlap = False
            for car in self.cars:
                dist = np.sqrt((x - car['x_pos'])**2 + (y - car['y_pos'])**2)
                if dist < min_dist_between_cars:
                    overlap = True
                    break
            
            if not overlap:
                car_id = p.loadURDF(
                    "racecar/racecar.urdf", [x, y, 0],
                    p.getQuaternionFromEuler([0, 0, 0]),
                    globalScaling=1
                )
                velocity = np.random.uniform(-0.02, -0.08)
                self.cars.append({
                    'id': car_id,
                    'vel': velocity,
                    'x_pos': x,
                    'y_pos': y
                })
                placed = True
                
    def move_cars(self):
        """Update car positions to simulate traffic"""
        for car in self.cars:
            cid = car['id']
            pos, orn = p.getBasePositionAndOrientation(cid)
            new_pos = [pos[0] - car['vel'], pos[1], pos[2]]
            p.resetBasePositionAndOrientation(cid, new_pos, orn)
            
    def cmd_callback(self, msg):
        """Handle velocity commands"""
        linear = msg.linear.x
        angular = msg.angular.z
        
        left = linear - angular * self.WHEEL_DISTANCE / 2
        right = linear + angular * self.WHEEL_DISTANCE / 2
        
        for i, j in enumerate(self.wheel_joints):
            vel = left if i % 2 == 0 else right
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=self.MAX_FORCE
            )
            
    def reset_callback(self, msg):
        """Handle reset requests from training node"""
        self.get_logger().info("Reset requested by training node")
        self.reset_simulation()
            
    def get_state(self):
        """Get comprehensive state from simulation"""
        try:
            robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        # Get beam sensor readings
        b1, b2, b3, b4 = self.four_parallel_robot_beams(robot_pos)
        b1 = self.norm_beam(b1)
        b2 = self.norm_beam(b2)
        b3 = self.norm_beam(b3)
        b4 = self.norm_beam(b4)
        
        # Beam-based obstacle detection
        hit_x, hit_y, angle = self.beam_sensor(robot_pos)
        if hit_x == 0 and hit_y == 0:
            beam_dist = self.max_range
        else:
            beam_dist = np.linalg.norm(
                np.array([hit_x, hit_y]) - np.array(robot_pos[:2])
            )
        beam_dist = np.clip(beam_dist, 0, self.max_range)
        beam_dist = (self.max_range - beam_dist) / self.max_range
        
        # Lane crossing check
        SIDE_THRESHOLD = 0.4
        dist_left = abs(robot_pos[1] - self.left_boundary)
        dist_right = abs(robot_pos[1] - self.right_boundary)
        
        if dist_left < SIDE_THRESHOLD or dist_right < SIDE_THRESHOLD:
            left_lane, right_lane = self.check_lane_crossing(self.robot_id)
        else:
            left_lane, right_lane = 0, 0
            
        # Lane offset
        lane_offset = np.clip(
            robot_pos[1] / (self.lane_width / 2), -1.0, 1.0
        )
        
        # Heading error (yaw)
        yaw = p.getEulerFromQuaternion(robot_orn)[2]
        heading_error = np.clip(yaw / np.pi, -1.0, 1.0)
        
            return np.array([
                b1, b2, b3, b4,
                beam_dist,
                left_lane, right_lane,
                lane_offset,
                heading_error
            ], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Error getting state: {e}")
            # Return safe default state
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0], dtype=np.float32)
        
    def four_parallel_robot_beams(self, robot_pos, max_range=None, rays_per_beam=5, beam_width=0.2):
        """Emit parallel rays for obstacle detection"""
        if max_range is None:
            max_range = self.max_range / 1.5
            
        BEAM_Z = self.z_offset
        ANG = math.radians(15)
        LATERAL_GAP = beam_width / 2
        FRONT_OFFSET = 0.3
        
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        
        fx = math.cos(yaw)
        fy = math.sin(yaw)
        fx_L = math.cos(yaw + ANG)
        fy_L = math.sin(yaw + ANG)
        fx_R = math.cos(yaw - ANG)
        fy_R = math.sin(yaw - ANG)
        
        def generate_rays(base_pos, forward_vec):
            rays_start = []
            rays_end = []
            for i in range(rays_per_beam):
                offset = -LATERAL_GAP + i * (beam_width / max(rays_per_beam-1, 1))
                lat_vec = [-forward_vec[1], forward_vec[0]]
                start = [
                    base_pos[0] + offset * lat_vec[0],
                    base_pos[1] + offset * lat_vec[1],
                    base_pos[2]
                ]
                end = [
                    start[0] + forward_vec[0] * max_range,
                    start[1] + forward_vec[1] * max_range,
                    start[2]
                ]
                rays_start.append(start)
                rays_end.append(end)
            return rays_start, rays_end
            
        px = -math.sin(yaw)
        py = math.cos(yaw)
        front_pos = [
            robot_pos[0] + FRONT_OFFSET * fx,
            robot_pos[1] + FRONT_OFFSET * fy,
            self.z_offset
        ]
        
        left_base = [
            robot_pos[0] + px * 0.17,
            robot_pos[1] + py * 0.17,
            BEAM_Z
        ]
        right_base = [
            robot_pos[0] - px * 0.17,
            robot_pos[1] - py * 0.17,
            BEAM_Z
        ]
        center_base = front_pos
        
        beams = [
            (left_base, [fx, fy]),
            (right_base, [fx, fy]),
            (center_base, [fx_L, fy_L]),
            (center_base, [fx_R, fy_R])
        ]
        
        all_starts = []
        all_ends = []
        beam_ranges = []
        for b_start, b_vec in beams:
            starts, ends = generate_rays(b_start, b_vec)
            all_starts.extend(starts)
            all_ends.extend(ends)
            beam_ranges.append(len(starts))
            
        results = p.rayTestBatch(all_starts, all_ends)
        car_ids = [c['id'] for c in self.cars]
        
        distances = []
        idx = 0
        for n_rays in beam_ranges:
            dists = []
            for _ in range(n_rays):
                res = results[idx]
                frac = res[2]
                hit = res[0]
                dist = frac * max_range if hit in car_ids else max_range
                dists.append(dist)
                idx += 1
            distances.append(sum(dists) / len(dists))
            
        return distances[0], distances[1], distances[2], distances[3]
        
    def beam_sensor(self, robot_pos, z_offset=0.1, max_range=5):
        """Cast rays around robot to detect obstacles"""
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        fx = math.cos(yaw)
        fy = math.sin(yaw)
        forward_offset = 0.35
        
        r_pos = [
            robot_pos[0] + fx * forward_offset,
            robot_pos[1] + fy * forward_offset,
            robot_pos[2] + z_offset
        ]
        
        angles = np.linspace(-math.pi/5, math.pi/5, 30, endpoint=False)
        from_points = []
        to_points = []
        for a in angles:
            start = [r_pos[0], r_pos[1], r_pos[2]]
            end = [
                r_pos[0] + max_range * math.cos(a),
                r_pos[1] + max_range * math.sin(a),
                r_pos[2]
            ]
            from_points.append(start)
            to_points.append(end)
            
        results = p.rayTestBatch(from_points, to_points)
        car_ids = [car['id'] for car in self.cars]
        hits_cube = []
        
        for i, res in enumerate(results):
            hit_object_uid = res[0]
            hit_fraction = res[2]
            hit_position = res[3]
            
            if hit_object_uid in car_ids:
                ang_w = angles[i]
                measured = hit_fraction * max_range
                dist = self.calculate_euclidean_dist(hit_position, r_pos, z_offset)
                hits_cube.append((dist, hit_position, ang_w))
                
        return self.get_closest_hit(hits_cube)
        
    def get_closest_hit(self, hits_cube):
        """Return the closest hit from ray casting"""
        if len(hits_cube) > 0:
            hits_cube.sort(key=lambda x: x[0])
            closest_hit = hits_cube[0][1]
            angle = hits_cube[0][2]
            return closest_hit[0], closest_hit[1], angle
        else:
            return 0, 0, 0
            
    def calculate_euclidean_dist(self, hit_position, r_pos, z_offset):
        """Calculate Euclidean distance"""
        return math.sqrt(
            (hit_position[0] - r_pos[0])**2 +
            (hit_position[1] - r_pos[1])**2 +
            (hit_position[2] - (r_pos[2] + z_offset))**2
        )
        
    def norm_beam(self, d):
        """Normalize beam distance: 0 = safe, 1 = close"""
        d = np.clip(d, 0, self.max_range)
        return (self.max_range - d) / self.max_range
        
    def check_lane_crossing(self, robot_id, threshold=20, forward_offset=0.3, z_offset=0.05):
        """Check if robot has crossed lane markings"""
        rgb, _, _ = self.get_lane_camera_image(robot_id)
        print("Mean pixel value:", np.mean(rgb))  # Debugging line
        lane_visible = (np.mean(rgb) > threshold)
        
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        robot_y = pos[1]
        
        left_lane = False
        right_lane = False
        
        if lane_visible:
            if robot_y > 0:
                left_lane = True
            elif robot_y < 0:
                right_lane = True
                
        return left_lane, right_lane
        
    def get_lane_camera_image(self, robot_id, z_offset=0.2, forward_offset=0.3):
        """Get camera image from robot perspective"""
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        local_camera_pos = [0.7, 0.0, 0.5]
        pitch_angle = -math.pi / 2
        local_camera_orn = p.getQuaternionFromEuler([0, pitch_angle, 0])
        
        cam_pos, cam_orn = p.multiplyTransforms(
            base_pos, base_orn,
            local_camera_pos, local_camera_orn
        )
        
        cam_yaw = p.getEulerFromQuaternion(cam_orn)[2]
        forward_x = math.cos(cam_yaw)
        forward_y = math.sin(cam_yaw)
        
        target = [cam_pos[0], cam_pos[1], base_pos[2]]
        up_vec = [forward_x, forward_y, 0]
        
        view = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target,
            cameraUpVector=up_vec
        )
        proj = p.computeProjectionMatrixFOV(
            fov=30, aspect=1.0, nearVal=0.01, farVal=1.0
        )
        
        width, height, rgb, depth, seg = p.getCameraImage(
            width=64, height=64,
            viewMatrix=view,
            projectionMatrix=proj
        )
        
        rgb_array = np.array(rgb, dtype=np.uint8).reshape((height, width, 4))
        gray = np.dot(rgb_array[:, :, :3], [0.2989, 0.5870, 0.1140])
        gray = gray.astype(np.uint8)
        
        return gray, depth, seg
        
    def check_collision(self):
        """Check if robot has collided with traffic"""
        try:
            contacts = p.getContactPoints(bodyA=self.robot_id)
            car_ids = {car['id'] for car in self.cars}
            return any((c[1] in car_ids) or (c[2] in car_ids) for c in contacts)
        except Exception as e:
            self.get_logger().error(f"Error checking collision: {e}")
            return False
        
    def check_lane_exit(self):
        """Check if robot has exited the road"""
        try:
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            return abs(pos[1]) > (self.lane_width + self.offset)
        except Exception as e:
            self.get_logger().error(f"Error checking lane exit: {e}")
            return False
        
    def simulation_step(self):
        """Main simulation step"""
        try:
            p.stepSimulation()
            self.move_cars()
            
            # Check for collision and lane exit
            collision = self.check_collision()
            lane_exit = self.check_lane_exit()
            done = collision or lane_exit
            
            # Publish done status
            done_msg = Float32MultiArray()
            done_msg.data = [float(collision), float(lane_exit), float(done)]
            self.done_pub.publish(done_msg)
            
            # Publish current state
            state = self.get_state()
            msg = Float32MultiArray()
            msg.data = state.tolist()
            self.step_pub.publish(msg)
            
            # Update camera to follow robot
            try:
                pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=5,
                    cameraYaw=0,
                    cameraPitch=-80,
                    cameraTargetPosition=pos
                )
            except Exception as e:
                self.get_logger().debug(f"Camera update error: {e}")
                
        except Exception as e:
            self.get_logger().error(f"Simulation step error: {e}")


def main():
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
