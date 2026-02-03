#!/usr/bin/env python3

from .d_star import DStar
from .lane_feature_slam import LaneFeatureSLAM
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
        self.sim_running = True
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

        # --- MCL parameters ---
        self.num_particles = 200
        self.particles = []  # [x, y, yaw]
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.motion_noise = np.array([0.02, 0.02, 0.01])
        self.sensor_noise = 0.2
        self.lane_y_sigma = 0.08
        self.lane_yaw_sigma = 0.05
        self.last_v = 0.0
        self.last_omega = 0.0


        
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

        self.reset_sub_inference = self.create_subscription(
            Float32MultiArray, '/sim/reset_inference', self.reset_callback, 10
        )
        
        self.step_pub = self.create_publisher(
            Float32MultiArray, '/sim/step', 10
        )
        
        # Publisher for collision/done status
        self.done_pub = self.create_publisher(
            Float32MultiArray, '/sim/done', 10
        )
        
        # Publisher for SLAM map (lane features)
        self.slam_map_pub = self.create_publisher(
            Float32MultiArray, '/slam/lane_map', 10
        )
        
        # Publisher for D* path planning
        self.path_pub = self.create_publisher(
            Float32MultiArray, '/planning/dstar_path', 10
        )
        
        # Simulation step timer
        self.timer = self.create_timer(self.timeStep, self.simulation_step)
        
        # Initialize SLAM system
        self.feature_slam = LaneFeatureSLAM()
        
        # Initialize D* path planner
        self.dstar = DStar(
            width=400,
            height=40
        )
        self.dstar.set_goal((self.num_steps, 0))  # far ahead center lane
        self.dstar.obstacles = set()  # Clean slate
        
        # D* planning parameters
        self.dstar_plan_frequency = 10  # Plan every N steps
        self.dstar_step_counter = 0


        # System readiness
        self.system_ready = False
        self.startup_delay = 5  # seconds to wait for ROS2 discovery
        self.startup_time = time.time()
        self.first_episode = True
        
        # Post-episode reset synchronization
        self.awaiting_post_episode_reset = False
        self.post_episode_reset_time = None
        self.post_episode_reset_delay = 2.5  # seconds to wait after reset request
        
        self.get_logger().info("SimNode initialized")
    
    def init_particles(self, num_particles=200):
        self.num_particles = num_particles
        self.particles = []
    
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw0 = p.getEulerFromQuaternion(orn)[2]
    
        for _ in range(num_particles):
            x = np.random.normal(pos[0], 0.05)
            y = np.random.normal(pos[1], 0.02)
            yaw = np.random.normal(yaw0, 0.02)
    
            # Keep particles within physical lane bounds
            y = float(np.clip(
                y,
                -self.lane_width + (self.offset * 2),
                self.lane_width - (self.offset * 2)
            ))
    
            self.particles.append({
                'x': x,
                'y': y,
                'yaw': yaw,
                'w': 1.0 / num_particles
            })
    
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def reset_simulation(self):
        """Reset the entire simulation"""
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
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
        self.init_particles()
        # self.particles[:, 0] = self.robot_start_pos[0]
        # self.particles[:, 1] = self.robot_start_pos[1]
        # self.particles[:, 2] = 0.0

        self.trajectory = []
        self.danger_points = []
        
        # Step simulation once to stabilize
        p.stepSimulation()
        
        self.get_logger().info("Simulation reset complete")
    
    def disconnect_environmnent(self):
        p.disconnect()

        
    def initialize_road(self):
        """Create lanes and road markings"""
        self.road_id = p.loadURDF("plane.urdf")
        self.left_boundary = self.lane_width - (self.offset * 2)
        self.right_boundary = -self.lane_width + (self.offset * 2)
        
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

            cx = int(car['x_pos'] / 0.5)
            cy = int((car['y_pos'] + self.lane_width) / 0.5)
            self.dstar.add_obstacle((cx, cy))

            p.resetBasePositionAndOrientation(cid, new_pos, orn)
    
    def mcl_motion_update(self, v, omega, dt):
        for p in self.particles:
            # Reduced noise to avoid runaway drift
            v_hat = v + np.random.normal(0, 0.005)
            omega_hat = omega + np.random.normal(0, 0.002)
    
            p['x'] += v_hat * math.cos(p['yaw']) * dt
            p['y'] += v_hat * math.sin(p['yaw']) * dt
            p['yaw'] += omega_hat * dt
    
            # Enforce lane boundary after motion
            p['y'] = float(np.clip(
                p['y'],
                -self.lane_width + (self.offset * 2),
                self.lane_width - (self.offset * 2)
            ))

            
    def cmd_callback(self, msg):
        """Handle velocity commands"""
        linear = msg.linear.x
        angular = msg.angular.z
    
        # Store control for MCL (used in get_state)
        self.last_v = linear
        self.last_omega = angular
    
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

    def shutdown_simulation(self):
        self.get_logger().info("Stopping simulation cleanly")
        self.sim_running = False

        if hasattr(self, 'timer'):
            self.timer.cancel()
        try:
            print("Disconnecting PyBullet...")
            p.disconnect()
        except:
            pass

    def reset_callback(self, msg):
        """Handle reset requests from training node"""
        self.get_logger().info("Reset requested by training node")
        if msg.data[0] == 0.0:
            # Training finished, stop reacting
            self.get_logger().info("Received stop signal. No further resets.")
            self.shutdown_simulation()
            return

        # For subsequent episodes, just reset without startup delay
        self.first_episode = False
        self.reset_simulation()
        self.awaiting_post_episode_reset = True
        self.post_episode_reset_time = time.time()
        self.system_ready = False  # Pause publishing until sync complete

    def mcl_lane_measurement_update(self, y_meas, yaw_meas):
        for i, ptl in enumerate(self.particles):
            y_hat = ptl['y']
            yaw_hat = ptl['yaw']

            err_y = y_meas - y_hat
            err_yaw = yaw_meas - yaw_hat

            weight = np.exp(
                -0.5 * (
                    (err_y**2) / (self.lane_y_sigma**2) +
                    (err_yaw**2) / (self.lane_yaw_sigma**2)
                )
            )
            ptl['w'] = weight
            self.weights[i] = weight

        # Normalize weights
        total_weight = sum(p['w'] for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p['w'] /= total_weight
            self.weights = np.array([p['w'] for p in self.particles])
        else:
            # Fallback to uniform if all weights are zero
            for p in self.particles:
                p['w'] = 1.0 / self.num_particles
            self.weights = np.ones(self.num_particles) / self.num_particles

    # def mcl_measurement_update(self, b1, b2, b3, b4):
    #     z = np.array([b1, b2, b3, b4])

    #     for i, ptl in enumerate(self.particles):
    #         fake_pos = [ptl[0], ptl[1], self.z_offset]
    #         d1, d2, d3, d4 = self.four_parallel_robot_beams(fake_pos, yaw_override=ptl[2])
    #         z_hat = np.array([
    #             self.norm_beam(d1),
    #             self.norm_beam(d2),
    #             self.norm_beam(d3),
    #             self.norm_beam(d4),
    #         ])
    #         err = np.linalg.norm(z - z_hat)
    #         self.weights[i] = np.exp(-0.5 * (err ** 2) / (self.sensor_noise ** 2))

    #     self.weights += 1e-9
    #     self.weights /= np.sum(self.weights)

    def mcl_resample(self):
        weights = [p['w'] for p in self.particles]
        # Avoid numerical issues: ensure weights sum to 1
        wsum = sum(weights)
        if wsum <= 0:
            weights = [1.0 / self.num_particles] * self.num_particles
        else:
            weights = [w / wsum for w in weights]
    
        indices = np.random.choice(
            range(self.num_particles),
            self.num_particles,
            p=weights
        )
    
        new_particles = []
        for i in indices:
            p = self.particles[i]
            # Add small jitter to maintain diversity
            x = p['x'] + np.random.normal(0, 0.01)
            y = p['y'] + np.random.normal(0, 0.005)
            yaw = p['yaw'] + np.random.normal(0, 0.005)
    
            # Clip to lane
            y = float(np.clip(
                y,
                -self.lane_width + (self.offset * 2),
                self.lane_width - (self.offset * 2)
            ))
    
            new_particles.append({
                'x': x,
                'y': y,
                'yaw': yaw,
                'w': 1.0 / self.num_particles
            })
    
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

    def mcl_sensor_update(self, observed_ranges):
        sigma = 0.3
    
        for p in self.particles:
            expected = self.simulate_rays(p['x'], p['y'], p['yaw'])
    
            error = np.linalg.norm(
                np.array(expected) - np.array(observed_ranges)
            )
    
            # Gaussian likelihood
            p['w'] = math.exp(-(error ** 2) / (2 * sigma ** 2))
    
        self.normalize_weights()
    def normalize_weights(self):
        total = sum(p['w'] for p in self.particles)
        if total == 0:
            for p in self.particles:
                p['w'] = 1.0 / self.num_particles
            return
    
        for p in self.particles:
            p['w'] /= total
    
    def effective_sample_size(self):
        return 1.0 / sum(p['w'] ** 2 for p in self.particles)
 

    def mcl_estimated_pose(self):
        x = sum(p['x'] * p['w'] for p in self.particles)
        y = sum(p['y'] * p['w'] for p in self.particles)
    
        sin_yaw = sum(math.sin(p['yaw']) * p['w'] for p in self.particles)
        cos_yaw = sum(math.cos(p['yaw']) * p['w'] for p in self.particles)
    
        yaw = math.atan2(sin_yaw, cos_yaw)
        return x, y, yaw
    
    # def mcl_resample(self):
    #     idx = np.random.choice(
    #         self.num_particles,
    #         size=self.num_particles,
    #         p=self.weights
    #     )
    #     self.particles = self.particles[idx]
    #     self.weights.fill(1.0 / self.num_particles)

    def neff(self):
        return 1.0 / sum(p['w'] ** 2 for p in self.particles)
    
    # def visualize_particles(self):
    #     """Draw particles on the road for debugging MCL convergence"""
    #     # Clear previous debug lines
    #     if not hasattr(self, 'particle_debug_ids'):
    #         self.particle_debug_ids = []
    #     else:
    #         for line_id in self.particle_debug_ids:
    #             try:
    #                 p.removeUserDebugItem(line_id)
    #             except:
    #                 pass
    #         self.particle_debug_ids = []
        
    #     # Draw each particle as a small sphere
    #     for particle in self.particles:
    #         # Color based on weight (brighter = higher weight)
    #         weight = particle['w']
    #         r = min(1.0, weight * 10)  # Scale weight for visibility
    #         g = 0.5
    #         b = min(1.0, weight * 5)
            
    #         # Draw particle position as a colored sphere
    #         line_id = p.addUserDebugLine(
    #             lineFromXYZ=[particle['x'], particle['y'], 0.02],
    #             lineToXYZ=[particle['x'], particle['y'], 0.02],
    #             lineColorRGB=[r, g, b],
    #             lineWidth=10,
    #             lifeTime=0  # Persistent until removed
    #         )
    #         self.particle_debug_ids.append(line_id)
        
    #     # # Draw estimated pose (weighted average) in red
    #     # x_hat, y_hat, yaw_hat = self.mcl_estimated_pose()
    #     # est_line = p.addUserDebugLine(
    #     #     lineFromXYZ=[x_hat, y_hat, 0.05],
    #     #     lineToXYZ=[x_hat + 0.3 * math.cos(yaw_hat), y_hat + 0.3 * math.sin(yaw_hat), 0.05],
    #     #     lineColorRGB=[1, 0, 0],  # Red for estimate
    #     #     lineWidth=4,
    #     #     lifeTime=0
    #     # )
    #     # self.particle_debug_ids.append(est_line)
        
    #     # # Draw ground truth robot pose in green
    #     # robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
    #     # gt_yaw = p.getEulerFromQuaternion(robot_orn)[2]
    #     # gt_line = p.addUserDebugLine(
    #     #     lineFromXYZ=[robot_pos[0], robot_pos[1], 0.05],
    #     #     lineToXYZ=[robot_pos[0] + 0.3 * math.cos(gt_yaw), robot_pos[1] + 0.3 * math.sin(gt_yaw), 0.05],
    #     #     lineColorRGB=[0, 1, 0],  # Green for ground truth
    #     #     lineWidth=4,
    #     #     lifeTime=0
    #     # )
    #     # self.particle_debug_ids.append(gt_line)


    def get_state(self):
        """
        Comprehensive State Construction:
        Indices: 0-3 (Beams), 4 (Dist), 5-6 (Lanes), 7 (SLAM Offset), 8 (Heading), 9 (D* Error)
        """
        try:
            # --- A. GROUND TRUTH (SIM ONLY, NOT BELIEF) ---
            robot_pos, gt_orn = p.getBasePositionAndOrientation(self.robot_id)
            gt_yaw = p.getEulerFromQuaternion(gt_orn)[2]

            # --- B. MCL: PREDICTION STEP ---
            # Use last applied control
            self.mcl_motion_update(
                self.last_v,
                self.last_omega,
                self.timeStep
            )

            # --- C. MCL: MEASUREMENT UPDATE ---
            # Lane-based measurement
            self.mcl_lane_measurement_update(robot_pos[1], gt_yaw)

            # --- D. MCL: RESAMPLING ---
            if self.neff() < self.num_particles / 2:
                self.mcl_resample()

            # --- E. MCL: POSE ESTIMATE ---
            if self.dstar_step_counter < 0:
                # Bootstrap using ground truth (simulation convenience)
                for pt in self.particles:
                    pt['x'] = robot_pos[0]
                    pt['y'] = robot_pos[1]
                    pt['yaw'] = gt_yaw
                    pt['w'] = 1.0 / self.num_particles
                # Keep the separate weights array in sync
                # self.weights = np.ones(self.num_particles) / self.num_particles
                x_hat, y_hat, yaw_hat = robot_pos[0], robot_pos[1], gt_yaw
                # print("Robot Pos used for : ", robot_pos)
            else:
                x_hat, y_hat, yaw_hat = self.mcl_estimated_pose()
                # print("mcl_estimated_pose y estimated", y_hat, "y ground truth", robot_pos[1])

            # --- B. FEATURE MAPPING (SLAM) ---
            # Get local estimates from the SLAM history
            est_left, est_right = self.feature_slam.get_estimated_lanes()
            left_flag, right_flag = self.check_lane_crossing(self.robot_id)
            # --- C. DYNAMIC LANE & SENSOR LOGIC ---
            # Fix: Default lane_center to 0.0 if SLAM is empty to prevent offset = 1.0 at start
            if est_left is not None and est_right is not None:
                lane_center = (est_left + est_right) / 2
            else:
                lane_center = 0.0 

            half_width = self.lane_width / 2.0
            lane_offset = (y_hat - lane_center) / half_width
            lane_offset = float(np.clip(lane_offset, -1.0, 1.0))

            if left_flag > 0 or right_flag > 0:
                self.feature_slam.update((x_hat, y_hat, yaw_hat), left_flag, right_flag)
            
            # # Logic for Active Sensing: Trigger camera if we are near edge OR map is empty
            # active_sensing_needed = True
            # if len(self.feature_slam.observations) > 5 and abs(lane_offset) < 0.7:
            #     active_sensing_needed = False

            # # Trigger camera check only when needed
            # left_flag, right_flag = 0.0, 0.0
            # if active_sensing_needed:
            #     l_check, r_check = self.check_lane_crossing(self.robot_id)
            #     left_flag, right_flag = float(l_check), float(r_check)
            #     # Feed camera findings back into SLAM to "fix" the map
            #     self.feature_slam.update((x_hat, y_hat, yaw_hat), l_check, r_check)

            # --- D. PATH PLANNING (D*) ---
            dstar_error = 0.0
            robot_cell = (int((x_hat + 10) / 0.5), int((y_hat + 2.0) / 0.5))
            
            came_from = self.dstar.plan(robot_cell)
            if came_from and self.dstar.goal in came_from:
                dx = self.dstar.goal[0] - robot_cell[0]
                dy = self.dstar.goal[1] - robot_cell[1]
                
                if abs(dx) > 1 or abs(dy) > 1:
                    target_yaw = math.atan2(dy, dx)
                    # Shortest path angular difference
                    diff = target_yaw - yaw_hat
                    while diff > math.pi: diff -= 2*math.pi
                    while diff < -math.pi: diff += 2*math.pi
                    dstar_error = np.clip(diff / math.pi, -1.0, 1.0)

            # --- E. COMPILE STATE VECTOR ---
            b1, b2, b3, b4 = self.four_parallel_robot_beams(robot_pos)
            
            hit_x, hit_y, _ = self.beam_sensor(robot_pos)
            beam_dist = np.linalg.norm(np.array([hit_x, hit_y]) - np.array(robot_pos[:2])) if hit_x != 0 else self.max_range
            
            state_list = [
                self.norm_beam(b1), self.norm_beam(b2), self.norm_beam(b3), self.norm_beam(b4),
                np.clip((self.max_range - beam_dist) / self.max_range, 0, 1),
                left_flag,    # Index 5
                right_flag,   # Index 6
                lane_offset,  # Index 7 (Fixed Normalization)
                np.clip(yaw_hat / np.pi, -1.0, 1.0),
                dstar_error   
            ]

            return np.array(state_list, dtype=np.float32)

        except Exception as e:
            self.get_logger().error(f"Critical State Error: {e}")
            return np.zeros(10, dtype=np.float32)
        
    # def get_state(self):
    #     """Get comprehensive state from simulation"""
    #     robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        
    #     # Get beam sensor readings
    #     b1, b2, b3, b4 = self.four_parallel_robot_beams(robot_pos)
    #     b1 = self.norm_beam(b1)
    #     b2 = self.norm_beam(b2)
    #     b3 = self.norm_beam(b3)
    #     b4 = self.norm_beam(b4)

    #     robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
    #     yaw = p.getEulerFromQuaternion(robot_orn)[2]

    #     y_meas = robot_pos[1]
    #     yaw_meas = yaw
    #     self.mcl_lane_measurement_update(y_meas, yaw_meas)
    #     if self.neff() < self.num_particles / 2:
    #         self.mcl_resample()
        
    #     # Beam-based obstacle detection
    #     hit_x, hit_y, angle = self.beam_sensor(robot_pos)
    #     if hit_x == 0 and hit_y == 0:
    #         beam_dist = self.max_range
    #     else:
    #         beam_dist = np.linalg.norm(
    #             np.array([hit_x, hit_y]) - np.array(robot_pos[:2])
    #         )
    #     beam_dist = np.clip(beam_dist, 0, self.max_range)
    #     beam_dist = (self.max_range - beam_dist) / self.max_range
        
    #     # Lane crossing check
    #     SIDE_THRESHOLD = 0.4
    #     dist_left = abs(robot_pos[1] - self.left_boundary)
    #     dist_right = abs(robot_pos[1] - self.right_boundary)
    #     if dist_left < SIDE_THRESHOLD or dist_right < SIDE_THRESHOLD:
    #         left_lane, right_lane = self.check_lane_crossing(self.robot_id)
    #     else:
    #         left_lane, right_lane = 0, 0
            
    #     # MCL-estimated pose (from SLAM-like particle filter)
    #     x_hat, y_hat, yaw_hat = self.mcl_estimated_pose()

    #     # Lane feature extraction via SLAM
    #     left_lane_y = self.left_boundary
    #     right_lane_y = self.right_boundary
    #     self.feature_slam.update(
    #         (x_hat, y_hat, yaw_hat),
    #         left_lane_y,
    #         right_lane_y
    #     )
    #     # Get SLAM estimated lanes (if available)
    #     est_left, est_right = self.feature_slam.get_estimated_lanes()
    #     if est_left is not None:
    #         # Use SLAM estimates
    #         lane_center_estimate = (est_left + est_right) / 2
    #     else:
    #         # Fallback to ground truth initially
    #         lane_center_estimate = (self.left_boundary + self.right_boundary) / 2

    #     # Lane state features (relative to SLAM estimate, not ground truth)
    #     lane_offset = np.clip(
    #         y_hat / lane_center_estimate, -1.0, 1.0
    #     )
    #     # # Lane state features
    #     # lane_offset = np.clip(
    #     #     y_hat / (self.lane_width / 2), -1.0, 1.0
    #     # )
    #     heading_error = np.clip(
    #         yaw_hat / np.pi, -1.0, 1.0
    #     )

    #     # D* path planning for navigation guidance
    #     robot_cell = (
    #         int((x_hat + 10) / 0.5),  # Offset for positive indexing
    #         int((y_hat + self.lane_width) / 0.5)
    #     )
        
    #     # Clamp to grid bounds
    #     robot_cell = (
    #         np.clip(robot_cell[0], 0, self.dstar.width - 1),
    #         np.clip(robot_cell[1], 0, self.dstar.height - 1)
    #     )

    #     # Plan path periodically to reduce computation
    #     dstar_heading_error = 0.0
    #     if robot_cell != getattr(self, 'last_robot_cell', None):
    #         self.last_robot_cell = robot_cell
    #         try:
    #             path = self.dstar.plan(robot_cell)
    #             if path and robot_cell in path:
    #                 next_cell = path[robot_cell]
    #                 dx = next_cell[0] - robot_cell[0]
    #                 dy = next_cell[1] - robot_cell[1]
    #                 dstar_heading = math.atan2(dy, dx)
    #                 dstar_heading_error = np.clip(dstar_heading / np.pi, -1.0, 1.0)
    #         except Exception as e:
    #             self.get_logger().debug(f"D* planning error: {e}")

    #     state = np.array([
    #         b1, b2, b3, b4,
    #         beam_dist,
    #         left_lane, right_lane,
    #         lane_offset,
    #         heading_error,
    #         dstar_heading_error  
    #     ], dtype=np.float32)
        
    #     return state
    
    def four_parallel_robot_beams(self, robot_pos, yaw_override=None, max_range=None, rays_per_beam=5, beam_width=0.2):
        """Emit parallel rays for obstacle detection"""
        if max_range is None:
            max_range = self.max_range / 1.5
            
        BEAM_Z = self.z_offset
        ANG = math.radians(15)
        LATERAL_GAP = beam_width / 2
        FRONT_OFFSET = 0.3
        
        if yaw_override is None:
            _, orn = p.getBasePositionAndOrientation(self.robot_id)
            yaw = p.getEulerFromQuaternion(orn)[2]
        else:
            yaw = yaw_override
        
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
    
    def check_lane_crossing(self, robot_id, threshold=20):
        """
        Detects if the robot is physically touching or crossing the lane lines.
        Returns: (left_flag, right_flag) as 1.0 or 0.0
        """
        gray, _, _ = self.get_lane_camera_image(robot_id)
        mean_val = np.mean(gray)
        
        # Ground Truth Y-position
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        robot_y = pos[1]
        
        left_flag = 0.0
        right_flag = 0.0
        
        # The 'Physical' Lane Boundary (usually half the lane width)
        boundary = self.lane_width / 2.0 

        # Case A: Camera sees the white line (Direct Sensing)
        if mean_val > threshold:
            if robot_y > 0: left_flag = 1.0
            else: right_flag = 1.0
        
        # Case B: Safety Fallback (If robot is physically past the line)
        # This prevents 'reward bleeding' if the camera is blind
        if robot_y > boundary:
            left_flag = 1.0
        elif robot_y < -boundary:
            right_flag = 1.0
                
        return left_flag, right_flag
    
    # def check_lane_crossing(self, robot_id, threshold=20, forward_offset=0.3, z_offset=0.05):
    #     """Check if robot has crossed lane markings"""
    #     rgb, _, _ = self.get_lane_camera_image(robot_id)
    #     print("Mean pixel value:", np.mean(rgb))
    #     lane_visible = (np.mean(rgb) > threshold)
        
    #     pos, _ = p.getBasePositionAndOrientation(robot_id)
    #     robot_y = pos[1]
        
    #     left_lane = False
    #     right_lane = False
        
    #     if lane_visible:
    #         if robot_y > 0:
    #             left_lane = True
    #         elif robot_y < 0:
    #             right_lane = True
                
    #     return left_lane, right_lane
        
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
        if not self.sim_running:
            return

        # Handle post-episode reset synchronization
        if self.awaiting_post_episode_reset:
            elapsed = time.time() - self.post_episode_reset_time
            if elapsed < self.post_episode_reset_delay:
                return  # Wait for sync delay
            else:
                self.awaiting_post_episode_reset = False
                self.system_ready = True
                self.get_logger().info("Sim node re-synchronized after episode reset!")
                return  # Skip simulation step this iteration
        
        # Check if startup delay has passed
        if not self.system_ready:
            elapsed = time.time() - self.startup_time
            if elapsed < self.startup_delay:
                # Waiting for system startup
                return
            else:
                self.system_ready = True
                self.get_logger().info("System ready! Starting simulation")
        
        # try:
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
        
        # Publish SLAM lane features periodically
        if self.dstar_step_counter % 5 == 0:
            slam_map = self.feature_slam.get_map()
            if slam_map:
                slam_msg = Float32MultiArray()
                # Publish observations with position context: [x1, left_y1, right_y1, robot_y1, x2, ...]
                slam_msg.data = [float(obs['x_pos']) for obs in slam_map for obs in [obs]] + \
                                [float(obs['left_y']) for obs in slam_map] + \
                                [float(obs['right_y']) for obs in slam_map] + \
                                [float(obs['robot_y']) for obs in slam_map]
                self.slam_map_pub.publish(slam_msg)
        
        # Publish D* path planning info periodically
        if self.dstar_step_counter % 10 == 0:
            robot_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            x_hat, y_hat, yaw_hat = self.mcl_estimated_pose()
            robot_cell = (
                np.clip(int((x_hat + 10) / 0.5), 0, self.dstar.width - 1),
                np.clip(int((y_hat + self.lane_width) / 0.5), 0, self.dstar.height - 1)
            )
            try:
                path = self.dstar.plan(robot_cell)
                # Publish goal position
                goal = self.dstar.goal
                path_msg = Float32MultiArray()
                path_msg.data = [
                    float(robot_cell[0]),
                    float(robot_cell[1]),
                    float(goal[0]),
                    float(goal[1]),
                    float(len(self.dstar.obstacles))  # obstacle count
                ]
                self.path_pub.publish(path_msg)
            except Exception as e:
                self.get_logger().debug(f"Path planning publish error: {e}")
        
        self.dstar_step_counter += 1
        
        # Update camera to follow robot
        try:
            # Visualize particles for debugging
            # self.visualize_particles()
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=5,
                cameraYaw=0,
                cameraPitch=-80,
                cameraTargetPosition=pos
            )
        except Exception as e:
            self.get_logger().debug(f"Camera update error: {e}")
                
        # except Exception as e:
        #     self.get_logger().error(f"Simulation step error: {e}")


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
