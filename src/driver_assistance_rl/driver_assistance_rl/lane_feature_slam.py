class LaneFeatureSLAM:
    def __init__(self):
        self.observations = []  # list of observations: {x_pos, left_y, right_y, robot_y, robot_yaw}
        self.last_stored_x = None
        self.position_threshold = 0.5  # Only store when robot moved >0.5 units

    def update(self, robot_pose, left_y, right_y):

        x, y, yaw = robot_pose

        # Only add observation if robot moved far enough from last stored position
        if self.last_stored_x is None or abs(x - self.last_stored_x) > self.position_threshold:
            self.observations.append({
                'x_pos': x,
                'left_y': left_y,
                'right_y': right_y,
                'robot_y': y,
                'robot_yaw': yaw
            })
            self.last_stored_x = x

    def get_map(self):
        return self.observations

    def get_estimated_lanes(self):
        if not self.observations:
            return None, None
        est_left = sum(obs['left_y'] for obs in self.observations) / len(self.observations)
        est_right = sum(obs['right_y'] for obs in self.observations) / len(self.observations)
        return est_left, est_right
