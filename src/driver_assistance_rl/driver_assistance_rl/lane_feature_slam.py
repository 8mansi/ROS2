class LaneFeatureSLAM:
    def __init__(self):
        self.features = []   # list of lane line parameters
        self.robot_poses = []  # (x, y, yaw)

    def extract_lane_features(self, robot_pose, left_y, right_y):
        """
        Lane lines are parallel to x-axis in your env
        """
        x, y, yaw = robot_pose

        # Left lane: y = left_y
        # Right lane: y = right_y
        left_line = (0.0, 1.0, -left_y)
        right_line = (0.0, 1.0, -right_y)

        self.features.append(left_line)
        self.features.append(right_line)

    def update(self, robot_pose, left_y, right_y):
        self.robot_poses.append(robot_pose)
        self.extract_lane_features(robot_pose, left_y, right_y)

    def get_map(self):
        return self.features
