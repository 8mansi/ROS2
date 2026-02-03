export AMENT_PREFIX_PATH=$AMENT_PREFIX_PATH:$(pwd)/install/driver_assistance_rl
# 2. Make sure ROS env is present
source /opt/ros/jazzy/setup.bash

# 3. Clean old build (built with system python)
rm -rf build install log

# 4. Rebuild using venv python
colcon build --symlink-install

# 5. Source the workspace
source install/setup.bash

ros2 launch driver_assistance_rl inference.launch.py