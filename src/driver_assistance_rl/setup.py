from setuptools import setup
from glob import glob
import os

package_name = 'driver_assistance_rl'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # Registers the package in ROS 2
        ('share/ament_index/resource_index/packages',
         [os.path.join('resource', package_name)]),
        # Include package.xml
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'torch', 'pybullet', 'numpy'],
    zip_safe=True,
    maintainer='Driver Assistance Team',
    maintainer_email='team@example.com',
    description='Driver assistance RL using ROS 2 and PyBullet - Adapted from DriverAssistanceSim',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_node = driver_assistance_rl.sim_node:main',
            'control_node = driver_assistance_rl.control_node:main',
            'sensor_node = driver_assistance_rl.sensor_node:main',
            'rl_agent_node = driver_assistance_rl.rl_agent_node:main',
            'training_node = driver_assistance_rl.training_node:main',
        ],
    },
)
