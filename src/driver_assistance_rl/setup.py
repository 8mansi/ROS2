from setuptools import setup
from glob import glob
import os

package_name = 'driver_assistance_rl'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],  # Your Python module folder
    data_files=[
        # Registers the package in ROS 2
        ('share/ament_index/resource_index/packages',
         [os.path.join('resource', package_name)]),
        # Include package.xml
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Driver assistance RL using ROS 2 and PyBullet',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Make sure these files exist inside driver_assistance_rl module
            'sim_node = driver_assistance_rl.sim_node:main',
            'control_node = driver_assistance_rl.control_node:main',
            'sensor_node = driver_assistance_rl.sensor_node:main',
            'rl_agent_node = driver_assistance_rl.rl_agent_node:main',
        ],
    },
)
