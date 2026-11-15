#!/usr/bin/env python3
"""Task 7 -- localization under landmark degradation.

Runs the demo with room F's markers made unreadable, so the filter dead-reckons across
the room and re-converges on the way out.

"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


BLOCKED_IDS = [7, 11, 16]

ALPHA = [0.30, 0.0, 0.1109, 0.0445]


def generate_launch_description():
    pkg = get_package_share_directory("exploration_assignment")

    args = [
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("gui", default_value="true"),
    ]

    world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, "launch", "house_aruco.launch.py")),
        launch_arguments={"gui": LaunchConfiguration("gui")}.items())

    def script(name, **params):
        return Node(package="exploration_assignment", executable=name,
                    name=os.path.splitext(name)[0], output="screen",
                    parameters=[dict(use_sim_time=True, **params)])

    return LaunchDescription(args + [
        world,
        script("aruco_detector_node.py",
               publish_debug_image=True,
               use_compressed=False,
               debug_compressed=True,
               blocked_ids=BLOCKED_IDS),
        script("noisy_odom_node.py", enable_noise=True, alpha=ALPHA),
        script("gt_pose_node.py"),
        script("ekf_node.py"),
        script("ekf_viz_node.py"),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", os.path.join(pkg, "rviz", "ekf.rviz")],
             parameters=[{"use_sim_time": True}],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
