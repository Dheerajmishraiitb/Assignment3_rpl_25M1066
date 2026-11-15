#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("exploration_assignment")

    args = [
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument(
            "map_csv",
            default_value="/home/ros/ws/Assignment_3/results/task1/manual_map.csv",
        ),
        DeclareLaunchArgument("random_seed", default_value="7"),
    ]

    # Assignment 3 only needs the simulator + robot bring-up.
    # Assignment 1 EKF / ArUco infrastructure is intentionally not launched.
    world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, "launch", "house_aruco.launch.py")
        ),
        launch_arguments={
            "gui": LaunchConfiguration("gui"),
        }.items(),
    )

    odom_tf = Node(
        package="exploration_assignment",
        executable="odom_tf_node.py",
        name="odom_tf_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    node = Node(
        package="exploration_assignment",
        executable="assignment3_node.py",
        name="assignment3_node",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "mode": "checkpoint",

            # Faster Task 2 waypoint execution.
            "max_linear_speed": 0.24,
            "max_angular_speed": 1.80,
            "map_csv": LaunchConfiguration("map_csv"),
            "output_dir": "/home/ros/ws/Assignment_3/results/task2",
            "random_seed": LaunchConfiguration("random_seed"),
        }],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            os.path.join(pkg, "rviz", "exploration.rviz"),
        ],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(args + [world, odom_tf, node, rviz])
