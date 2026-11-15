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

    rviz = LaunchConfiguration("rviz")
    gui = LaunchConfiguration("gui")
    output_dir = LaunchConfiguration("output_dir")
    map_name = LaunchConfiguration("map_name")

    # Original house used for Tasks 1 and 2.
    house = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, "launch", "house_aruco.launch.py")
        ),
        launch_arguments={
            "gui": gui,
        }.items(),
    )

    # Gazebo publishes /odom but not odom -> base_footprint TF.
    odom_tf = Node(
        package="exploration_assignment",
        executable="odom_tf_node.py",
        name="odom_tf_node",
        output="screen",
    )

    assignment = Node(
        package="exploration_assignment",
        executable="assignment3_node.py",
        name="assignment3_node",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "mode": "mapping",
            "output_dir": output_dir,
            "map_name": map_name,

            # Task 1 occupancy-grid geometry.
            "resolution": 0.10,
            "origin_x": -8.5,
            "origin_y": -4.8,
            "width": 170,
            "height": 108,
        }],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            os.path.join(pkg, "rviz", "exploration.rviz"),
        ],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "rviz",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "gui",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value="/home/ros/ws/Assignment_3/results/task1",
        ),
        DeclareLaunchArgument(
            "map_name",
            default_value="manual_map",
        ),

        house,
        odom_tf,
        assignment,
        rviz_node,
    ])
