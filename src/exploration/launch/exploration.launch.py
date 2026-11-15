#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node


def generate_launch_description():

    pkg_dir = get_package_share_directory("exploration_assignment")

    method = LaunchConfiguration("method")
    rviz = LaunchConfiguration("rviz")
    gui = LaunchConfiguration("gui")
    random_seed = LaunchConfiguration("random_seed")

    # Task 4 = greedy exploration
    # Task 5 = MCTS exploration
    output_dir = PythonExpression([
        "'/home/ros/ws/Assignment_3/results/task4' "
        "if '", method, "' == 'greedy' "
        "else '/home/ros/ws/Assignment_3/results/task5'"
    ])

    house_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, "launch", "house_exploration.launch.py")
        ),
        launch_arguments={
            "gui": gui,
        }.items(),
    )

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
            "mode": method,
            "output_dir": output_dir,
            "random_seed": random_seed,

            # Tasks 4/5 exploration-map geometry.
            "resolution": 0.10,
            "origin_x": -8.5,
            "origin_y": -4.8,
            "width": 170,
            "height": 108,

            # Autonomous exploration turns substantially faster than
            # manual Task 1 mapping.
            "mapping_max_angular_rate": 1.50,
        }],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            os.path.join(pkg_dir, "rviz", "exploration.rviz"),
        ],
        condition=IfCondition(rviz),
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "method",
            default_value="greedy",
            description="Exploration method: greedy or mcts",
        ),
        DeclareLaunchArgument(
            "rviz",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "gui",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "random_seed",
            default_value="7",
        ),

        house_launch,
        odom_tf,
        assignment,
        rviz_node,
    ])
