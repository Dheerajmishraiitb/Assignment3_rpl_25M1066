#!/usr/bin/env python3
import os

from ament_index_python.packages import (PackageNotFoundError,
                                         get_package_share_directory)
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            LogInfo)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("exploration_assignment")

    args = [
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("noise", default_value="true",
                              description="corrupt odometry with the alpha model"),
        DeclareLaunchArgument("alpha", default_value="[0.30, 0.0, 0.1109, 0.0445]"),
        DeclareLaunchArgument("debug_image", default_value="true"),
        DeclareLaunchArgument("compressed", default_value="false"),
        DeclareLaunchArgument("debug_compressed", default_value="true"),
        DeclareLaunchArgument("plotjuggler", default_value="false",
                              description="open PlotJuggler to plot the "
                                          "/diag/* time series"),
    ]

    want_pj = IfCondition(LaunchConfiguration("plotjuggler"))
    try:
        get_package_share_directory("plotjuggler")
        plotjuggler = Node(
            package="plotjuggler", executable="plotjuggler", name="plotjuggler",
            output="screen", condition=want_pj,
            arguments=["--nosplash", "--start_streamer", "DataStreamROS2"])
    except PackageNotFoundError:
        plotjuggler = LogInfo(
            condition=want_pj,
            msg="plotjuggler:=true but PlotJuggler is not installed in this image.\n"
                "    sudo apt install ros-humble-plotjuggler-ros\n"
                "The /diag/* topics publish regardless; rqt_plot also works:\n"
                "    ros2 run rqt_plot rqt_plot /diag/ekf_err_pos/data")

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
               publish_debug_image=LaunchConfiguration("debug_image"),
               use_compressed=LaunchConfiguration("compressed"),
               debug_compressed=LaunchConfiguration("debug_compressed")),
        script("noisy_odom_node.py",
               enable_noise=LaunchConfiguration("noise"),
               alpha=LaunchConfiguration("alpha")),
        script("gt_pose_node.py"),
        
        script("ekf_node.py"),

        script("ekf_viz_node.py"),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", os.path.join(pkg, "rviz", "ekf.rviz")],
             parameters=[{"use_sim_time": True}],
             condition=IfCondition(LaunchConfiguration("rviz"))),
        plotjuggler,
    ])
