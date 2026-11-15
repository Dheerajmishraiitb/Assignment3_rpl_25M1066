#!/usr/bin/env python3
"""World + robot bring-up, Gazebo Classic (ROS 2 Humble).
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def cpu_quota():
    """CPUs this process may actually use, not how many the kernel can see.

    In a resource-limited container `nproc` reports the HOST's core count while the cgroup
    quota is far lower: measured 20 vs 4 in the course container. Every threaded library
    sizes its pool from the former -- OpenCV opened 20 threads, llvmpipe 8 -- so a 4-core
    student allocation ends up oversubscribed ~7x and spends its time context switching.
    """
    try:                                            # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(float(quota) / float(period)))
    except Exception:
        pass
    try:                                            # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, q // p)
    except Exception:
        pass
    return os.cpu_count() or 2


MODEL = "burger_cam"


def generate_launch_description():
    pkg = get_package_share_directory("exploration_assignment")
    gazebo_ros = get_package_share_directory("gazebo_ros")

    cfg = yaml.safe_load(open(os.path.join(pkg, "config", "markers.yaml")))
    sp = cfg["spawn"]
    rooms = {r["letter"]: r for r in cfg.get("rooms", []) if r.get("spawn")}
    room = os.environ.get("CS7001_ROOM", "").strip().upper()
    if room:
        if room not in rooms:
            raise RuntimeError(
                f"room '{room}' has no spawn preset; choose from "
                f"{sorted(rooms)}")
        x, y, yaw = rooms[room]["spawn"]
        sp = {"x": x, "y": y, "yaw": yaw}
        print(f"[cs7001] starting in room {room} ({rooms[room]['name']}) "
              f"at ({x:+.2f}, {y:+.2f}, {yaw:+.0f} deg)")

    world = os.path.join(pkg, "worlds", "turtlebot3_house_exploration.world")
    urdf = os.path.join(pkg, "urdf", f"turtlebot3_{MODEL}.urdf")
    sdf = os.path.join(pkg, "models", "turtlebot3_burger_cam_pinhole", "model.sdf")

    args = [
        DeclareLaunchArgument("x_pose", default_value=str(sp["x"])),
        DeclareLaunchArgument("y_pose", default_value=str(sp["y"])),
        DeclareLaunchArgument("yaw", default_value=str(sp["yaw"])),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("gui", default_value="true"),
    ]

    GAZEBO_PREFIX = "/usr/share/gazebo-11"

    resources = [
        SetEnvironmentVariable("LP_NUM_THREADS", str(max(1, min(8, cpu_quota() - 2)))),
        SetEnvironmentVariable("OMP_NUM_THREADS", str(max(1, cpu_quota() - 2))),
        SetEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            os.pathsep.join([os.path.join(GAZEBO_PREFIX, "models"),
                             os.path.join(pkg, "models")])),
        SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
        AppendEnvironmentVariable("GAZEBO_RESOURCE_PATH", GAZEBO_PREFIX),
        AppendEnvironmentVariable("GAZEBO_RESOURCE_PATH",
                                  os.path.join(pkg, "models")),
    ]

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, "launch", "gzserver.launch.py")),
        launch_arguments={"world": world,
                          "verbose": "true"}.items())

    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros, "launch", "gzclient.launch.py")),
        condition=IfCondition(LaunchConfiguration("gui")))

    with open(urdf) as f:
        robot_description = f.read()

    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        name="robot_state_publisher", output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time"),
                     "robot_description": robot_description}])

    spawn = Node(
        package="gazebo_ros", executable="spawn_entity.py", output="screen",
        arguments=["-entity", MODEL, "-file", sdf,
                   "-x", LaunchConfiguration("x_pose"),
                   "-y", LaunchConfiguration("y_pose"),
                   "-z", "0.01",
                   "-Y", LaunchConfiguration("yaw")])

    ld = LaunchDescription()
    for a in args:
        ld.add_action(a)
    for r in resources:
        ld.add_action(r)
    for n in (gzserver, gzclient, rsp, spawn):
        ld.add_action(n)
    return ld
