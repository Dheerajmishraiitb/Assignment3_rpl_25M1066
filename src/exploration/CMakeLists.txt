cmake_minimum_required(VERSION 3.8)
project(exploration_assignment)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch models worlds rviz data urdf
  DESTINATION share/${PROJECT_NAME}
)

# ekf_localization.py must be installed alongside ekf_node.py because ekf_node
# imports it from its own directory.
install(
  PROGRAMS
    scripts/assignment3_teleop.py
    scripts/odom_tf_node.py
    scripts/ekf_localization.py
    scripts/aruco_detector_node.py
    scripts/noisy_odom_node.py
    scripts/gt_pose_node.py
    scripts/ekf_node.py
    scripts/ekf_viz_node.py
    scripts/record_dataset.py
    scripts/evaluate.py
    scripts/exploration.py
    scripts/assignment3_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
