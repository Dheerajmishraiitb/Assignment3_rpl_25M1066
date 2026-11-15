# Vendored from ROBOTIS turtlebot3_simulations (Apache License 2.0)

These assets are copied verbatim from `turtlebot3_gazebo` so that this workspace
builds and runs from a clone with no external checkout:

    models/turtlebot3_common/meshes/    burger_base, lds, left_tire, right_tire
    models/turtlebot3_house/            first_2015_trash_can, mailbox
    ../urdf/turtlebot3_burger_cam.urdf  mesh paths repointed at this package
    ../urdf/common_properties.urdf
    ../params/turtlebot3_burger_cam_bridge.yaml

The model directory names are kept identical on purpose: the generated SDFs refer
to `model://turtlebot3_common/...` and `model://turtlebot3_house/...`, and Gazebo
resolves those against GZ_SIM_RESOURCE_PATH, so nothing had to be rewritten.

Only the 4 meshes actually used are here, not the whole 40 MB mesh set, and only
the burger_cam variant of the urdf/bridge params -- the launch sets
MODEL = "burger_cam", which is what tripped the first attempt at this.

Source: https://github.com/ROBOTIS-GIT/turtlebot3_simulations
The asset generator (not shipped here) reads the UPSTREAM house and burger_cam SDFs
when regenerating, so that checkout is needed to regenerate assets -- but not to
build or run.
