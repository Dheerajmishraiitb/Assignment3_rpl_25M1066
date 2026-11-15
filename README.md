# Assignment 3 - Mapping, Path Planning and Exploration

**Course:** CS7001: Robot Perception and Learning

**Semester:** Autumn 2026

**Due date:** 24th Oct 2026

**Total marks:** 10

---

## Description

In this assignment, you will build the main components of an autonomous exploration system for a TurtleBot3 Burger in Gazebo.

The assignment covers:

- **Occupancy-grid mapping** from lidar measurements;
- **A\*** path planning on the occupancy grid;
- **Frontier detection and clustering**;
- **Greedy exploration** using the largest available frontier;
- **MCTS-based exploration** over possible frontier sequences;
- **Evaluation and comparison** using exploration progress, map entropy, trajectories, and qualitative analysis.

You need to edit **ONLY one file**:

`src/exploration/scripts/exploration.py`

Complete only the regions marked by:

```python
# INSERT CODE HERE

# CODE ENDS HERE
```

Do not modify the provided function signatures or surrounding infrastructure.

## Setup

Use the same course Docker setup used for the previous assignments. No additional package installation is required for Assignment 3.

Place or clone the Assignment 3 folder inside:

```text
~/CS7001-docker-setup/workspace/
```

so that the assignment is available at:

```text
Laptop:
~/CS7001-docker-setup/workspace/Assignment_3

Inside the container:
/home/ros/ws/Assignment_3
```

Start the existing course container as usual, then build the assignment from inside the container:

```bash
cd /home/ros/ws/Assignment_3
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Edit `exploration.py` on your laptop. Run all ROS 2 commands inside the container. Gazebo and RViz are accessed through the same noVNC browser interface used in the previous assignments.

---

# Task 1 — Build and Save an Occupancy Grid Map

## Goal

Manually drive the TurtleBot through the environment and build an occupancy-grid map using your implementation of `update_occupancy_grid(...)`.

At the end of the task, save:

```text
results/task1/
├── manual_map.png
├── manual_map.csv
├── manual_map.yaml
└── trajectory.csv
```

The saved map will be reused in Task 2.

## 1. Build the workspace

Open a terminal inside the container:

```bash
cd /home/ros/ws/Assignment_3
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

You only need to rebuild after changing or adding installed files. With `--symlink-install`, normal edits to Python source files are picked up directly.

## 2. Start Task 1

In **Terminal 1**:

```bash
cd /home/ros/ws/Assignment_3
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch exploration_assignment mapping.launch.py
```

Wait for Gazebo to start and for the robot to appear.

## 3. Start the localization backend

In **Terminal 2**:

```bash
source /opt/ros/humble/setup.bash
source /home/ros/ws/Assignment_3/install/setup.bash

ros2 run slam_toolbox async_slam_toolbox_node   --ros-args   --params-file $(ros2 pkg prefix slam_toolbox)/share/slam_toolbox/config/mapper_params_online_async.yaml   -p use_sim_time:=true   -r /map:=/slam_backend/map   -r /map_metadata:=/slam_backend/map_metadata   -r /map_updates:=/slam_backend/map_updates
```

The SLAM backend is used only to provide the robot pose. Its occupancy map is not used for the assignment map.

## 4. Start the Assignment 3 teleop

Open a terminal inside the noVNC desktop:

```bash
source /opt/ros/humble/setup.bash
source /home/ros/ws/Assignment_3/install/setup.bash
ros2 run exploration_assignment assignment3_teleop.py
```

Controls:

```text
W       increase forward speed incrementally
X       increase backward speed incrementally
S       stop immediately
Hold A  turn left
Hold D  turn right
Q       stop and quit
```

Click inside the terminal before using the keys.

## 5. Drive through the environment

Drive through the accessible parts of the house so that the lidar observes the walls and free space.

Try to:

- cover all rooms and corridors;
- pass through doorways carefully;
- avoid collisions with walls;
- avoid unnecessary rapid turning;
- revisit areas when useful to improve coverage.

The occupancy grid should update continuously in RViz.

The assignment map is published on:

```text
/map
```

The current localization pose is published on:

```text
/assignment3/current_pose
```

## 6. Check the map in RViz

Use:

```text
Fixed Frame: map
```

Useful displays are:

```text
Map        → /map
Pose       → /assignment3/current_pose
LaserScan  → /scan
RobotModel
TF
```

Before saving, check that:

- the main walls are visible;
- rooms and doorways are represented;
- enough free space has been observed for later navigation;
- the map is not missing a major section of the environment.

## 7. Save the map

When you are satisfied with the map, run:

```bash
ros2 service call /assignment3/save_map std_srvs/srv/Trigger "{}"
```

The task should save:

```text
results/task1/manual_map.png
results/task1/manual_map.csv
results/task1/manual_map.yaml
results/task1/trajectory.csv
```

## 8. Verify the output

Run:

```bash
ls -lh /home/ros/ws/Assignment_3/results/task1
```

You should see all four files.

### File meanings

`manual_map.png`
: Image representation of the occupancy map.

`manual_map.csv`
: Exact occupancy-grid values used by later tasks.

`manual_map.yaml`
: Map metadata including resolution and origin.

`trajectory.csv`
: Robot trajectory recorded while building the map.

## Task 1 completion check

Task 1 is complete when:

- your occupancy-grid mapping implementation runs correctly;
- you have manually covered the environment;
- the resulting map is usable for navigation;
- all four Task 1 output files have been saved.

---
# Task 2 — A* Checkpoint Navigation

## Goal

Use the saved Task 1 occupancy map and your implementation of `astar(...)` to autonomously navigate through six spatially distributed checkpoints and finally return Home.

The mission sequence is:

```text
HOME
  ↓
CP1
  ↓
CP2
  ↓
CP3
  ↓
CP4
  ↓
CP5
  ↓
CP6
  ↓
HOME
```

The six checkpoints are generated automatically from different spatial regions of the known free map.

At the end of the task, the navigation results are saved automatically.

## 1. Make sure Task 1 has been completed

Task 2 requires:

```text
results/task1/manual_map.csv
results/task1/manual_map.yaml
results/task1/manual_map.png
```

Check:

```bash
ls -lh /home/ros/ws/Assignment_3/results/task1
```

## 2. Build the workspace

Inside the container:

```bash
cd /home/ros/ws/Assignment_3
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 3. Start Task 2

In **Terminal 1**:

```bash
cd /home/ros/ws/Assignment_3
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch exploration_assignment checkpoint_navigation.launch.py
```

This starts the simulator, loads the saved Task 1 occupancy map, launches the autonomous checkpoint-navigation node, and opens RViz.

Do not use teleop during Task 2.

## 4. Start the localization backend

After Gazebo has started and the robot has spawned, use **Terminal 2**:

```bash
source /opt/ros/humble/setup.bash
source /home/ros/ws/Assignment_3/install/setup.bash

ros2 run slam_toolbox async_slam_toolbox_node   --ros-args   --params-file $(ros2 pkg prefix slam_toolbox)/share/slam_toolbox/config/mapper_params_online_async.yaml   -p use_sim_time:=true   -r /map:=/slam_backend/map   -r /map_metadata:=/slam_backend/map_metadata   -r /map_updates:=/slam_backend/map_updates
```

Once localization becomes available, the mission starts automatically.

## 5. Watch the mission begin

The Task 2 terminal should print messages similar to:

```text
Home recorded at (...)
Mission: START -> CP1(...) -> CP2(...) -> CP3(...) -> CP4(...) -> CP5(...) -> CP6(...) -> HOME(...)
Navigating to CP1
```

The robot should then move autonomously.

## 6. Visualize Task 2 in RViz

Use:

```text
Fixed Frame: map
```

Recommended displays:

```text
Map          → /map
Pose         → /assignment3/current_pose
Path         → /exploration/path
MarkerArray  → /assignment3/checkpoints
Marker       → /exploration/goal
LaserScan    → /scan
RobotModel
TF
```

The six checkpoints should appear as markers distributed across the known map.

The currently planned A* route is published on:

```text
/exploration/path
```

The robot replans while executing the mission, so the displayed path may update during navigation.

## 7. Let the mission run to completion

Do not manually control the robot.

The expected sequence is:

```text
Reached CP1
Navigating to CP2

Reached CP2
Navigating to CP3

...

Reached CP6
Navigating to HOME

Reached HOME
MISSION_COMPLETE ...
```

The final `MISSION_COMPLETE` message is shown prominently in the terminal.

## 8. Check the automatically saved results

After the robot returns Home, Task 2 automatically saves:

```text
results/task2/
├── checkpoints.csv
└── trajectory.csv
```

Check:

```bash
ls -lh /home/ros/ws/Assignment_3/results/task2
```

### `checkpoints.csv`

Contains the Home position and the generated checkpoint locations, including map/grid and image coordinates.

### `trajectory.csv`

Contains the executed robot trajectory, including:

```text
time_s
x_m
y_m
yaw_rad
grid_row
grid_col
image_u
image_v
target
```

The `image_u` and `image_v` columns are provided so that the executed trajectory can be plotted directly over:

```text
results/task1/manual_map.png
```

You are expected to produce your own visualization from these saved results.

## Reproducible checkpoint generation

Checkpoint generation uses a random seed.

The default run is:

```bash
ros2 launch exploration_assignment checkpoint_navigation.launch.py
```

To reproduce or test a different checkpoint set:

```bash
ros2 launch exploration_assignment checkpoint_navigation.launch.py random_seed:=11
```

The same map and same random seed should reproduce the same checkpoint selection.

## Task 2 completion check

Task 2 is complete when:

- your `astar(...)` implementation successfully plans paths;
- six checkpoints are generated across the known map;
- the robot autonomously visits CP1 through CP6;
- the robot returns Home;
- `MISSION_COMPLETE` is printed;
- `checkpoints.csv` and `trajectory.csv` are saved;
- you can use the saved data to visualize the executed trajectory on the Task 1 map.

---
# Task 3 — Frontier Detection and Clustering

## Goal

Implement `frontier_clustering(...)` in:

```text
src/exploration/scripts/exploration.py
```

Task 3 is a pure algorithmic task. You do **not** need to launch Gazebo or move the robot for this task.

Your function must:

1. detect frontier cells in an occupancy grid;
2. group connected frontier cells into clusters;
3. discard clusters smaller than `min_cluster_size`.

## Frontier definition

The occupancy-grid convention is:

```text
unknown   = -1
free      = 0
occupied  = 100
```

A cell is a **frontier cell** if:

- the cell itself is known free (`0`), and
- at least one of its **4-connected neighbours** is unknown (`-1`).

Diagonal unknown cells alone do not make a free cell a frontier.

## Frontier clustering

After frontier cells have been identified, group connected frontier cells into clusters.

Use **8-connectivity** for clustering, so frontier cells may belong to the same cluster even when they touch diagonally.

The function returns:

```python
List[List[GridCell]]
```

Clusters with fewer than `min_cluster_size` cells must be discarded.

## Function to implement

```python
def frontier_clustering(
    occupancy: np.ndarray,
    min_cluster_size: int = 4,
) -> List[List[GridCell]]:
```

Only edit the section between:

```python
# INSERT CODE HERE
```

and:

```python
# CODE ENDS HERE
```

Do not modify the provided function signature or surrounding helper functions.

## Offline self-check

Run:

```bash
cd ~/CS7001-docker-setup/workspace/Assignment_3
python3 tests/task3_self_check.py
```

The script checks:

- a single frontier cluster;
- two disconnected frontier clusters;
- minimum cluster-size filtering;
- a fully known map with no frontiers;
- correct use of 4-connected unknown neighbours.

Expected successful ending:

```text
Passed 5/5 tests.
ALL TASK 3 SELF-CHECKS PASSED
```

## Important implementation notes

Make sure that:

- neighbour indices are checked before accessing the grid;
- only free cells can become frontier cells;
- frontier detection uses 4-connectivity;
- clustering uses 8-connectivity;
- each frontier cell appears in at most one cluster;
- small clusters are filtered using `min_cluster_size`.

Do not rely on ROS topics, Gazebo, or simulator state inside `frontier_clustering(...)`.

## Task 3 completion check

Task 3 is complete when:

- `frontier_clustering(...)` correctly detects frontier cells;
- frontier cells are grouped using 8-connectivity;
- clusters smaller than `min_cluster_size` are removed;
- `python3 tests/task3_self_check.py` reports all tests passed.

The online integration of Task 3 is tested in Task 4.

---
# Task 4 — Greedy Frontier Exploration

## Goal

Use the frontier clusters from Task 3 to perform autonomous exploration.

The Task 4 policy is:

> **Select the largest currently available frontier cluster.**

Frontier size is measured by the number of frontier cells contained in each cluster.

This gives a simple greedy baseline for comparison with MCTS in Task 5.

Unlike Task 2, Task 4 does **not** load the saved Task 1 map. The robot starts with an initially unknown assignment occupancy grid and builds the map online while exploring.

## 1. Greedy frontier-selection rule

At each exploration decision:

1. compute the current frontier clusters using `frontier_clustering(...)`;
2. determine the number of frontier cells in each cluster;
3. select the cluster containing the largest number of frontier cells;
4. choose the representative point of the selected cluster;
5. plan to the selected frontier using your A* implementation;
6. navigate toward the selected frontier;
7. once the goal is reached, fails, or times out, select the next frontier.

The greedy policy reasons only about the current frontier set. It does not explicitly simulate future frontier choices.

## 2. Function to implement

Complete the provided greedy frontier-selection function in:

```text
src/exploration/scripts/exploration.py
```

Only edit the section between:

```python
# INSERT CODE HERE
```

and:

```python
# CODE ENDS HERE
```

Do not modify the provided function signature or surrounding infrastructure.

## 3. Start Task 4

In **Terminal 1**:

```bash
cd /home/ros/ws/Assignment_3
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch exploration_assignment exploration.launch.py method:=greedy
```

Wait for Gazebo and RViz to start.

Do not use teleop during Task 4.

## 4. Start the localization backend

In **Terminal 2**:

```bash
source /opt/ros/humble/setup.bash
source /home/ros/ws/Assignment_3/install/setup.bash

ros2 run slam_toolbox async_slam_toolbox_node   --ros-args   --params-file $(ros2 pkg prefix slam_toolbox)/share/slam_toolbox/config/mapper_params_online_async.yaml   -p use_sim_time:=true   -r /map:=/slam_backend/map   -r /map_metadata:=/slam_backend/map_metadata   -r /map_updates:=/slam_backend/map_updates
```

The SLAM backend provides localization only. The assignment occupancy map is built independently and published on `/map`.

## 5. Visualize Task 4 in RViz

Use:

```text
Fixed Frame: map
```

Recommended displays:

```text
Map          → /map
Pose         → /assignment3/current_pose
Path         → /exploration/path
Marker       → /exploration/goal
LaserScan    → /scan
RobotModel
TF
```

During the run, RViz should show the online occupancy map, detected frontiers, selected frontier, current A* path, and robot pose.

## 6. Online exploration loop

Conceptually, Task 4 repeats:

```text
update occupancy map
        ↓
detect frontier cells
        ↓
cluster frontiers
        ↓
select largest frontier
        ↓
plan with A*
        ↓
move toward selected frontier
        ↓
repeat
```

Once a frontier goal has been selected, the robot continues toward that goal until it is reached, becomes unreachable, or times out. A new frontier is selected only after the current goal is completed or abandoned.

## 7. Exploration completion

Exploration terminates automatically when no usable frontier remains for the configured completion interval.

At completion, the terminal prints:

```text
EXPLORATION_COMPLETE ...
```

The robot stops and remains stopped.

There is no post-exploration checkpoint mission.

## 8. Saved results

Task 4 saves the final map and exploration trace under:

```text
results/task4/
```

The main files are:

```text
results/task4/
├── exploration_greedy.png
├── exploration_greedy.csv
├── exploration_greedy.yaml
└── exploration_greedy_coverage.csv
```

The coverage CSV contains:

```text
time_s
known_cells
travel_m
map_entropy
frontier_cells
frontier_clusters
```

## 9. Data for later analysis

Task 4 records the exploration statistics needed for the later plotting and analysis tasks in:

```text
results/task4/exploration_greedy_coverage.csv
```

You will use this data in Task 6.

**No plotting helper code is provided.** You are expected to write your own plotting script using the saved CSV data.

## Task 4 completion check

Task 4 is complete when:

- Task 3 frontier clustering works online;
- the greedy policy selects the largest frontier cluster;
- A* generates paths to selected frontiers;
- the robot explores autonomously;
- exploration terminates automatically;
- `EXPLORATION_COMPLETE` is printed;
- the final map and coverage trace are saved successfully.

---
# Task 5 — MCTS Frontier Exploration

## Goal

Replace the one-step greedy frontier-selection rule from Task 4 with a Monte Carlo Tree Search policy.

Instead of selecting a frontier using only the current map, MCTS evaluates possible **sequences of frontier visits** and balances:

- expected information gain;
- travel cost;
- exploration of less-tested choices.

The occupancy-grid mapper, frontier detector, A* planner, and robot controller are reused from earlier tasks.

## 1. MCTS search structure

At each exploration decision, frontier clusters define the available actions.

MCTS repeatedly performs:

```text
Selection
    ↓
Expansion
    ↓
Rollout
    ↓
Backpropagation
```

After the search budget is exhausted, the selected root action becomes the next real frontier goal.

The selected frontier becomes the current navigation goal. The robot commits to that goal until it is reached, becomes unreachable, or times out. MCTS is then run again using the updated real map.

## 2. Function to implement

Implement:

```python
mcts_frontier(...)
```

in:

```text
src/exploration/scripts/exploration.py
```

Only edit between:

```python
# INSERT CODE HERE
```

and:

```python
# CODE ENDS HERE
```

Do not modify the supplied function signature.

The function is provided with the information needed for the search, including the frontier clusters, occupancy grid, robot pose, map geometry, sensor range, iteration budget, rollout depth, exploration constant, reward weights, candidate limit, and random-number generator.

## 3. Reward

The search balances information gain against travel cost:

```text
reward
=
information-gain reward
-
travel-cost penalty
```

Information already assumed to have been revealed earlier in the same hypothetical sequence should not be rewarded again.

Therefore, the search should reason using **marginal information gain**.

## 4. UCT selection

During tree selection, use the supplied exploration constant in the UCT rule to balance:

```text
exploitation + exploration
```

The exploitation term favors actions with high estimated value.

The exploration term favors actions that have been tested less often.

## 5. Start Task 5

In **Terminal 1**:

```bash
cd /home/ros/ws/Assignment_3
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch exploration_assignment exploration.launch.py method:=mcts
```

Do not use teleop during Task 5.

## 6. Start the localization backend

In **Terminal 2**:

```bash
source /opt/ros/humble/setup.bash
source /home/ros/ws/Assignment_3/install/setup.bash

ros2 run slam_toolbox async_slam_toolbox_node   --ros-args   --params-file $(ros2 pkg prefix slam_toolbox)/share/slam_toolbox/config/mapper_params_online_async.yaml   -p use_sim_time:=true   -r /map:=/slam_backend/map   -r /map_metadata:=/slam_backend/map_metadata   -r /map_updates:=/slam_backend/map_updates
```

## 7. Visualize Task 5 in RViz

Use:

```text
Fixed Frame: map
```

Recommended displays:

```text
Map          → /map
Pose         → /assignment3/current_pose
Path         → /exploration/path
Marker       → /exploration/goal
MarkerArray  → /exploration/mcts_debug
LaserScan    → /scan
RobotModel
TF
```

The MCTS diagnostic visualization is published on:

```text
/exploration/mcts_debug
```

It can show:

```text
sampled rollouts     simulated future frontier sequences
best branch          strongest future branch found by the search
N / Q labels         root visit counts and estimated values
```

These markers visualize the search only. The robot still executes the normal A* route toward the selected frontier.

## 8. Online exploration loop

Task 5 repeatedly performs:

```text
update occupancy map
        ↓
detect and cluster frontiers
        ↓
construct candidate actions
        ↓
run MCTS
        ↓
select root action
        ↓
plan with A*
        ↓
move
        ↓
repeat
```

Because MCTS reasons about future possibilities, the selected frontier does not necessarily have to be the largest or closest current frontier.

## 9. Exploration completion

Task 5 uses the same exploration-completion condition as Task 4.

At completion:

```text
EXPLORATION_COMPLETE ...
```

is printed, the robot stops, and the exploration visualization is cleared.

There is no post-exploration checkpoint mission.

## 10. Saved results

Task 5 saves:

```text
results/task5/
├── exploration_mcts.png
├── exploration_mcts.csv
├── exploration_mcts.yaml
└── exploration_mcts_coverage.csv
```

The coverage CSV contains:

```text
time_s
known_cells
travel_m
map_entropy
frontier_cells
frontier_clusters
```

## 11. Data for later analysis

Task 5 records the exploration statistics needed for the later plotting and analysis tasks in:

```text
results/task5/exploration_mcts_coverage.csv
```

You will use this data in Task 7.

**No plotting helper code is provided.** You are expected to write your own plotting script using the saved CSV data.

## Task 5 completion check

Task 5 is complete when:

- `mcts_frontier(...)` performs a valid MCTS search;
- selection uses UCT;
- expansion introduces untried actions;
- rollout evaluates possible future frontier sequences;
- backpropagation updates visits and values;
- the executed action is selected from the root search result;
- the robot explores autonomously;
- the MCTS diagnostic visualization is available in RViz;
- exploration terminates automatically;
- `EXPLORATION_COMPLETE` is printed;
- the final map and coverage trace are saved successfully.

---

# Assignment completion summary

```text
Task 1
✓ occupancy-grid mapping
✓ saved map + trajectory

Task 2
✓ A* checkpoint navigation
✓ CP1 → CP6 → Home
✓ saved checkpoints + trajectory

Task 3
✓ frontier detection
✓ frontier clustering
✓ offline self-checks pass

Task 4
✓ largest-frontier greedy exploration
✓ autonomous completion
✓ saved map + exploration statistics

Task 5
✓ MCTS frontier exploration
✓ autonomous completion
✓ saved map + exploration statistics

Task 6  
✓ greedy exploration progress and entropy plots  
  
Task 7  
✓ MCTS exploration progress and entropy plots  
  
Task 8  
✓ greedy vs MCTS analysis  
```

The assignment progression is:

```text
Mapping
   ↓
A* Planning
   ↓
Frontier Detection
   ↓
Greedy Exploration
   ↓
MCTS Exploration
```

---

# Submission Instructions

Create a folder named:

```text
submission/
```

Your final submission should have the following structure:

```text
submission/
├── results/
│   ├── task1/
│   ├── task2/
│   ├── task4/
│   └── task5/
│
├── plots/
│   ├── task1_trajectory_overlay.png
│   ├── task2_final_path_overlay.png
│   ├── task4_trajectory_overlay.png
│   ├── task5_trajectory_overlay.png
│   ├── task6_greedy_progress_vs_time.png
│   ├── task6_greedy_entropy_vs_time.png
│   ├── task7_mcts_progress_vs_time.png
│   └── task7_mcts_entropy_vs_time.png
│
└── exploration.md
```

The exact plot filenames may differ, but keep all report figures inside:

```text
submission/plots/
```

Copy the complete generated result folders for Tasks 1, 2, 4, and 5 into:

```text
submission/results/
```

Task 3 does not require a result folder because it is evaluated using the offline self-check output.

---

## Main report — `exploration.md`

Your main report must be named:

```text
exploration.md
```

The report should begin with:

```text
Name:
Roll Number:
```

Keep the report organized task-wise using the sections below.

You may briefly describe implementation details where useful, but the main purpose of the report is to present and analyse the results produced by your system.

---

## Task 1 — Occupancy-grid mapping

Briefly describe your mapping approach if required.

The main result for Task 1 should be:

- the final occupancy-grid map produced by your implementation;
- the complete manual teleoperation trajectory overlaid on top of the occupancy-grid map.

Your figure should make it easy to see both the reconstructed environment and the route followed while manually mapping it.

Include the final overlay figure in the report and keep a copy in:

```text
submission/plots/
```

The corresponding raw Task 1 results must be copied to:

```text
submission/results/task1/
```

---

## Task 2 — A* checkpoint navigation

You may briefly describe your A* implementation or navigation behaviour if useful.

The main Task 2 result should show:

- the Task 1 occupancy-grid map;
- the complete final trajectory followed by the robot during the checkpoint mission;
- the trajectory overlaid on the occupancy map.

The full path should show the complete mission:

```text
HOME → CP1 → CP2 → CP3 → CP4 → CP5 → CP6 → HOME
```

In addition, include **two intermediate RViz screenshots** captured while the autonomous run was in progress.

These screenshots should show useful intermediate states such as:

- the current checkpoint goal;
- the planned A* path;
- checkpoint markers;
- the robot part-way through the mission.

Keep the final path-overlay figure and the two RViz screenshots in:

```text
submission/plots/
```

Copy the raw Task 2 results to:

```text
submission/results/task2/
```

---

## Task 3 — Frontier detection and clustering

For Task 3, include the terminal output from:

```bash
python3 tests/task3_self_check.py
```

You may paste the output as formatted text or include a clear terminal screenshot.

The report should show that all provided Task 3 self-checks passed.

For example:

```text
Passed 5/5 tests.
ALL TASK 3 SELF-CHECKS PASSED
```

No separate Task 3 results folder is required.

---

## Task 4 — Greedy largest-frontier exploration

Briefly describe your greedy frontier-selection method if useful.

The main Task 4 result should show:

- the final occupancy-grid map built during greedy exploration;
- the complete trajectory followed by the robot;
- the trajectory overlaid on top of the final occupancy-grid map.

The figure should make the overall exploration behaviour easy to inspect.

Keep the overlay figure in:

```text
submission/plots/
```

Copy the complete Task 4 output folder to:

```text
submission/results/task4/
```

---

## Task 5 — MCTS frontier exploration

Briefly describe your MCTS frontier-selection method if useful.

The main Task 5 result should show:

- the final occupancy-grid map built during MCTS exploration;
- the complete trajectory followed by the robot;
- the trajectory overlaid on top of the final occupancy-grid map.

The figure should allow the MCTS exploration path to be compared visually with the greedy trajectory from Task 4.

Keep the overlay figure in:

```text
submission/plots/
```

Copy the complete Task 5 output folder to:

```text
submission/results/task5/
```

---

## Task 6 — Greedy exploration plots

Using the data recorded during Task 4, **write your own plotting script** and produce the following two plots for the greedy exploration run.

No plotting script or helper code is provided for this task. Your script should read the saved Task 4 result data and generate the required figures.

### Plot 1 — Exploration progress vs time

Use:

```text
x-axis: Time (s)
y-axis: Exploration progress (%)
```

### Plot 2 — Map entropy vs time

Use:

```text
x-axis: Time (s)
y-axis: Map entropy
```

Both plots must be included in `exploration.md` and copied to:

```text
submission/plots/
```

---

## Task 7 — MCTS exploration plots

Using the data recorded during Task 5, **write your own plotting script** and produce the same two plots for the MCTS exploration run.

No plotting script or helper code is provided for this task. Your script should read the saved Task 5 result data and generate the required figures.

### Plot 1 — Exploration progress vs time

Use:

```text
x-axis: Time (s)
y-axis: Exploration progress (%)
```

### Plot 2 — Map entropy vs time

Use:

```text
x-axis: Time (s)
y-axis: Map entropy
```

Both plots must be included in `exploration.md` and copied to:

```text
submission/plots/
```

Use consistent axis labels and presentation across Tasks 6 and 7 so that the two exploration methods can be compared fairly.

---

## Task 8 — Analysis: Greedy vs MCTS

Compare the behaviour of the greedy and MCTS exploration methods.

Your discussion should address at least the following questions:

1. **Was greedy exploration effective in this environment?**  
   Discuss what worked well and where the greedy largest-frontier strategy was limited.

2. **How did MCTS differ from greedy exploration?**  
   Use your trajectories, exploration-progress plots, and entropy plots to support the comparison.

3. **Which method explored more efficiently in your runs?**  
   Consider factors such as:
   - time required to reach high coverage;
   - trajectory behaviour;
   - unnecessary travel;
   - how quickly uncertainty or map entropy decreased.

4. **Which method would you expect to generalize better across different environments, and why?**  
   Do not base this only on which method happened to be faster in the provided environment. Discuss the underlying planning behaviour of both methods and how that could affect performance in environments with different layouts.

Your analysis should refer directly to the experimental results you obtained.

---

# Final submission checklist

Before submitting, verify that:

```text
submission/
├── results/
│   ├── task1/
│   ├── task2/
│   ├── task4/
│   └── task5/
├── plots/
└── exploration.md
```

contains all required material.

Check that:

- `exploration.md` begins with your name and roll number;
- Task 1 contains the final map with teleop trajectory overlay;
- Task 2 contains the final complete trajectory overlay and two intermediate RViz screenshots;
- Task 3 contains the self-check output;
- Task 4 contains the greedy final map with trajectory overlay;
- Task 5 contains the MCTS final map with trajectory overlay;
- Task 6 contains the two greedy plots;
- Task 7 contains the two MCTS plots;
- Task 8 contains the greedy-vs-MCTS analysis;
- raw result folders for Tasks 1, 2, 4, and 5 are present inside `submission/results/`;
- all report figures are also present inside `submission/plots/`.

