# ARNA Teleoperation and Manipulation System

ARNA is a distributed, safety-critical robot teleoperation and autonomous manipulation platform. It combines a Kinova Gen3 7-DOF arm with an omnidirectional mobile base, controlled remotely through a browser-based GUI over a Cloudflare-tunneled WebSocket connection.

A layered safety architecture continuously filters operator commands through network-quality monitoring, Model Predictive Control with Control Barrier Functions (MPC-CBF) on the arm, a CBF-QP filter on the base, network-aware constraint tightening, and an online operator-intent estimator — all before any velocity command reaches the hardware.

---

## System Overview

| Machine | Role | OS / ROS | IP |
|---------|------|----------|----|
| Legion | ROS master, arm control, safety nodes, web server | Ubuntu 20.04 / ROS Noetic | `10.0.0.101` |
| Blackbird | Base EtherCAT controller, base safety filter | Ubuntu 16.04 / ROS Kinetic | `10.0.0.20` |
| Jetson | Velodyne LiDAR, navigation stack | Ubuntu 18.04 / ROS Kinetic | `10.0.0.60` |
| Velodyne VLP-16 | 3-D LiDAR sensor | — | `10.0.0.40` |

- **Web GUI:** [https://arnaconnect.stream](https://arnaconnect.stream) (Cloudflare tunnel)
- **Main package:** `pick_place` (on Legion)
- **Base package:** `arna_teleop` (on Blackbird)

---

## Safety-Critical Control Architecture

Operator commands pass through four cascaded safety layers before reaching the robot hardware.

### Network Quality Monitor

Runs on Legion at 10 Hz. Measures browser-reported round-trip time (RTT) over a rolling 30-sample window and publishes a normalised quality score to `/network_quality`. This signal feeds both the watchdog and the arm CBF filter.

### Arm MPC-CBF Filter

Runs on Legion at 100 Hz. Intercepts desired Cartesian arm velocities and solves a Control Barrier Function QP (CasADi / qpOASES) that enforces joint-limit and collision-avoidance constraints. The desired command is attenuated in proportion to the combined safety signal before entering the QP, so authority is reduced continuously as network quality or operator alignment degrades.

Configuration: `ros/src/pick_place/config/mpc_cbf_params.yaml`

### Base CBF-QP Filter

Runs on **Blackbird** at 50 Hz (C++, OSQP v0.6.3). Intercepts desired base velocities, fuses live LiDAR scan data for obstacle proximity, and publishes a guaranteed-safe velocity to the EtherCAT drive. Obstacle clearance is maintained even if the network fails. Like the arm filter, the desired command is attenuated by the combined safety signal before the QP is solved.

Key parameters: `d_safe = 0.50 m`, `d_activate = 1.50 m`, `v_max_lin = 0.15 m/s`, `v_max_ang = 0.10 rad/s`

Configuration: `config/base_cbf_params.yaml` (on Blackbird)

### Network Watchdog

Runs on Legion. Monitors network quality and transitions through four modes, automatically tightening safety constraints and reducing operator authority as conditions worsen:

| Mode | Trigger | Effect |
|------|---------|--------|
| NOMINAL | Normal RTT | No constraint tightening |
| DEGRADED | Moderate RTT | Tighten CBF bounds, reduce MPC horizon |
| POOR | High RTT | Further tighten bounds, cap MPC horizon |
| FAILED | Link loss | Flood zero commands to both arm and base |

Updates arm and base filter parameters in real time via `dynamic_reconfigure` and publishes the active mode to `/safety_mode` for display in the GUI.

### Operator Intent Estimator

Runs on Legion at 20 Hz. Estimates online how well the operator's commands align with the robot's safe reference trajectory using a scalar alignment coefficient updated by a normalised gradient descent rule. When the operator consistently fights the safety filters, authority is reduced for both the arm and base. Authority recovers automatically when alignment improves or the operator is idle.

Configuration: `ros/src/pick_place/config/operator_intent_params.yaml`

---

## Dependencies

### Legion (ROS Noetic)

- [kortex_driver](https://github.com/Kinovarobotics/ros_kortex)
- [ros_kortex_vision](https://github.com/Kinovarobotics/ros_kortex_vision)
- `ros-noetic-rosbridge-server`
- `ros-noetic-web-video-server`
- `ros-noetic-dynamic-reconfigure`
- `python3-catkin-tools`

### Blackbird (ROS Kinetic)

- OSQP v0.6.3
- Eigen3

### Python

- Python 3.8+
- See `requirements.txt` for the full list (key packages: `casadi`, `numpy`, `ultralytics`)

---

## Building

**Legion:**
```bash
cd ~/ros
source devel/setup.bash
rosdep install --from-paths src --ignore-src -r -y
pip install -r src/pick_place/requirements.txt
catkin build pick_place
```

> Do **not** use `catkin_make` on Legion — the workspace uses `catkin build`.

**Blackbird** (SSH in first):
```bash
cd ~/ros/arna_ws
catkin_make --only-pkg-with-deps arna_teleop
```

---

## Running

**1. Start Legion ROS stack:**
```bash
export ROS_IP=10.0.0.101
roslaunch pick_place pick_place.launch
```

All safety nodes launch automatically. Individual nodes can be disabled via launch arguments:

```bash
roslaunch pick_place pick_place.launch \
  enable_network_monitor:=true \
  enable_network_watchdog:=true \
  enable_operator_intent:=true
```

**2. Start Blackbird base stack** (SSH into Blackbird, then):
```bash
sudo -s
cd ros/arna_ws/src/arna_teleop/src/
./base_interface.sh
```

**3. Web GUI and Cloudflare tunnel** are systemd services on Legion:
```bash
sudo systemctl restart arna-control
sudo systemctl status cloudflared
```

The GUI is accessible remotely at [https://arnaconnect.stream](https://arnaconnect.stream) or locally over the LAN.

---

## Web Interface

The GUI is a Next.js application connecting to three independent rosbridge WebSocket endpoints:

| Endpoint | Purpose |
|----------|---------|
| `wss://websocket.arnaconnect.stream` | Control plane (arm, base, gripper, services) |
| `wss://basewebsocket.arnaconnect.stream` | Base camera stream |
| `wss://armwebsocket.arnaconnect.stream` | Arm camera stream |

**Controls:**
- Arm Cartesian velocity (2D joystick)
- Base translation and rotation (1D / 2D joystick)
- Gripper open / close with force feedback
- Home action
- Click-to-pick: click on the arm camera feed to segment an object and trigger autonomous grasping
- Safety mode badge (NOMINAL / DEGRADED / POOR / FAILED)
- Live arm and base camera feeds

---

## Autonomous Pick-and-Place

### Object Segmentation

1. The operator clicks a point in the arm camera feed.
2. [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) segments the object at the clicked point.
3. The mask is applied to both colour and depth images.
4. Object orientation is estimated by fitting a minimum-area bounding rectangle to the mask contour.
5. The 3-D grasp point is computed from the mask centroid and a noise-robust depth estimate.

### Grasp Execution

1. Align the camera over the object using visual servoing.
2. Re-segment from the camera centre.
3. Approach along the object surface normal until it fills the image.
4. Re-segment; align the gripper to the object orientation.
5. Open gripper → move to grasp pose → close gripper → return to home pose.

---

## Repository Structure

```
ros/src/pick_place/
  scripts/
    main.py                      # Pick-and-place pipeline + arm camera node
    network_monitor_node.py      # Network quality monitor
    mpc_cbf_arm_node.py          # MPC-CBF arm safety filter (100 Hz)
    network_watchdog_node.py     # Network watchdog + mode coordinator
    operator_intent_node.py      # Operator intent estimator
  config/
    mpc_cbf_params.yaml          # Arm CBF tuning
    operator_intent_params.yaml  # Operator intent tuning
  cfg/
    MpcCbfArm.cfg                # dynamic_reconfigure schema for arm filter
    OperatorIntent.cfg           # dynamic_reconfigure schema for intent node
  launch/
    pick_place.launch            # Main launch file

ros/arna_ws/src/arna_teleop/     # On Blackbird
  src/
    base_cbf_filter_node.cpp     # CBF-QP base safety filter (50 Hz)
    arna_teleop_fwd_node.cpp     # Teleop forwarder
  config/
    base_cbf_params.yaml         # Base CBF tuning

arna-control/                    # Next.js web GUI
  components/
    Joystick2D.tsx
    Joystick1D.tsx
    TFViewer.tsx
```

---

## Tuning

All safety parameters are set via YAML config files. **Do not edit node source files to change tuning values.**

| Config file | Controls |
|-------------|----------|
| `config/mpc_cbf_params.yaml` | Arm CBF: MPC horizon, CBF bounds, joint limits, velocity caps |
| `config/base_cbf_params.yaml` (Blackbird) | Base CBF: obstacle margins, velocity limits, jerk weight, slack penalty |
| `config/operator_intent_params.yaml` | Intent estimator: learning rate, sigmoid sharpness, authority bounds, idle timeout |

Parameters can also be updated at runtime via `dynamic_reconfigure` or automatically by the network watchdog as network conditions change.
