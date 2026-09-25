# ARNA Teleoperation and Manipulation System

ARNA is a distributed, safety-critical teleoperation and semi-autonomous manipulation platform. It combines a Kinova Gen3 7-DOF arm with an omnidirectional mobile base, both operated remotely through a browser GUI over a Cloudflare-tunneled WebSocket connection.

Every operator velocity command passes through a layered safety architecture before it reaches the hardware: a network-quality monitor, Model Predictive Control with Control Barrier Functions (MPC-CBF) on the arm and on the base, a network watchdog that tightens constraints as the link degrades, and an operator-intent estimator that reduces operator authority when the operator fights the safety filters on a degraded link.

> **Scope of this repository.** This is the Legion ROS workspace: the `pick_place` package (arm safety, network layers, operator intent, pick-and-place) plus the Kinova driver submodules. The base package `arna_teleop` (Blackbird) and the `arna-control` web GUI live in separate workspaces and are not included here; they are described below for context.

---

## System Overview

| Machine | Role | OS / ROS | IP |
|---------|------|----------|----|
| Legion | ROS master, arm control, safety layers 0/1/3/4, web GUI server | Ubuntu 20.04 / ROS Noetic | `10.0.0.101` |
| Blackbird | Base EtherCAT controller, base safety filter (layer 2) | Ubuntu 16.04 / ROS Kinetic | `10.0.0.20` |
| Jetson | Velodyne LiDAR driver, navigation stack | Ubuntu 18.04 / ROS Kinetic | `10.0.0.60` |
| Velodyne VLP-16 | 3-D LiDAR sensor | — | `10.0.0.40` |
| Kinova Gen3 | 7-DOF arm | — | `kinova.lan` |

- **Web GUI:** served from Legion through a Cloudflare Tunnel, behind Cloudflare Access
- **Legion package:** `pick_place` (this repository)
- **Blackbird package:** `arna_teleop` (separate workspace)

---

## Safety Architecture

```
Browser ping/pong ──► L0 network_monitor ──► /network_quality ──┬──► L3 network_watchdog ──► mode, λ_net, zero-flood
                                                               │            │ dynamic_reconfigure (margins, horizon)
                                                               │            ▼
Operator arm cmd ──────────────────────────────────────────────┴──► L1 mpc_cbf_arm_node (Legion) ──► Kinova arm
Operator base cmd ─────────────────────────────────────────────────► L2 base_mpc_cbf_node (Blackbird) ──► base drives
                                                                             ▲
                  L4 operator_intent_node: compares operator vs. safe commands ──► λ_combined ──┘ (both filters)
```

### Layer 0 — Network Quality Monitor (Legion, 10 Hz)

The browser sends ping messages that `network_probe_relay` echoes back over the same WebSocket path, so the measured round-trip time (RTT) reflects the real control link. `network_monitor_node` keeps a rolling 300-sample window (about 30 s at the 10 Hz probe rate) and publishes `/network_quality` (`pick_place/NetworkQuality`: RTT mean and standard deviation, jitter, loss rate, worst-case delay, and a discrete state).

| State | Condition |
|-------|-----------|
| NOMINAL | worst-case delay < 80 ms and loss < 1 % |
| DEGRADED | worst-case delay ≥ 80 ms or loss ≥ 1 % |
| POOR | worst-case delay ≥ 200 ms or loss ≥ 5 % |
| FAILED | no RTT sample received for 500 ms |

### Layer 1 — Arm MPC-CBF Filter (Legion, 100 Hz)

`mpc_cbf_arm_node` intercepts the desired Cartesian end-effector velocity and solves a parametric MPC-CBF quadratic program (CasADi, OSQP backend) that enforces discrete-time CBF constraints on the workspace box (six faces), end-effector speed, and acceleration, each with a penalized slack.

- Safety margins grow with RTT variability: `ε = ε_base + k_ε · σ_RTT`.
- The prediction horizon adapts to the measured delay, `N = clip(⌈δ_max / dt⌉, N_min, N_max)` with `N` between 10 and 25, and is pinned to `N_max` in FAILED.
- The desired command is low-pass filtered and scaled by `(1 − λ_combined)` before the QP. Only the tracking pull is reduced; the CBF constraints stay hard.
- A solve that overruns the 10 ms period is skipped rather than queued, so long horizons lower the achieved rate.
- While an autonomous pick is running (`/pick_running`), teleop output is suppressed.
- `enable_mpc_cbf:=false` replaces the filter with `arm_cmd_passthrough_node`, a zero-processing relay.

Configuration: `src/pick_place/config/mpc_cbf_params.yaml` · runtime schema: `src/pick_place/cfg/MpcCbfArm.cfg`

### Layer 2 — Base MPC-CBF Filter (Blackbird, 50 Hz, not in this repository)

`base_mpc_cbf_node` (C++, OSQP v0.6.3) solves an N = 20 horizon QP that tracks the desired base velocity subject to LiDAR-sector CBF constraints propagated over the full horizon. It runs on the base computer, so obstacle constraints remain enforced when the operator link drops. Like the arm filter, it scales the desired command by `(1 − λ_combined)` before the QP. The earlier single-step CBF-QP filter, `base_cbf_filter_node`, is kept as a fallback (`enable_base_cbf:=true`).

Key parameters: `d_safe = 0.40 m`, `d_activate = 1.50 m`, `v_max_lin = 0.20 m/s`, `v_max_ang = 0.15 rad/s`

Configuration (on Blackbird): `arna_teleop/config/base_mpc_cbf_params.yaml`

### Layer 3 — Network Watchdog (Legion, 10 Hz)

`network_watchdog_node` maps the network state to a safety mode. It retunes both filters at runtime through `dynamic_reconfigure`, publishes the mode to `/safety_mode` for the GUI badge, and publishes the network authority term `λ_net`.

| Mode | Margin multiplier | Arm horizon | λ_net | Additional effect |
|------|-------------------|-------------|-------|-------------------|
| NOMINAL | ×1.0 | nominal (10–25) | 0.0 | — |
| DEGRADED | ×1.3 | `N_min = 15` | 0.3 | — |
| POOR | ×1.8 | pinned at 25 | 0.6 | — |
| FAILED | ×1.8 | nominal | 1.0 | zero commands flooded to arm and base at 20 Hz |

Transitions to a worse mode take effect immediately; transitions to a better mode require a 3 s dwell to prevent chattering. Baseline margins come from the watchdog's own parameters (not from the filter nodes it writes), and every write is clamped to a per-parameter ceiling, so margins cannot ratchet across restarts.

### Layer 4 — Operator Intent Estimator (Legion, 20 Hz)

`operator_intent_node` measures how strongly each safety filter is overriding the operator. For the arm and the base it computes the normalized intervention

```
d = ‖u_H − u_R‖ / max(‖u_H‖, v_ref)
```

where `u_H` is the operator command and `u_R` is the filter's safe reference. It takes the larger of the two (`D_raw`), smooths it with an EMA (`β = 0.2`), and maps `D ∈ [0, 1]` linearly to `λ_op ∈ [0.1, 0.9]`. The published authority term is

```
λ_combined = λ_net · λ_op
```

so operator disagreement reduces authority only when the network is degraded; on a healthy link (`λ_net = 0`) the operator keeps full authority. Commands below an excitation threshold do not register as disagreement, and `D` resets to zero after 5 s of operator idle.

Configuration: `src/pick_place/config/operator_intent_params.yaml` · runtime schema: `src/pick_place/cfg/OperatorIntent.cfg`

---

## Semi-Autonomous Pick-and-Place

`main.py` orchestrates the pipeline; the other modules are imported libraries, not separate nodes.

1. The operator clicks an object in the arm camera feed (`/pick_click_point`).
2. [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) segments the object at the clicked point (`segmentation.py`).
3. The mask and aligned depth image give a 3-D object point and surface normals (`depth_processing.py`).
4. [Contact-GraspNet](https://github.com/elchun/contact_graspnet_pytorch) proposes ranked 6-DOF grasps (`grasp_net.py`). Upward approaches are rejected and the shortlist is re-ranked toward the most top-down grasp.
5. The best grasp is transformed from the camera frame to `base_link` (`transform.py`).
6. The arm executes a single position-only pick through Kortex actions (`arm_cmd.py`), holding the wrist orientation fixed:
   pre-grasp standoff (0.15 m) → closed-loop re-segmentation → advance to grasp → close gripper → vertical lift (0.08 m) → home.
   Any stage failure aborts and returns the arm to its start pose.

The annotated camera view is published on `/pick_place_cam` (and `/compressed`) for the GUI.

---

## Repository Structure

```
.
├── requirements.txt                  # snapshot of the Legion Python environment (see Dependencies)
└── src/
    ├── ros_kortex/                   # submodule: Kinova Gen3 ROS driver
    ├── kortex_vision/                # submodule: Kinova arm camera driver
    └── pick_place/
        ├── msg/NetworkQuality.msg
        ├── cfg/                      # dynamic_reconfigure schemas (MpcCbfArm, OperatorIntent)
        ├── config/                   # mpc_cbf_params.yaml, operator_intent_params.yaml
        ├── launch/
        │   ├── pick_place.launch     # main Legion launch
        │   ├── start.sh              # sets ROS_IP / ROS_MASTER_URI, then launches pick_place.launch
        │   ├── kortex_driver.launch
        │   └── kinova_vision_rgbd.launch
        ├── scripts/
        │   ├── network_monitor_node.py       # L0 network quality monitor
        │   ├── network_probe_relay.py        # L0 ping → pong relay
        │   ├── mpc_cbf_arm_node.py           # L1 arm MPC-CBF filter
        │   ├── arm_cmd_passthrough_node.py   # L1 bypass (enable_mpc_cbf:=false)
        │   ├── network_watchdog_node.py      # L3 mode coordinator
        │   ├── operator_intent_node.py       # L4 operator intent estimator
        │   ├── main.py                       # pick-and-place orchestrator
        │   ├── segmentation.py, depth_processing.py, grasp_net.py,
        │   │   transform.py, arm_cmd.py, camera.py, proto_pub.py   # pick-and-place libraries
        │   └── export_fastsam_to_onnx.py     # FastSAM → ONNX export utility
        └── pick_place.rviz
```

---

## Dependencies

### Legion (ROS Noetic)

- [ros_kortex](https://github.com/Kinovarobotics/ros_kortex) and [ros_kortex_vision](https://github.com/Kinovarobotics/ros_kortex_vision) (git submodules)
- `ros-noetic-rosbridge-server`
- `ros-noetic-dynamic-reconfigure`
- `python3-catkin-tools`

### Python (3.8)

Key packages: `casadi` (its OSQP plugin is used), `numpy`, `scipy`, `sympy`, `opencv-python`, `ultralytics`, `torch`, and `torchvision`.

`requirements.txt` is a full `pip freeze` of the Legion environment. It includes ROS and Ubuntu system packages that are not on PyPI, so treat it as a version reference rather than an install list.

### Models

- **FastSAM:** `segmentation.py` loads `FastSAM-s.pt` by relative path, so under `roslaunch` it resolves in `~/.ros/`. Ultralytics downloads it on first use if it is missing. Weight files (`*.pt`) are not tracked.
- **Contact-GraspNet:** `grasp_net.py` expects [contact_graspnet_pytorch](https://github.com/elchun/contact_graspnet_pytorch), with its checkpoints, in a directory next to the workspace (`../contact_graspnet_pytorch/checkpoints/contact_graspnet` relative to the workspace root).

### Blackbird (ROS Kinetic)

- OSQP v0.6.3
- Eigen3

---

## Building

**Legion:**
```bash
git clone --recurse-submodules <repo-url> ~/ros
cd ~/ros
source /opt/ros/noetic/setup.bash
rosdep install --from-paths src --ignore-src -r -y
catkin init
catkin build pick_place
source devel/setup.bash
```

> Use `catkin build`, not `catkin_make`; the workspace is configured for catkin tools. If you cloned without `--recurse-submodules`, run `git submodule update --init` first.

**Blackbird** (over SSH):
```bash
cd ~/ros/arna_ws
catkin_make --only-pkg-with-deps arna_teleop
```

---

## Running

**1. Legion ROS stack**

```bash
src/pick_place/launch/start.sh
```

This sets `ROS_IP` and `ROS_MASTER_URI` to `10.0.0.101`, which keeps ROS bound to the ARNA network when a second interface is present, and then runs `roslaunch pick_place pick_place.launch`. The launch file starts the Kinova driver, the arm camera, three rosbridge servers, the pick-and-place node, all Legion safety layers, and RViz. Individual layers can be switched off:

| Argument | Default | Effect when `false` |
|----------|---------|---------------------|
| `enable_mpc_cbf` | `true` | Replace the arm filter with the passthrough relay |
| `enable_network_monitor` | `true` | Skip Layer 0 (monitor and probe relay) |
| `enable_network_watchdog` | `true` | Skip Layer 3 (requires the monitor) |
| `enable_operator_intent` | `true` | Skip Layer 4 |

```bash
roslaunch pick_place pick_place.launch enable_operator_intent:=false
```

**2. Blackbird base stack** (SSH into Blackbird):
```bash
sudo -s
cd ros/arna_ws/src/arna_teleop/src/
./base_interface.sh
```

**3. Web GUI and Cloudflare tunnel** run as systemd services on Legion:
```bash
sudo systemctl restart arna-control
sudo systemctl status cloudflared
```

The GUI is reachable remotely through the tunnel (login required) or locally over the LAN.

> **Security.** rosbridge has no authentication: any client that can reach one of its ports can publish to every topic, including the post-filter arm and base command topics, and call every service. Every tunnel hostname that routes to a rosbridge port must therefore be protected by Cloudflare Access, and the ports must not be reachable from untrusted networks. Keep credentials (SSH passwords, the Kinova login, tunnel tokens) out of this repository.

---

## Web Interface

The GUI (`arna-control`, Next.js; separate repository) connects through the tunnel to three rosbridge WebSocket servers started by `pick_place.launch`:

| Port | Purpose |
|------|---------|
| 9090 | Control plane: arm, base, gripper, pick-and-place, network probes |
| 9091 | Base camera stream |
| 9092 | Arm camera stream |

**Controls:**
- Arm Cartesian velocity (2-D joystick)
- Base translation and rotation (1-D / 2-D joysticks)
- Gripper open / close
- Home action
- Click-to-pick on the arm camera feed
- Safety mode badge (NOMINAL / DEGRADED / POOR / FAILED)
- Live arm and base camera feeds

---

## Tuning

All tuning lives in YAML files; do not edit node source to change tuning values.

| Config file | Controls |
|-------------|----------|
| `src/pick_place/config/mpc_cbf_params.yaml` | Arm: workspace box, speed and acceleration limits, CBF rates, margins, horizon bounds, cost weights |
| `src/pick_place/config/operator_intent_params.yaml` | Intent: EMA rate, normalization floors, λ_op bounds, excitation threshold, idle reset |
| `arna_teleop/config/base_mpc_cbf_params.yaml` (Blackbird) | Base: velocity limits, `d_safe` / `d_activate`, CBF rate, LiDAR margins, horizon |

Parameters can also be changed at runtime through `dynamic_reconfigure` (for example with `rqt_reconfigure`). The network watchdog overwrites margins and horizon bounds automatically as the network state changes.
