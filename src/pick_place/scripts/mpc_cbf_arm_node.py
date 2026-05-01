#!/usr/bin/env python3
"""
mpc_cbf_arm_node — Phase 1 full MPC-CBF arm safety filter (CasADi/qpOASES).

System model
------------
First-order Cartesian integrator with first-order velocity tracking lag.

State (12D):   x = [p_x, p_y, p_z, θ_x, θ_y, θ_z,
                    v_x, v_y, v_z, ω_x, ω_y, ω_z]
Input (6D):    u = [u_vx, u_vy, u_vz, u_ωx, u_ωy, u_ωz]

Discrete-time dynamics (dt = 10 ms):
    p_{k+1} = p_k + dt * v_k
    v_{k+1} = (1 - dt/τ) * v_k + (dt/τ) * u_k   (first-order lag, τ = 0.05 s)
    (same for orientation / angular velocity)

MPC formulation (horizon N steps)
-----------------------------------
min  Σ_{k=0}^{N-1} [ (x_k-x_ref_k)' Q (x_k-x_ref_k) + u_k' R u_k
                    + jerk_cost(u_k, u_{k-1}, u_{k-2}) ]
   + (x_N-x_ref_N)' Q_T (x_N-x_ref_N)
   + Σ slack penalty terms
s.t.
  x_{k+1} = A x_k + B u_k                      (dynamics)
  CBF-workspace:  h_ws_i(k+1) >= (1-γ_ws)*h_ws_i(k) - s_ws_i   ∀i, ∀k
  CBF-speed:      h_sp(k+1)   >= (1-γ_sp)*h_sp(k)   - s_sp      ∀k
  CBF-accel:      h_ac(k)     >= (1-γ_ac)*h_ac(k-1) - s_ac      ∀k  (Constraint A)
  s_ws_i ≥ 0,  s_sp ≥ 0,  s_ac ≥ 0             (slack non-negativity)
  -v_max ≤ u_k ≤ v_max                          (hard velocity clamp)

Safety functions
----------------
h_ws_i(x)  — distance to workspace face i (6 constraints, affine in p):
    h_ws_0..2 = p - p_min_eff     (lower faces)
    h_ws_3..5 = p_max_eff - p     (upper faces)
    p_min_eff = p_min + ε_ws,  p_max_eff = p_max - ε_ws
    ε_ws = epsilon_base_workspace + k_epsilon_workspace * rtt_std_ms

h_sp(x)    — speed limit (linearised around current velocity for QP):
    h_sp = (v_max_ee_eff)^2 - ||v_lin||^2
    v_max_ee_eff = v_max_ee - ε_sp
    ε_sp = epsilon_base_speed + k_epsilon_speed * rtt_std_ms

h_ac(u_k)  — acceleration CBF (Constraint A, linearised):
    h_ac = (arm_accel_max * dt)^2 - ||u_k_lin - v_ee_cur||^2

Jerk cost (Constraint B)
------------------------
    J_jerk_k = jerk_weight_xy * ( (u_kx - 2*u_{k-1,x} + u_{k-2,x})^2
                                 + (u_ky - 2*u_{k-1,y} + u_{k-2,y})^2 )
             + jerk_weight_z  *   (u_kz - 2*u_{k-1,z} + u_{k-2,z})^2
    u_{-1} = u_prev,  u_{-2} = u_prev2  (ring buffer updated each cycle)

Reference trajectory
--------------------
Forward-integrate the 5 Hz LPF-filtered operator command for N+1 steps
using the same discrete-time dynamics.

Adaptive horizon
----------------
N = clip(ceil(delta_max_ms / dt_ms), N_min, N_max)
Pinned to N_max when network_state == 'FAILED'.

Soft infeasibility
------------------
Slack variables on all CBF constraints.  Large quadratic penalties keep
slacks near zero under normal operation.  ROS warnings emitted when any
slack exceeds 1e-4.

Timer
-----
rospy.Timer at 100 Hz (period = dt).  Solver overruns are detected and
the cycle is skipped to prevent command queue growth.
"""

import math
import threading
import time
import numpy as np
import casadi as ca
import rospy
from dynamic_reconfigure.server import Server as DynReconfigureServer
from kortex_driver.msg import TwistCommand, BaseCyclic_Feedback
from pick_place.msg import NetworkQuality
from pick_place.cfg import MpcCbfArmConfig
from std_msgs.msg import Bool


# ─────────────────────────────────────────────────────────────────────────────
# Discrete-time dynamics matrices
# ─────────────────────────────────────────────────────────────────────────────
def build_dynamics(dt: float, tau: float = 0.05):
    """
    Return (A_d, B_d) numpy matrices for the 12-state first-order lag model.
    State: [p(3), theta(3), v_lin(3), v_ang(3)]
    Input: [u_lin(3), u_ang(3)]
    """
    a_lag = 1.0 - dt / tau
    b_lag = dt / tau

    A = np.zeros((12, 12))
    B = np.zeros((12, 6))

    A[0:3,  0:3]  = np.eye(3)            # p ← p
    A[0:3,  6:9]  = dt * np.eye(3)       # p ← v_lin
    A[3:6,  3:6]  = np.eye(3)            # theta ← theta
    A[3:6,  9:12] = dt * np.eye(3)       # theta ← v_ang
    A[6:9,  6:9]  = a_lag * np.eye(3)    # v_lin ← v_lin
    A[9:12, 9:12] = a_lag * np.eye(3)    # v_ang ← v_ang
    B[6:9,  0:3]  = b_lag * np.eye(3)    # v_lin ← u_lin
    B[9:12, 3:6]  = b_lag * np.eye(3)    # v_ang ← u_ang

    return A, B


# ─────────────────────────────────────────────────────────────────────────────
# MPC-CBF QP solver (CasADi / qpOASES)
# ─────────────────────────────────────────────────────────────────────────────
class MpcCbfSolver:
    """
    Parametric QP built once per horizon N.  All time-varying quantities
    are passed through the parameter vector θ at each solve call.

    Parameter vector θ layout (total length = 12 + 12*(N+1) + 3+3+1+3+6+6+1+1+1+1):
      x0         (12)        — current state
      x_ref_flat (12*(N+1))  — reference trajectory (N+1 states)
      p_min_eff  (3)         — effective workspace lower bound
      p_max_eff  (3)         — effective workspace upper bound
      v_max_eff  (1)         — effective speed limit
      v_ee_cur   (3)         — current linear EE velocity (for accel CBF)
      u_prev     (6)         — u_{k-1} (jerk + accel CBF at k=0)
      u_prev2    (6)         — u_{k-2} (jerk at k=0)
      gamma_acc  (1)         — accel CBF decay gain (dynamic_reconfigure)
      amax_dt    (1)         — arm_accel_max * dt
      jw_xy      (1)         — jerk weight xy
      jw_z       (1)         — jerk weight z

    Decision variables w layout:
      U_flat  (N*6)   — velocity commands u_0 … u_{N-1}
      sl_ws   (6)     — workspace CBF slack (per face)
      sl_sp   (1)     — speed CBF slack
      sl_ac   (1)     — accel CBF slack
    """

    def __init__(self, N, dt,
                 Q_pos, Q_vel, R_lin, R_ang, Q_T_scale,
                 gamma_ws, gamma_sp,
                 v_max_lin, v_max_ang,
                 slack_pen_ws, slack_pen_sp, slack_pen_ac):

        self.N  = N
        self.dt = dt
        nx, nu  = 12, 6

        A_np, B_np = build_dynamics(dt)
        A = ca.DM(A_np)
        B = ca.DM(B_np)

        # Decision variable dimension
        n_u   = N * nu
        n_sl  = 6 + 1 + 1       # ws, speed, accel
        n_dec = n_u + n_sl

        w      = ca.MX.sym('w', n_dec)
        n_p    = 12 + 12*(N+1) + 3 + 3 + 1 + 3 + 6 + 6 + 1 + 1 + 1 + 1
        theta  = ca.MX.sym('theta', n_p)

        # ── Unpack parameters ─────────────────────────────────────────────
        idx = 0
        def take(n):
            nonlocal idx
            v = theta[idx:idx+n]; idx += n; return v

        x0_p      = take(12)
        xref_p    = take(12*(N+1))
        pmin_p    = take(3)
        pmax_p    = take(3)
        vmax_sp   = take(1)[0]
        vee_cur   = take(3)
        u_prev_p  = take(6)
        u_prev2_p = take(6)
        g_acc     = take(1)[0]
        amax_dt   = take(1)[0]
        jw_xy     = take(1)[0]
        jw_z      = take(1)[0]

        # ── Unpack decision variables ─────────────────────────────────────
        U_list = [w[k*nu:(k+1)*nu] for k in range(N)]
        sl_ws  = w[n_u:n_u+6]
        sl_sp  = w[n_u+6]
        sl_ac  = w[n_u+7]

        # ── Cost and constraint assembly ──────────────────────────────────
        Q_d  = ca.diag(ca.vertcat(*Q_pos, *Q_pos, *Q_vel, *Q_vel))
        R_d  = ca.diag(ca.vertcat(*R_lin, *R_ang))
        QT_d = Q_T_scale * Q_d

        cost   = ca.MX(0)
        g_list, lbg, ubg = [], [], []

        x       = x0_p
        u_km1   = u_prev_p
        u_km2   = u_prev2_p

        # ── Fixed linearisation points (parameters, constant w.r.t. w) ────────
        # Speed CBF  h_sp(v) = vmax^2 - ||v||^2  is non-linear in v.
        # We linearise it once at the current measured velocity vee_cur so that
        # h_sp constraints remain LINEAR in w for all horizon steps, satisfying
        # the ca.qpsol (linear Jacobian) requirement.
        # h_sp_lin(v) = (vmax^2 + ||vee_cur||^2) - 2*vee_cur'*v
        v_lin_pt   = vee_cur
        v0_sq      = ca.dot(v_lin_pt, v_lin_pt)
        h_sp_const = vmax_sp**2 + v0_sq

        # Accel CBF  h_ac(u) = amax_dt^2 - ||u - vee_cur||^2  similarly non-linear.
        # Linearise at d_lin_pt = u_prev - vee_cur (also a parameter).
        # h_ac_lin(u) = (amax_dt^2 + ||d_lin_pt||^2) - 2*d_lin_pt'*(u[0:3] - vee_cur)
        d_lin_pt   = u_prev_p[0:3] - vee_cur
        d0_sq      = ca.dot(d_lin_pt, d_lin_pt)
        h_ac_const = amax_dt**2 + d0_sq

        # ── Initial CBF values at k=0 ─────────────────────────────────────
        h_ws_k  = ca.vertcat(x[0:3] - pmin_p, pmax_p - x[0:3])
        h_sp_k  = h_sp_const - 2.0 * ca.dot(v_lin_pt, x[6:9])
        h_ac_k  = h_ac_const - 2.0 * ca.dot(d_lin_pt, u_km1[0:3] - vee_cur)

        for k in range(N):
            uk        = U_list[k]
            x_ref_k   = xref_p[k*12:(k+1)*12]

            # ── Stage cost: tracking + control effort ─────────────────────
            dx   = x - x_ref_k
            cost += ca.bilin(Q_d, dx, dx) + ca.bilin(R_d, uk, uk)

            # ── Jerk penalty (Constraint B) ───────────────────────────────
            jerk = uk[0:3] - 2*u_km1[0:3] + u_km2[0:3]
            cost += jw_xy * (jerk[0]**2 + jerk[1]**2)
            cost += jw_z  *  jerk[2]**2

            # ── Dynamics ─────────────────────────────────────────────────
            x_next = ca.mtimes(A, x) + ca.mtimes(B, uk)

            # ── CBF-workspace (affine in p — no linearisation needed) ─────
            h_ws_next = ca.vertcat(x_next[0:3] - pmin_p,
                                   pmax_p - x_next[0:3])
            for i in range(6):
                g_list.append(h_ws_next[i] - (1.0 - gamma_ws)*h_ws_k[i] + sl_ws[i])
                lbg.append(0.0); ubg.append(float('inf'))

            # ── CBF-speed (linearised at vee_cur for ALL horizon steps) ───
            # h_sp_lin(v) = h_sp_const - 2*v_lin_pt'*v   → linear in x_next[6:9]
            h_sp_next = h_sp_const - 2.0 * ca.dot(v_lin_pt, x_next[6:9])
            g_list.append(h_sp_next - (1.0 - gamma_sp)*h_sp_k + sl_sp)
            lbg.append(0.0); ubg.append(float('inf'))

            # ── CBF-accel (Constraint A, linearised at d_lin_pt for all steps)
            # h_ac_lin(u) = h_ac_const - 2*d_lin_pt'*(u[0:3] - vee_cur)  → linear in uk
            h_ac_next = h_ac_const - 2.0 * ca.dot(d_lin_pt, uk[0:3] - vee_cur)
            g_list.append(h_ac_next - (1.0 - g_acc)*h_ac_k + sl_ac)
            lbg.append(0.0); ubg.append(float('inf'))

            # ── Hard velocity box ─────────────────────────────────────────
            for j in range(3):
                g_list.append( uk[j] + v_max_lin); lbg.append(0.0); ubg.append(float('inf'))
                g_list.append(-uk[j] + v_max_lin); lbg.append(0.0); ubg.append(float('inf'))
            for j in range(3, 6):
                g_list.append( uk[j] + v_max_ang); lbg.append(0.0); ubg.append(float('inf'))
                g_list.append(-uk[j] + v_max_ang); lbg.append(0.0); ubg.append(float('inf'))

            # Advance
            h_ws_k = h_ws_next
            h_sp_k = h_sp_next
            h_ac_k = h_ac_next
            x      = x_next
            u_km2  = u_km1
            u_km1  = uk

        # ── Terminal cost ─────────────────────────────────────────────────
        dx_N  = x - xref_p[N*12:(N+1)*12]
        cost += ca.bilin(QT_d, dx_N, dx_N)

        # ── Slack penalties ───────────────────────────────────────────────
        for i in range(6):
            cost += slack_pen_ws * sl_ws[i]**2
        cost += slack_pen_sp * sl_sp**2
        cost += slack_pen_ac * sl_ac**2

        # ── Slack non-negativity ──────────────────────────────────────────
        for i in range(6):
            g_list.append(sl_ws[i]); lbg.append(0.0); ubg.append(float('inf'))
        g_list.append(sl_sp); lbg.append(0.0); ubg.append(float('inf'))
        g_list.append(sl_ac); lbg.append(0.0); ubg.append(float('inf'))

        # ── Build solver ──────────────────────────────────────────────────
        g_expr = ca.vertcat(*g_list)
        nlp    = {'x': w, 'f': cost, 'g': g_expr, 'p': theta}
        # 'qpoases' is selected as 2nd arg to ca.qpsol — do NOT repeat it in opts.
        # qpOASES-specific options (printLevel, nWSR, etc.) go at the top level.
        opts = {'print_time': False}
        self._solver = ca.qpsol('mpc_cbf_N%d' % N, 'qpoases', nlp, opts)
        self._n_u    = n_u
        self._lbg    = lbg
        self._ubg    = ubg
        self._n_p    = n_p

        # Warm-start
        self._w0   = np.zeros(n_dec)
        self._lam0 = np.zeros(len(lbg))

    def solve(self, theta_val: np.ndarray):
        """
        Returns (u0, slacks_dict, solve_ok).
        u0: 6D numpy array (safe command at k=0).
        """
        try:
            res    = self._solver(
                x0=self._w0, lam_g0=self._lam0,
                p=theta_val, lbg=self._lbg, ubg=self._ubg)
            w_opt  = np.array(res['x']).flatten()
            self._w0   = w_opt
            self._lam0 = np.array(res['lam_g']).flatten()

            u0    = w_opt[0:6]
            sl_ws = w_opt[self._n_u:self._n_u+6]
            sl_sp = float(w_opt[self._n_u+6])
            sl_ac = float(w_opt[self._n_u+7])
            return u0, {'workspace': sl_ws, 'speed': sl_sp, 'accel': sl_ac}, True

        except Exception as exc:
            rospy.logerr('[mpc_cbf_arm] QP solver exception: %s', exc)
            return None, {}, False


# ─────────────────────────────────────────────────────────────────────────────
# ROS node entry point
# ─────────────────────────────────────────────────────────────────────────────
def _publish_twist(pub, u6):
    msg = TwistCommand()
    msg.reference_frame = 0
    msg.duration        = 0
    msg.twist.linear_x  = float(u6[0])
    msg.twist.linear_y  = float(u6[1])
    msg.twist.linear_z  = float(u6[2])
    msg.twist.angular_x = float(u6[3])
    msg.twist.angular_y = float(u6[4])
    msg.twist.angular_z = float(u6[5])
    pub.publish(msg)


def main():
    rospy.init_node('mpc_cbf_arm_node', anonymous=False)

    # ── Load parameters ───────────────────────────────────────────────────
    def gp(name, default):
        return rospy.get_param('~' + name, default)

    p_min_nom = np.array([gp('workspace_x_min', -0.46),
                          gp('workspace_y_min', -0.37),
                          gp('workspace_z_min',  0.01)])
    p_max_nom = np.array([gp('workspace_x_max',  0.9),
                          gp('workspace_y_max',  0.9),
                          gp('workspace_z_max',  0.94)])

    v_max_ee_nom  = float(gp('v_max_ee_ms',             0.4))
    dt            = float(gp('dt_s',                     0.01))
    N_min         = int(  gp('horizon_min',              10))
    N_max         = int(  gp('horizon_max',              25))
    gamma_ws      = float(gp('cbf_gamma_workspace',      1.0))
    gamma_sp      = float(gp('cbf_gamma_speed',          1.0))
    v_max_lin     = float(gp('v_max_ee_ms',              0.4))
    v_max_ang     = float(gp('v_max_angular',            1.0))
    lpf_alpha     = float(gp('lpf_alpha',                0.73))
    Q_pos         = list( gp('Q_pos',  [100., 100., 100.]))
    Q_vel         = list( gp('Q_vel',  [10.,  10.,  10.]))
    R_lin         = list( gp('R_lin',  [1.,   1.,   1.]))
    R_ang         = list( gp('R_ang',  [1.,   1.,   1.]))
    Q_T_scale     = float(gp('Q_terminal_scale',         10.0))
    slack_pen_ws  = float(gp('slack_penalty_workspace',  1e5))
    slack_pen_sp  = float(gp('slack_penalty_speed',      1e5))
    slack_pen_ac  = float(gp('slack_penalty_accel',      1e4))

    arm_accel_max0 = float(gp('jerk_and_accel/arm_accel_max',  2.0))
    gamma_acc0     = float(gp('jerk_and_accel/gamma_acc',       1.0))
    jerk_xy0       = float(gp('jerk_and_accel/jerk_weight_xy',  0.5))
    jerk_z0        = float(gp('jerk_and_accel/jerk_weight_z',   0.5))
    eps_base_ws0   = float(gp('epsilon_base_workspace',  0.02))
    k_eps_ws0      = float(gp('k_epsilon_workspace',     0.001))
    eps_base_sp0   = float(gp('epsilon_base_speed',      0.02))
    k_eps_sp0      = float(gp('k_epsilon_speed',         0.0005))

    dt_ms = dt * 1000.0

    # ── Shared mutable state ──────────────────────────────────────────────
    lock = threading.Lock()

    arm_pos          = np.zeros(3)
    arm_vel          = np.zeros(6)
    arm_state_valid  = False
    u_desired_lpf    = np.zeros(6)
    last_desired_time = time.time()   # watchdog: zero cmd if no msg for 300 ms
    rtt_std_ms       = 0.0
    N_current        = N_min
    pick_running     = False
    u_prev           = np.zeros(6)
    u_prev2          = np.zeros(6)

    # Dynamic reconfigure state (initialised from param server)
    dr = {
        'eps_base_ws':    eps_base_ws0,
        'k_eps_ws':       k_eps_ws0,
        'eps_base_sp':    eps_base_sp0,
        'k_eps_sp':       k_eps_sp0,
        'arm_accel_max':  arm_accel_max0,
        'gamma_acc':      gamma_acc0,
        'jerk_weight_xy': jerk_xy0,
        'jerk_weight_z':  jerk_z0,
    }

    # ── Build initial solver ──────────────────────────────────────────────
    rospy.loginfo('[mpc_cbf_arm] Building CasADi QP for N=%d, dt=%.3fs ...', N_min, dt)
    solver_cache = {}
    def get_solver(N):
        if N not in solver_cache:
            rospy.loginfo('[mpc_cbf_arm] Building QP for N=%d ...', N)
            solver_cache[N] = MpcCbfSolver(
                N=N, dt=dt,
                Q_pos=Q_pos, Q_vel=Q_vel, R_lin=R_lin, R_ang=R_ang,
                Q_T_scale=Q_T_scale,
                gamma_ws=gamma_ws, gamma_sp=gamma_sp,
                v_max_lin=v_max_lin, v_max_ang=v_max_ang,
                slack_pen_ws=slack_pen_ws,
                slack_pen_sp=slack_pen_sp,
                slack_pen_ac=slack_pen_ac,
            )
            rospy.loginfo('[mpc_cbf_arm] QP N=%d ready.', N)
        return solver_cache[N]

    get_solver(N_min)   # build N_min eagerly at startup
    rospy.loginfo('[mpc_cbf_arm] Startup QP build complete.')

    # ── Publisher ─────────────────────────────────────────────────────────
    safe_pub = rospy.Publisher(
        '/my_gen3/in/cartesian_velocity', TwistCommand, queue_size=1)

    # ── Dynamic reconfigure server ────────────────────────────────────────
    # Seed the parameter server with the values from the YAML file so that
    # dynamic_reconfigure always starts from the YAML defaults, not from
    # whatever stale values a previous session left on the parameter server.
    rospy.set_param('~epsilon_base_workspace', eps_base_ws0)
    rospy.set_param('~k_epsilon_workspace',    k_eps_ws0)
    rospy.set_param('~epsilon_base_speed',     eps_base_sp0)
    rospy.set_param('~k_epsilon_speed',        k_eps_sp0)
    rospy.set_param('~arm_accel_max',          arm_accel_max0)
    rospy.set_param('~gamma_acc',              gamma_acc0)
    rospy.set_param('~jerk_weight_xy',         jerk_xy0)
    rospy.set_param('~jerk_weight_z',          jerk_z0)

    def dynreconf_cb(config, level):
        with lock:
            dr['eps_base_ws']    = config.epsilon_base_workspace
            dr['k_eps_ws']       = config.k_epsilon_workspace
            dr['eps_base_sp']    = config.epsilon_base_speed
            dr['k_eps_sp']       = config.k_epsilon_speed
            dr['arm_accel_max']  = config.arm_accel_max
            dr['gamma_acc']      = config.gamma_acc
            dr['jerk_weight_xy'] = config.jerk_weight_xy
            dr['jerk_weight_z']  = config.jerk_weight_z
        return config

    DynReconfigureServer(MpcCbfArmConfig, dynreconf_cb)

    # ── ROS subscribers ───────────────────────────────────────────────────
    def feedback_cb(msg):
        nonlocal arm_pos, arm_vel, arm_state_valid
        b = msg.base
        with lock:
            arm_pos[:]       = [b.tool_pose_x, b.tool_pose_y, b.tool_pose_z]
            arm_vel[:]       = [b.tool_twist_linear_x,  b.tool_twist_linear_y,
                                b.tool_twist_linear_z,  b.tool_twist_angular_x,
                                b.tool_twist_angular_y, b.tool_twist_angular_z]
            arm_state_valid  = True

    def quality_cb(msg):
        nonlocal rtt_std_ms, N_current
        with lock:
            rtt_std_ms = msg.rtt_std_ms
            if msg.network_state == 'FAILED':
                N_current = N_max
            else:
                N_current = int(np.clip(
                    math.ceil(msg.delta_max_ms / dt_ms),
                    N_min, N_max))

    def desired_cb(msg):
        nonlocal u_desired_lpf, last_desired_time
        raw = np.array([
            msg.twist.linear_x,  msg.twist.linear_y,  msg.twist.linear_z,
            msg.twist.angular_x, msg.twist.angular_y, msg.twist.angular_z,
        ])
        with lock:
            u_desired_lpf = lpf_alpha * u_desired_lpf + (1.0 - lpf_alpha) * raw
            last_desired_time = time.time()

    def pick_cb(msg):
        nonlocal pick_running
        with lock:
            pick_running = msg.data

    rospy.Subscriber('/my_gen3/base_feedback',
                     BaseCyclic_Feedback, feedback_cb, queue_size=1)
    rospy.Subscriber('/network_quality',
                     NetworkQuality, quality_cb, queue_size=1)
    rospy.Subscriber('/my_gen3/in/cartesian_velocity_desired',
                     TwistCommand, desired_cb, queue_size=1)
    rospy.Subscriber('/pick_running', Bool, pick_cb, queue_size=1)

    # ── 100 Hz control timer ──────────────────────────────────────────────
    timer_busy = threading.Event()
    A_np, B_np = build_dynamics(dt)

    def control_cb(event):
        nonlocal u_prev, u_prev2

        if timer_busy.is_set():
            rospy.logwarn_throttle(1.0, '[mpc_cbf_arm] Solver overran — skipping cycle')
            return
        timer_busy.set()

        try:
            with lock:
                _pick   = pick_running
                _valid  = arm_state_valid
                _pos    = arm_pos.copy()
                _vel    = arm_vel.copy()
                _u_des  = u_desired_lpf.copy()
                _cmd_age = time.time() - last_desired_time
                _rtt_s  = rtt_std_ms
                _N      = N_current
                _d      = dr.copy()

            # Command-age watchdog: if no desired message for >300 ms, treat as zero.
            # This stops the arm when the joystick is released and the frontend
            # has sent its last non-zero command.
            CMD_TIMEOUT = 0.30
            if _cmd_age > CMD_TIMEOUT:
                _u_des = np.zeros(6)

            if _pick:
                return

            if not _valid:
                rospy.logwarn_throttle(5.0,
                    '[mpc_cbf_arm] No base_feedback yet — passing through')
                _publish_twist(safe_pub, _u_des)
                return

            # Adaptive margins
            eps_ws  = _d['eps_base_ws'] + _d['k_eps_ws'] * _rtt_s
            eps_sp  = _d['eps_base_sp'] + _d['k_eps_sp'] * _rtt_s
            p_min_eff = p_min_nom + eps_ws
            p_max_eff = p_max_nom - eps_ws
            v_max_eff = max(0.01, v_max_ee_nom - eps_sp)

            cur_solver = get_solver(_N)

            # Current state vector
            x0 = np.concatenate([_pos, np.zeros(3), _vel])

            # Reference trajectory: forward-integrate LPF command
            xr = x0.copy()
            xref_list = []
            for _ in range(_N + 1):
                xref_list.append(xr)
                xr = A_np @ xr + B_np @ _u_des
            x_ref_flat = np.concatenate(xref_list)

            # Parameter vector θ
            theta_val = np.concatenate([
                x0,
                x_ref_flat,
                p_min_eff,
                p_max_eff,
                [v_max_eff],
                _vel[0:3],                        # v_ee_cur
                u_prev,
                u_prev2,
                [_d['gamma_acc']],
                [_d['arm_accel_max'] * dt],       # amax_dt
                [_d['jerk_weight_xy']],
                [_d['jerk_weight_z']],
            ])

            u0, slacks, ok = cur_solver.solve(theta_val)

            if not ok:
                rospy.logwarn('[mpc_cbf_arm] Solver failed — clamped passthrough')
                u_safe = np.clip(_u_des[0:3], -v_max_lin, v_max_lin)
                u_ang  = np.clip(_u_des[3:6], -v_max_ang, v_max_ang)
                _publish_twist(safe_pub, np.concatenate([u_safe, u_ang]))
                return

            # Warn on active slack
            face_labels = ['x-', 'y-', 'z-', 'x+', 'y+', 'z+']
            for i, s in enumerate(slacks.get('workspace', [])):
                if s > 1e-4:
                    rospy.logwarn_throttle(0.5,
                        '[mpc_cbf_arm] Workspace CBF slack[%d/%s] = %.4f',
                        i, face_labels[i], s)
            sp_sl = slacks.get('speed', 0.0)
            ac_sl = slacks.get('accel', 0.0)
            if sp_sl > 1e-4:
                rospy.logwarn_throttle(0.5,
                    '[mpc_cbf_arm] Speed CBF slack = %.4f', sp_sl)
            if ac_sl > 1e-4:
                rospy.logwarn_throttle(0.5,
                    '[mpc_cbf_arm] Accel CBF slack = %.4f', ac_sl)

            _publish_twist(safe_pub, u0)

            # Update jerk ring buffer
            u_prev2 = u_prev.copy()
            u_prev  = u0.copy()

        finally:
            timer_busy.clear()

    rospy.Timer(rospy.Duration(dt), control_cb)

    rospy.loginfo(
        '[mpc_cbf_arm] Ready | N=[%d,%d] | dt=%.3fs | '
        'workspace x=[%.2f,%.2f] y=[%.2f,%.2f] z=[%.2f,%.2f] | v_max=%.2f',
        N_min, N_max, dt,
        p_min_nom[0], p_max_nom[0],
        p_min_nom[1], p_max_nom[1],
        p_min_nom[2], p_max_nom[2],
        v_max_ee_nom)
    rospy.spin()


if __name__ == '__main__':
    main()
