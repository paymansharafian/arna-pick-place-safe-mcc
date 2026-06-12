#!/usr/bin/env python3
"""
operator_intent_node — Layer 4 operator-adaptive safety blending.

Algorithm (disagreement metric)
-------------------------------
Layer 4 measures how much the operator is *fighting* the safety filter and uses
it to scale extra caution, gated by network quality. The signal is the
normalized CBF-intervention magnitude, computed PER SUBSYSTEM so arm and base
(different units/scales) are never mixed:

    d_base = ||u_H_base - u_R_base|| / max(||u_H_base||, v_ref_base)
    d_arm  = ||u_H_arm  - u_R_arm||  / max(||u_H_arm||,  v_ref_arm)
    D_raw  = clip( max(d_base, d_arm) + w_dir·relu(-cos_active),  0, 1 )

where u_R is the CBF safe reference and u_H is the raw operator command (same
units). The L2 difference already contains the cosine
(||a-b||² = ||a||²+||b||²-2||a||||b||·cos), so direction is penalized without a
separate term; `w_dir` (default 0) optionally snaps genuine opposition
(cos < 0 of the active subsystem) toward max caution.

  D = 0  → filter passed the command through (operator agrees with safety)
  D = 1  → filter fully overrode / operator reversed (operator fighting)

`max()` selects whichever subsystem is being fought; an idle subsystem
(u_H ≈ u_R ≈ 0) contributes ≈ 0. When the operator is not commanding at all
(‖u_H‖ < excitation_thresh on both), D_raw = 0.

Smoothing + lambda mapping:
    D     ← (1-β)·D + β·D_raw                     (EMA, ema_beta)
    λ_op  = λ_op_min + (λ_op_max - λ_op_min)·D    (linear, monotonic)
    λ_combined = λ_network × λ_op

This replaces the earlier scalar-α / PGD model, whose estimate α = ‖u_H‖/‖u_R‖
saturated to ~1 because u_R is a (1-λ)-scaled copy of u_H — see
docs/superpowers/specs/2026-06-11-operator-intent-redesign-design.md.

Published topics
----------------
/operator_intent/alpha           (std_msgs/Float32)  alias of alignment (back-compat)
/operator_intent/alignment       (std_msgs/Float32)  = 1 - D
/operator_intent/lambda_operator (std_msgs/Float32)
/operator_intent/lambda_combined (std_msgs/Float32)  → consumed by CBF filters

Subscribed topics
-----------------
/mpc_cbf_arm/safe_reference      (geometry_msgs/Twist)       arm u_R
/base_cbf/safe_reference         (geometry_msgs/Twist)       base u_R
/my_gen3/in/cartesian_velocity_desired (kortex TwistCommand) arm u_H
/cmd_vel_desired                 (geometry_msgs/Twist)       base u_H (m/s, rad/s)
/network_watchdog/lambda_network (std_msgs/Float32)          λ_net
"""

import threading
import time

import numpy as np
import rospy

from dynamic_reconfigure.server import Server as DynReconfigureServer
from geometry_msgs.msg          import Twist
from kortex_driver.msg          import TwistCommand
from std_msgs.msg               import Float32

from pick_place.cfg import OperatorIntentConfig


def _subsystem_disagreement(u_H, u_R, v_ref):
    """
    Normalized intervention for one subsystem.

    Returns (d, cos, norm_uH):
      d       = ||u_H - u_R|| / max(||u_H||, v_ref)   (dimensionless, ~[0, 2])
      cos     = cosine(u_H, u_R)                      (1.0 when either is ~zero)
      norm_uH = ||u_H||                               (for the excitation guard)
    """
    norm_uH = float(np.linalg.norm(u_H))
    norm_uR = float(np.linalg.norm(u_R))
    d   = float(np.linalg.norm(u_H - u_R)) / max(norm_uH, v_ref)
    cos = (float(np.dot(u_H, u_R)) / (norm_uH * norm_uR)
           if norm_uH > 1e-9 and norm_uR > 1e-9 else 1.0)
    return d, cos, norm_uH


def main():
    rospy.init_node('operator_intent_node', anonymous=False)

    # ── Dynamic reconfigure ────────────────────────────────────────────────────
    # Seed param server from node params so rqt_reconfigure always shows YAML values.
    for key, default in [
        ('ema_beta',          0.2),
        ('v_ref_base',        0.02),
        ('v_ref_arm',         0.04),
        ('w_dir',             0.0),
        ('lambda_op_min',     0.1),
        ('lambda_op_max',     0.9),
        ('excitation_thresh', 0.01),
        ('reset_on_idle',     True),
        ('idle_timeout_s',    5.0),
    ]:
        if not rospy.has_param('~' + key):
            rospy.set_param('~' + key, default)

    lock = threading.Lock()
    cfg  = {
        'ema_beta':          rospy.get_param('~ema_beta',          0.2),
        'v_ref_base':        rospy.get_param('~v_ref_base',        0.02),
        'v_ref_arm':         rospy.get_param('~v_ref_arm',         0.04),
        'w_dir':             rospy.get_param('~w_dir',             0.0),
        'lambda_op_min':     rospy.get_param('~lambda_op_min',     0.1),
        'lambda_op_max':     rospy.get_param('~lambda_op_max',     0.9),
        'excitation_thresh': rospy.get_param('~excitation_thresh', 0.01),
        'reset_on_idle':     rospy.get_param('~reset_on_idle',     True),
        'idle_timeout_s':    rospy.get_param('~idle_timeout_s',    5.0),
    }

    def dynrec_cb(new_cfg, _level):
        with lock:
            cfg.update({
                'ema_beta':          new_cfg.ema_beta,
                'v_ref_base':        new_cfg.v_ref_base,
                'v_ref_arm':         new_cfg.v_ref_arm,
                'w_dir':             new_cfg.w_dir,
                'lambda_op_min':     new_cfg.lambda_op_min,
                'lambda_op_max':     new_cfg.lambda_op_max,
                'excitation_thresh': new_cfg.excitation_thresh,
                'reset_on_idle':     new_cfg.reset_on_idle,
                'idle_timeout_s':    new_cfg.idle_timeout_s,
            })
        return new_cfg

    _dr = DynReconfigureServer(OperatorIntentConfig, dynrec_cb)

    # ── Shared state ───────────────────────────────────────────────────────────
    D           = 0.0          # smoothed disagreement (0 = aligned, 1 = fighting)
    lambda_net  = 0.0          # from network_watchdog_node

    # Safe references (u_R)
    u_R_arm  = np.zeros(6)    # from /mpc_cbf_arm/safe_reference (6-DOF twist)
    u_R_base = np.zeros(3)    # from /base_cbf/safe_reference   (vx, vy, wz)

    # Observed human commands (u_H)
    u_H_arm  = np.zeros(6)    # from /my_gen3/in/cartesian_velocity_desired
    u_H_base = np.zeros(3)    # from /cmd_vel_desired

    last_cmd_time = time.time()   # for idle reset

    # ── Subscribers ───────────────────────────────────────────────────────────
    def arm_safe_ref_cb(msg):
        nonlocal u_R_arm
        with lock:
            u_R_arm = np.array([
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z,
            ])

    def base_safe_ref_cb(msg):
        nonlocal u_R_base
        with lock:
            u_R_base = np.array([msg.linear.x, msg.linear.y, msg.angular.z])

    def arm_desired_cb(msg):
        nonlocal u_H_arm, last_cmd_time
        with lock:
            u_H_arm = np.array([
                msg.twist.linear_x, msg.twist.linear_y, msg.twist.linear_z,
                msg.twist.angular_x, msg.twist.angular_y, msg.twist.angular_z,
            ])
            last_cmd_time = time.time()

    def base_desired_cb(msg):
        """
        /cmd_vel_desired is geometry_msgs/Twist — actual velocity (m/s, rad/s),
        the same units as /base_cbf/safe_reference (no scale mismatch).
        """
        nonlocal u_H_base, last_cmd_time
        with lock:
            u_H_base = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
            last_cmd_time = time.time()

    def lambda_net_cb(msg):
        nonlocal lambda_net
        with lock:
            lambda_net = float(msg.data)

    rospy.Subscriber('/mpc_cbf_arm/safe_reference',
                     Twist, arm_safe_ref_cb, queue_size=1)
    rospy.Subscriber('/base_cbf/safe_reference',
                     Twist, base_safe_ref_cb, queue_size=1)
    rospy.Subscriber('/my_gen3/in/cartesian_velocity_desired',
                     TwistCommand, arm_desired_cb, queue_size=1)
    rospy.Subscriber('/cmd_vel_desired',
                     Twist, base_desired_cb, queue_size=1)
    rospy.Subscriber('/network_watchdog/lambda_network',
                     Float32, lambda_net_cb, queue_size=1)

    # ── Publishers ─────────────────────────────────────────────────────────────
    pub_alpha     = rospy.Publisher('/operator_intent/alpha',           Float32, queue_size=1)
    pub_align     = rospy.Publisher('/operator_intent/alignment',       Float32, queue_size=1)
    pub_lambda_op = rospy.Publisher('/operator_intent/lambda_operator', Float32, queue_size=1)
    pub_lambda_c  = rospy.Publisher('/operator_intent/lambda_combined', Float32, queue_size=1)

    # ── 20 Hz control loop ─────────────────────────────────────────────────────
    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        rate.sleep()

        with lock:
            _u_R_arm  = u_R_arm.copy()
            _u_R_base = u_R_base.copy()
            _u_H_arm  = u_H_arm.copy()
            _u_H_base = u_H_base.copy()
            _lnet     = lambda_net
            _cfg      = dict(cfg)
            _last_cmd = last_cmd_time

        # Idle reset: no command for idle_timeout_s → fully aligned (no caution).
        idle = (time.time() - _last_cmd) > _cfg['idle_timeout_s']
        if _cfg['reset_on_idle'] and idle:
            D = 0.0

        # Per-subsystem normalized intervention.
        d_base, cos_base, nH_base = _subsystem_disagreement(
            _u_H_base, _u_R_base, _cfg['v_ref_base'])
        d_arm,  cos_arm,  nH_arm  = _subsystem_disagreement(
            _u_H_arm,  _u_R_arm,  _cfg['v_ref_arm'])

        # Excitation guard: no operator command → no disagreement.
        if max(nH_base, nH_arm) < _cfg['excitation_thresh']:
            D_raw = 0.0
        else:
            if d_base >= d_arm:
                d_max, cos_active = d_base, cos_base
            else:
                d_max, cos_active = d_arm, cos_arm
            D_raw = d_max + _cfg['w_dir'] * max(0.0, -cos_active)
            D_raw = float(np.clip(D_raw, 0.0, 1.0))

        # EMA smoothing.
        beta = _cfg['ema_beta']
        D = (1.0 - beta) * D + beta * D_raw

        # λ_operator: linear map, more disagreement → more caution.
        l_min     = _cfg['lambda_op_min']
        l_max     = _cfg['lambda_op_max']
        lambda_op = float(np.clip(l_min + (l_max - l_min) * D, l_min, l_max))

        alignment       = 1.0 - D
        lambda_combined = _lnet * lambda_op

        # Publish (alpha kept as an alias of alignment for back-compat).
        pub_alpha.publish(Float32(data=alignment))
        pub_align.publish(Float32(data=alignment))
        pub_lambda_op.publish(Float32(data=lambda_op))
        pub_lambda_c.publish(Float32(data=lambda_combined))

    rospy.loginfo('[operator_intent] Shutting down.')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
