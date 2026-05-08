#!/usr/bin/env python3
"""
operator_intent_node — Phase 4 operator-adaptive safety blending.

Algorithm
---------
A single scalar α ∈ (0,1) models the operator's command as a linear
fraction of the robot's safe reference:

    u_H*(α)  =  α · u_R

where  u_R  is the safe reference published by the Phase 1/2 CBF-QP
filters, and  u_H_obs  is the raw joystick command.

Projected Gradient Descent (PGD) update (20 Hz):
    e        = α·u_R − u_H_obs
    ∇J       = dot(e, u_R)          (scalar, chain-rule of ||e||²/2)
    α        ← clip(α − μ·γ·∇J,  α_floor, 1−α_floor)

Excitation guard: skip update when ||u_R|| < excitation_thresh to
avoid gradient collapse.

λ_operator computation:
    alignment = α
    λ_op      = 1 − sigmoid(k · (alignment − 0.5))
    λ_op      = clip(λ_op, λ_op_min, λ_op_max)

λ_combined = λ_network × λ_op

Published topics
----------------
/operator_intent/alpha           (std_msgs/Float32)  raw PGD estimate
/operator_intent/alignment       (std_msgs/Float32)  = alpha
/operator_intent/lambda_operator (std_msgs/Float32)
/operator_intent/lambda_combined (std_msgs/Float32)  → consumed by CBF filters

Subscribed topics
-----------------
/mpc_cbf_arm/safe_reference      (geometry_msgs/Twist)       arm u_R
/base_cbf/safe_reference         (geometry_msgs/Twist)       base u_R
/my_gen3/in/cartesian_velocity_desired (kortex TwistCommand) arm u_H_obs
/ARNA_TELEOP_MOV                 (std_msgs/Float32MultiArray) base u_H_obs
/network_watchdog/lambda_network (std_msgs/Float32)           λ_net
"""

import math
import threading
import time

import numpy as np
import rospy

from dynamic_reconfigure.server import Server as DynReconfigureServer
from geometry_msgs.msg          import Twist
from kortex_driver.msg          import TwistCommand
from std_msgs.msg               import Float32

from pick_place.cfg import OperatorIntentConfig


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def main():
    rospy.init_node('operator_intent_node', anonymous=False)

    # ── Dynamic reconfigure ────────────────────────────────────────────────────
    # Seed param server from node params so rqt_reconfigure always shows YAML values.
    for key, default in [
        ('pgd_step_size',     0.05),
        ('pgd_gamma',         0.01),
        ('sigmoid_sharpness', 8.0),
        ('lambda_op_min',     0.1),
        ('lambda_op_max',     0.9),
        ('alpha_floor',       0.01),
        ('excitation_thresh', 0.01),
        ('reset_on_idle',     True),
        ('idle_timeout_s',    5.0),
    ]:
        if not rospy.has_param('~' + key):
            rospy.set_param('~' + key, default)

    lock = threading.Lock()
    cfg  = {
        'pgd_step_size':     rospy.get_param('~pgd_step_size',     0.05),
        'pgd_gamma':         rospy.get_param('~pgd_gamma',         0.01),
        'sigmoid_sharpness': rospy.get_param('~sigmoid_sharpness', 8.0),
        'lambda_op_min':     rospy.get_param('~lambda_op_min',     0.1),
        'lambda_op_max':     rospy.get_param('~lambda_op_max',     0.9),
        'alpha_floor':       rospy.get_param('~alpha_floor',       0.01),
        'excitation_thresh': rospy.get_param('~excitation_thresh', 0.01),
        'reset_on_idle':     rospy.get_param('~reset_on_idle',     True),
        'idle_timeout_s':    rospy.get_param('~idle_timeout_s',    5.0),
    }

    def dynrec_cb(new_cfg, _level):
        with lock:
            cfg.update({
                'pgd_step_size':     new_cfg.pgd_step_size,
                'pgd_gamma':         new_cfg.pgd_gamma,
                'sigmoid_sharpness': new_cfg.sigmoid_sharpness,
                'lambda_op_min':     new_cfg.lambda_op_min,
                'lambda_op_max':     new_cfg.lambda_op_max,
                'alpha_floor':       new_cfg.alpha_floor,
                'excitation_thresh': new_cfg.excitation_thresh,
                'reset_on_idle':     new_cfg.reset_on_idle,
                'idle_timeout_s':    new_cfg.idle_timeout_s,
            })
        return new_cfg

    _dr = DynReconfigureServer(OperatorIntentConfig, dynrec_cb)

    # ── Shared state ───────────────────────────────────────────────────────────
    alpha       = 0.5          # neutral starting point
    lambda_net  = 0.0          # from network_watchdog_node

    # Safe references (u_R) — 6-DOF combined (arm 6D, base 3D → padded to 6D)
    u_R_arm  = np.zeros(6)    # from /mpc_cbf_arm/safe_reference
    u_R_base = np.zeros(6)    # from /base_cbf/safe_reference (vx,vy,oz → padded)

    # Observed human commands (u_H_obs)
    u_H_arm  = np.zeros(6)    # from /my_gen3/in/cartesian_velocity_desired
    u_H_base = np.zeros(6)    # from /ARNA_TELEOP_MOV

    last_joystick_time = time.time()   # for idle reset

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
            u_R_base = np.array([msg.linear.x, msg.linear.y, 0.0,
                                  0.0, 0.0, msg.angular.z])

    def arm_desired_cb(msg):
        nonlocal u_H_arm, last_joystick_time
        with lock:
            u_H_arm = np.array([
                msg.twist.linear_x, msg.twist.linear_y, msg.twist.linear_z,
                msg.twist.angular_x, msg.twist.angular_y, msg.twist.angular_z,
            ])
            last_joystick_time = time.time()

    def base_desired_cb(msg):
        """
        /cmd_vel_desired is geometry_msgs/Twist — actual velocity (m/s, rad/s),
        same units as /base_cbf/safe_reference.  Using this instead of the raw
        joystick /ARNA_TELEOP_MOV avoids a scale mismatch in the PGD model.
        """
        nonlocal u_H_base, last_joystick_time
        with lock:
            u_H_base = np.array([msg.linear.x, msg.linear.y, 0.0,
                                  0.0, 0.0, msg.angular.z])
            last_joystick_time = time.time()

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
            _last_joy = last_joystick_time

        # Idle reset
        if _cfg['reset_on_idle'] and (time.time() - _last_joy > _cfg['idle_timeout_s']):
            alpha = 0.5

        # Combine arm + base into unified 12-DOF vectors for PGD
        # (concatenate so arm and base contribute equally to gradient)
        u_R_all  = np.concatenate([_u_R_arm, _u_R_base])
        u_H_all  = np.concatenate([_u_H_arm, _u_H_base])

        # Excitation guard: skip PGD when safe reference is near-zero
        norm_uR = np.linalg.norm(u_R_all)
        if norm_uR >= _cfg['excitation_thresh']:
            # Normalized LMS gradient: divide by ||u_R||² so convergence rate
            # is scale-invariant regardless of command magnitude.
            # With pgd_step_size=0.1 this converges in ~10 steps (0.5 s at 20 Hz).
            e        = alpha * u_R_all - u_H_all       # prediction error (12D)
            grad_raw = np.dot(e, u_R_all)              # scalar gradient
            grad     = grad_raw / max(norm_uR ** 2, 1e-6)  # normalized
            a_fl = _cfg['alpha_floor']
            alpha = float(np.clip(
                alpha - _cfg['pgd_step_size'] * grad,
                a_fl, 1.0 - a_fl
            ))

        # λ_operator from alignment (= alpha)
        alignment = alpha
        k         = _cfg['sigmoid_sharpness']
        lambda_op = 1.0 - _sigmoid(k * (alignment - 0.5))
        lambda_op = float(np.clip(lambda_op, _cfg['lambda_op_min'], _cfg['lambda_op_max']))

        # Combined signal
        lambda_combined = _lnet * lambda_op

        # Publish
        pub_alpha.publish(Float32(data=alpha))
        pub_align.publish(Float32(data=alignment))
        pub_lambda_op.publish(Float32(data=lambda_op))
        pub_lambda_c.publish(Float32(data=lambda_combined))

    rospy.loginfo('[operator_intent] Shutting down.')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
