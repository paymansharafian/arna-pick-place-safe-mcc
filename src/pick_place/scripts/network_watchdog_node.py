#!/usr/bin/env python3
"""
network_watchdog_node — Phase 3 safety coordinator.

Subscribes to /network_quality and transitions between four modes:
  NOMINAL  — normal operation, no overrides
  DEGRADED — tightens epsilon margins (×1.3), sets N_min = 15
  POOR     — tightens epsilon margins (×1.8), sets N = 25
  FAILED   — publishes zero-velocity to /cmd_vel_desired and
             /my_gen3/in/cartesian_velocity_desired every 50 ms

On each state transition the node calls the dynamic_reconfigure service on:
  - mpc_cbf_arm_node      (epsilon_base_workspace, epsilon_base_speed,
                            k_epsilon_workspace, k_epsilon_speed,
                            N_min, N_max via param update trick)
  - base_cbf_filter_node  (epsilon_base_lidar)

The FAILED safe-stop is additive on top of the existing 500 ms watchdog in
arna_teleop_fwd_node — neither that watchdog nor mpc_cbf_arm_node are modified.

Publishes /safety_mode (std_msgs/String) at 10 Hz for GUI display.
"""

import rospy
import threading

from std_msgs.msg     import String
from geometry_msgs.msg import Twist
from kortex_driver.msg import TwistCommand
from pick_place.msg   import NetworkQuality

import dynamic_reconfigure.client as drc

# ── Mode transition table ──────────────────────────────────────────────────────
# (epsilon_multiplier, N_arm_override)
# N_arm_override = None → let mpc_cbf_arm_node compute N from delta_max_ms
MODE_TABLE = {
    'NOMINAL':  {'eps_mult': 1.0,  'N_min_override': None, 'N_max_override': None},
    'DEGRADED': {'eps_mult': 1.3,  'N_min_override': 15,   'N_max_override': None},
    'POOR':     {'eps_mult': 1.8,  'N_min_override': 25,   'N_max_override': 25},
    'FAILED':   {'eps_mult': 1.8,  'N_min_override': None, 'N_max_override': None},
}

# Watchdog: if no /network_quality message arrives within this many seconds,
# force FAILED mode independently of the monitor node.
QUALITY_TIMEOUT_S = 1.0

# How often (Hz) to publish the zero-velocity flood in FAILED mode
FAILED_PUBLISH_HZ = 20


class NetworkWatchdog:
    def __init__(self):
        rospy.init_node('network_watchdog_node', anonymous=False)

        # ── Load baseline epsilon values from param server (written by the
        #    respective nodes before they create their DynReconfigureServers)
        self._arm_eps_base_ws  = rospy.get_param('/mpc_cbf_arm_node/epsilon_base_workspace', 0.02)
        self._arm_k_eps_ws     = rospy.get_param('/mpc_cbf_arm_node/k_epsilon_workspace',    0.001)
        self._arm_eps_base_sp  = rospy.get_param('/mpc_cbf_arm_node/epsilon_base_speed',     0.02)
        self._arm_k_eps_sp     = rospy.get_param('/mpc_cbf_arm_node/k_epsilon_speed',        0.0005)
        self._arm_N_min_base   = rospy.get_param('/mpc_cbf_arm_node/N_min',                  5)
        self._arm_N_max_base   = rospy.get_param('/mpc_cbf_arm_node/N_max',                  30)
        self._base_eps_lidar   = rospy.get_param('/base_cbf_filter_node/epsilon_base_lidar', 0.02)

        self._lock       = threading.Lock()
        self._mode       = 'NOMINAL'
        self._last_qual_t = None    # rospy.Time of last /network_quality message

        # ── dynrec clients (lazy — created on first use so the node starts
        #    even if the filter nodes have not launched yet) ──────────────────
        self._arm_dr   = None
        self._base_dr  = None

        # ── Publishers ────────────────────────────────────────────────────────
        self._mode_pub = rospy.Publisher('/safety_mode', String, queue_size=1, latch=True)

        # Zero-velocity publishers for FAILED mode
        self._base_zero_pub = rospy.Publisher(
            '/cmd_vel_desired', Twist, queue_size=1)
        self._arm_zero_pub  = rospy.Publisher(
            '/my_gen3/in/cartesian_velocity_desired', TwistCommand, queue_size=1)

        # ── Subscribers ───────────────────────────────────────────────────────
        rospy.Subscriber('/network_quality', NetworkQuality, self._quality_cb, queue_size=1)

        # ── Timers ────────────────────────────────────────────────────────────
        # 10 Hz: publish /safety_mode and check watchdog timeout
        rospy.Timer(rospy.Duration(0.1),  self._tick_cb)
        # 50 ms: flood zeros in FAILED mode
        rospy.Timer(rospy.Duration(1.0 / FAILED_PUBLISH_HZ), self._failed_zero_cb)

        # Publish initial mode
        self._mode_pub.publish(String(data='NOMINAL'))
        rospy.loginfo('[network_watchdog] Ready — monitoring /network_quality')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _arm_client(self):
        if self._arm_dr is None:
            try:
                self._arm_dr = drc.Client('mpc_cbf_arm_node', timeout=2.0)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f'[network_watchdog] arm dynrec unavailable: {e}')
        return self._arm_dr

    def _base_client(self):
        if self._base_dr is None:
            try:
                self._base_dr = drc.Client('base_cbf_filter_node', timeout=2.0)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f'[network_watchdog] base dynrec unavailable: {e}')
        return self._base_dr

    def _apply_mode(self, mode: str):
        """Push dynrec updates for the given mode."""
        cfg = MODE_TABLE[mode]
        mult = cfg['eps_mult']

        # ── arm dynrec ────────────────────────────────────────────────────────
        arm = self._arm_client()
        if arm is not None:
            arm_cfg = {
                'epsilon_base_workspace': self._arm_eps_base_ws * mult,
                'k_epsilon_workspace':    self._arm_k_eps_ws    * mult,
                'epsilon_base_speed':     self._arm_eps_base_sp * mult,
                'k_epsilon_speed':        self._arm_k_eps_sp    * mult,
            }
            if cfg['N_min_override'] is not None:
                arm_cfg['N_min'] = cfg['N_min_override']
            else:
                arm_cfg['N_min'] = self._arm_N_min_base
            if cfg['N_max_override'] is not None:
                arm_cfg['N_max'] = cfg['N_max_override']
            else:
                arm_cfg['N_max'] = self._arm_N_max_base
            try:
                arm.update_configuration(arm_cfg)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f'[network_watchdog] arm dynrec update failed: {e}')
                self._arm_dr = None   # force reconnect next cycle

        # ── base dynrec ───────────────────────────────────────────────────────
        base = self._base_client()
        if base is not None:
            base_cfg = {
                'epsilon_base_lidar': self._base_eps_lidar * mult,
            }
            try:
                base.update_configuration(base_cfg)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f'[network_watchdog] base dynrec update failed: {e}')
                self._base_dr = None

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _quality_cb(self, msg: NetworkQuality):
        with self._lock:
            self._last_qual_t = rospy.Time.now()
            new_mode = msg.network_state   # NOMINAL / DEGRADED / POOR / FAILED

        self._transition(new_mode)

    def _tick_cb(self, _event):
        """10 Hz: publish /safety_mode; check watchdog timeout."""
        with self._lock:
            mode = self._mode
            last = self._last_qual_t

        # Watchdog: if monitor is silent, escalate to FAILED
        if last is not None:
            age = (rospy.Time.now() - last).to_sec()
            if age > QUALITY_TIMEOUT_S and mode != 'FAILED':
                rospy.logerr(
                    f'[network_watchdog] /network_quality silent for {age:.1f} s — FAILED')
                self._transition('FAILED')
                return

        self._mode_pub.publish(String(data=mode))

    def _failed_zero_cb(self, _event):
        """20 Hz: flood zero-velocity in FAILED mode."""
        with self._lock:
            mode = self._mode
        if mode != 'FAILED':
            return

        self._base_zero_pub.publish(Twist())

        arm_zero = TwistCommand()
        arm_zero.reference_frame = 0
        arm_zero.duration = 0
        self._arm_zero_pub.publish(arm_zero)

    def _transition(self, new_mode: str):
        with self._lock:
            old_mode = self._mode
            if new_mode == old_mode:
                return
            self._mode = new_mode

        if new_mode == 'NOMINAL':
            rospy.loginfo('[network_watchdog] → NOMINAL')
        elif new_mode == 'DEGRADED':
            rospy.loginfo('[network_watchdog] → DEGRADED  (eps ×1.3, N_min=15)')
        elif new_mode == 'POOR':
            rospy.logwarn('[network_watchdog] → POOR  (eps ×1.8, N forced=25)')
        elif new_mode == 'FAILED':
            rospy.logerr('[network_watchdog] → FAILED  (zero-velocity flood active)')

        self._apply_mode(new_mode)
        self._mode_pub.publish(String(data=new_mode))


def main():
    watchdog = NetworkWatchdog()
    rospy.spin()


if __name__ == '__main__':
    main()
