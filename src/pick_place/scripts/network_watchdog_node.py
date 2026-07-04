#!/usr/bin/env python3
"""
network_watchdog_node — Layer 3 safety coordinator.

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
  - base_mpc_cbf_node     (epsilon_base_lidar)

The FAILED safe-stop is additive on top of the existing 500 ms watchdog in
arna_teleop_fwd_node — neither that watchdog nor mpc_cbf_arm_node are modified.

Publishes /safety_mode (std_msgs/String) at 10 Hz for GUI display.
"""

import rospy
import threading

from std_msgs.msg     import String, Float32
from geometry_msgs.msg import Twist
from kortex_driver.msg import TwistCommand
from pick_place.msg   import NetworkQuality

import dynamic_reconfigure.client as drc

# ── Mode transition table ──────────────────────────────────────────────────────
# (epsilon_multiplier, N_arm_override, lambda_network)
MODE_TABLE = {
    'NOMINAL':  {'eps_mult': 1.0,  'N_min_override': None, 'N_max_override': None, 'lambda_network': 0.0},
    'DEGRADED': {'eps_mult': 1.3,  'N_min_override': 15,   'N_max_override': None, 'lambda_network': 0.3},
    'POOR':     {'eps_mult': 1.8,  'N_min_override': 25,   'N_max_override': 25,   'lambda_network': 0.6},
    'FAILED':   {'eps_mult': 1.8,  'N_min_override': None, 'N_max_override': None, 'lambda_network': 1.0},
}

# Watchdog: if no /network_quality message arrives within this many seconds,
# force FAILED mode independently of the monitor node.
QUALITY_TIMEOUT_S = 1.0

# How often (Hz) to publish the zero-velocity flood in FAILED mode
FAILED_PUBLISH_HZ = 20

# Minimum dwell time (s) in the current mode before any downward transition
# (POOR→DEGRADED, DEGRADED→NOMINAL, etc.).  Upward transitions are always immediate.
DWELL_DOWN_S = 3.0

# Severity order used to decide transition direction (higher = more severe)
MODE_SEVERITY = {'NOMINAL': 0, 'DEGRADED': 1, 'POOR': 2, 'FAILED': 3}

# Per-parameter maxima from the dynrec .cfg files (BaseMpcCbfFilter.cfg /
# MpcCbfArm.cfg).  Every dynrec write is clamped to these so a large eps_mult
# (or any stale baseline) can never push a safety margin past its configured
# ceiling — rather than silently relying on the dynrec server to saturate it.
EPS_BASE_LIDAR_MAX     = 0.11   # base_mpc_cbf_node/epsilon_base_lidar
EPS_BASE_WORKSPACE_MAX = 0.5    # mpc_cbf_arm_node/epsilon_base_workspace
K_EPS_WORKSPACE_MAX    = 0.05   # mpc_cbf_arm_node/k_epsilon_workspace
EPS_BASE_SPEED_MAX     = 0.2    # mpc_cbf_arm_node/epsilon_base_speed
K_EPS_SPEED_MAX        = 0.01   # mpc_cbf_arm_node/k_epsilon_speed


class NetworkWatchdog:
    def __init__(self):
        rospy.init_node('network_watchdog_node', anonymous=False)

        # ── Baseline (NOMINAL) margins that eps_mult is applied to ────────────
        # These MUST come from a source this node never writes.  Previously they
        # were read from /<filter_node>/<param> — the very params the watchdog
        # reconfigures.  Because those values persist on the param server across
        # node restarts, each watchdog restart re-read an already-multiplied
        # value as its new baseline and multiplied it again, ratcheting the
        # margins up to the cfg ceiling (epsilon_base_lidar pinned at 0.09 while
        # driving).  Reading from private ~*_nominal params (defaulting to the
        # cfg/YAML nominal values) decouples the baseline from the written value,
        # so margins always return to nominal when the network recovers.
        self._arm_eps_base_ws  = rospy.get_param('~arm_epsilon_base_workspace_nominal', 0.02)
        self._arm_k_eps_ws     = rospy.get_param('~arm_k_epsilon_workspace_nominal',    0.001)
        self._arm_eps_base_sp  = rospy.get_param('~arm_epsilon_base_speed_nominal',     0.02)
        self._arm_k_eps_sp     = rospy.get_param('~arm_k_epsilon_speed_nominal',        0.0005)
        self._arm_N_min_base   = rospy.get_param('~arm_N_min_nominal',                  10)
        self._arm_N_max_base   = rospy.get_param('~arm_N_max_nominal',                  25)
        self._base_eps_lidar   = rospy.get_param('~base_epsilon_base_lidar_nominal',    0.02)

        self._lock       = threading.Lock()
        self._mode       = 'NOMINAL'
        self._mode_enter_t = rospy.Time.now()   # time we entered the current mode
        self._last_qual_t = None    # rospy.Time of last /network_quality message

        # ── dynrec clients (lazy — created on first use so the node starts
        #    even if the filter nodes have not launched yet) ──────────────────
        self._arm_dr   = None
        self._base_dr  = None

        # ── Background dynrec worker ──────────────────────────────────────────
        # _apply_mode() can block for seconds on drc.Client() timeouts when the
        # filter nodes are not running.  Running it on the ROS callback thread
        # would starve _quality_cb and _tick_cb, causing spurious FAILED states.
        # Solution: a single daemon thread processes dynrec updates; only the
        # latest pending mode is kept so rapid transitions collapse to one call.
        self._dynrec_pending = None
        self._dynrec_cond    = threading.Condition()
        self._dynrec_thread  = threading.Thread(
            target=self._dynrec_worker, daemon=True, name='dynrec-worker')
        self._dynrec_thread.start()

        # ── Publishers ────────────────────────────────────────────────────────
        self._mode_pub = rospy.Publisher('/safety_mode', String, queue_size=1, latch=True)
        self._lambda_network_pub = rospy.Publisher(
            '/network_watchdog/lambda_network', Float32, queue_size=1, latch=True)

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
        self._lambda_network_pub.publish(Float32(data=0.0))
        rospy.loginfo('[network_watchdog] Ready — monitoring /network_quality')

    # ── Background dynrec worker ──────────────────────────────────────────────

    def _dynrec_worker(self):
        """Applies dynrec updates in background without blocking ROS callbacks."""
        while not rospy.is_shutdown():
            with self._dynrec_cond:
                while self._dynrec_pending is None:
                    self._dynrec_cond.wait(timeout=0.5)
                mode = self._dynrec_pending
                self._dynrec_pending = None
            self._apply_mode(mode)

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
                self._base_dr = drc.Client('base_mpc_cbf_node', timeout=2.0)
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
                'epsilon_base_workspace': min(self._arm_eps_base_ws * mult, EPS_BASE_WORKSPACE_MAX),
                'k_epsilon_workspace':    min(self._arm_k_eps_ws    * mult, K_EPS_WORKSPACE_MAX),
                'epsilon_base_speed':     min(self._arm_eps_base_sp * mult, EPS_BASE_SPEED_MAX),
                'k_epsilon_speed':        min(self._arm_k_eps_sp    * mult, K_EPS_SPEED_MAX),
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
                'epsilon_base_lidar': min(self._base_eps_lidar * mult, EPS_BASE_LIDAR_MAX),
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
        self._lambda_network_pub.publish(Float32(data=MODE_TABLE.get(mode, MODE_TABLE['FAILED'])['lambda_network']))

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
            # Upward (more severe) transitions are immediate — protection first.
            # Downward transitions require the node to have dwelt in the current
            # mode for at least DWELL_DOWN_S seconds to suppress chattering.
            if MODE_SEVERITY[new_mode] < MODE_SEVERITY[old_mode]:
                age = (rospy.Time.now() - self._mode_enter_t).to_sec()
                if age < DWELL_DOWN_S:
                    return   # not ready to downgrade yet
            self._mode = new_mode
            self._mode_enter_t = rospy.Time.now()

        if new_mode == 'NOMINAL':
            rospy.loginfo('[network_watchdog] → NOMINAL')
        elif new_mode == 'DEGRADED':
            rospy.loginfo('[network_watchdog] → DEGRADED  (eps ×1.3, N_min=15)')
        elif new_mode == 'POOR':
            rospy.logwarn('[network_watchdog] → POOR  (eps ×1.8, N forced=25)')
        elif new_mode == 'FAILED':
            rospy.logerr('[network_watchdog] → FAILED  (zero-velocity flood active)')

        # Enqueue dynrec update — processed by background thread so this
        # method returns immediately and never blocks the callback thread.
        with self._dynrec_cond:
            self._dynrec_pending = new_mode
            self._dynrec_cond.notify()

        self._mode_pub.publish(String(data=new_mode))
        lam = MODE_TABLE[new_mode]['lambda_network']
        self._lambda_network_pub.publish(Float32(data=lam))


def main():
    watchdog = NetworkWatchdog()
    rospy.spin()


if __name__ == '__main__':
    main()
