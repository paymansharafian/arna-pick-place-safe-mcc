#!/usr/bin/env python3
"""
network_monitor_node — Phase 0 network quality monitor.

Subscribes to /browser_rtt_ms (std_msgs/Float64) published by the frontend
after each ping/pong round-trip.  Maintains a rolling window of 300 samples
(≈30 s at 10 Hz probe rate) and publishes /network_quality (pick_place/NetworkQuality)
at 10 Hz.

Network state classification thresholds:
  NOMINAL  — delta_max_ms < 80   AND loss_rate_pct < 1.0
  DEGRADED — 80 ≤ delta_max_ms < 200  OR  1.0 ≤ loss_rate_pct < 5.0
  POOR     — delta_max_ms ≥ 200  OR  loss_rate_pct ≥ 5.0
  FAILED   — no /browser_rtt_ms message received within 500 ms
"""

import rospy
import numpy as np
from collections import deque
from std_msgs.msg import Float64
from pick_place.msg import NetworkQuality

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_SIZE      = 300          # samples (30 s at 10 Hz)
PUBLISH_RATE_HZ  = 10
FAILED_TIMEOUT_S = 0.5          # seconds without a pong → FAILED
PROBE_INTERVAL_S = 0.1          # browser sends pings at 100 ms = 10 Hz


class NetworkMonitor:
    def __init__(self):
        rospy.init_node('network_monitor_node', anonymous=False)

        self._window: deque[float] = deque(maxlen=WINDOW_SIZE)
        self._last_rtt_time = None   # rospy.Time of last received sample
        self._start_time    = None   # rospy.Time of first received sample (loss denominator)

        self._pub = rospy.Publisher(
            '/network_quality', NetworkQuality, queue_size=5
        )
        rospy.Subscriber(
            '/browser_rtt_ms', Float64, self._rtt_cb, queue_size=50
        )

        rospy.Timer(
            rospy.Duration(1.0 / PUBLISH_RATE_HZ), self._publish_cb
        )

        rospy.loginfo('[network_monitor_node] Ready — waiting for /browser_rtt_ms')

    # ── Incoming RTT sample ────────────────────────────────────────────────────
    def _rtt_cb(self, msg: Float64):
        rtt = float(msg.data)
        if rtt < 0:
            return  # ignore bogus values
        self._window.append(rtt)
        self._last_rtt_time = rospy.Time.now()
        if self._start_time is None:
            self._start_time = self._last_rtt_time

    # ── Periodic publisher ─────────────────────────────────────────────────────
    def _publish_cb(self, _event):
        now = rospy.Time.now()

        # ── FAILED check: no sample in the last 500 ms ─────────────────────
        if self._last_rtt_time is None or \
                (now - self._last_rtt_time).to_sec() > FAILED_TIMEOUT_S:
            msg = NetworkQuality()
            msg.rtt_mean_ms    = 0.0
            msg.rtt_std_ms     = 0.0
            msg.jitter_ms      = 0.0
            msg.loss_rate_pct  = 100.0
            msg.delta_max_ms   = 9999.0
            msg.network_state  = 'FAILED'
            self._pub.publish(msg)
            return

        # ── Statistics ─────────────────────────────────────────────────────
        arr = np.array(self._window, dtype=np.float64)

        rtt_mean   = float(np.mean(arr))
        rtt_std    = float(np.std(arr))

        # jitter = mean absolute deviation between consecutive samples
        if len(arr) > 1:
            jitter = float(np.mean(np.abs(np.diff(arr))))
        else:
            jitter = 0.0

        # 99th-percentile one-way delay = 99th-pct RTT / 2
        delta_max = float(np.percentile(arr, 99)) / 2.0

        # ── Loss rate ──────────────────────────────────────────────────────
        # Use wall-clock time since the first probe arrived as the denominator,
        # capped at the rolling-window span (WINDOW_SIZE * PROBE_INTERVAL_S).
        # This ensures that missing probes correctly increase the expected count
        # rather than shrinking the window duration and cancelling themselves out.
        observation_s = min(
            (now - self._start_time).to_sec(),
            WINDOW_SIZE * PROBE_INTERVAL_S
        )
        expected  = max(1, observation_s / PROBE_INTERVAL_S)
        loss_pct  = max(0.0, (1.0 - len(arr) / expected) * 100.0)

        # ── State classification ───────────────────────────────────────────
        if delta_max < 80.0 and loss_pct < 1.0:
            state = 'NOMINAL'
        elif delta_max >= 200.0 or loss_pct >= 5.0:
            state = 'POOR'
        elif delta_max >= 80.0 or loss_pct >= 1.0:
            state = 'DEGRADED'
        else:
            state = 'NOMINAL'

        msg = NetworkQuality()
        msg.rtt_mean_ms   = rtt_mean
        msg.rtt_std_ms    = rtt_std
        msg.jitter_ms     = jitter
        msg.loss_rate_pct = loss_pct
        msg.delta_max_ms  = delta_max
        msg.network_state = state
        self._pub.publish(msg)


def main():
    monitor = NetworkMonitor()
    rospy.spin()


if __name__ == '__main__':
    main()
