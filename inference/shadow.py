"""Shadow scoring: run a challenger bundle beside the champion on live traffic.

The champion alone decides every /score response; a loaded shadow scores the
same readings in the background and its agreement with the champion is
accumulated as *online* promotion evidence (docs/MLOPS.md §2.5) — the offline
gate (ml/promote.py) parks refused candidates in ml/model/candidate/, which is
exactly where the shadow loads from by default. A misbehaving shadow can never
affect serving: its failures are counted and the request completes normally.
"""

from __future__ import annotations

import threading
import time


class ShadowTracker:
    """Champion-vs-shadow agreement stats, updated once per scored reading."""

    def __init__(self, shadow_version: str, champion_version: str) -> None:
        self.shadow_version = shadow_version
        self.champion_version = champion_version
        self.since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._lock = threading.Lock()
        self.n = 0
        self.agree = 0
        self.champion_only = 0  # champion flagged anomaly, shadow did not
        self.shadow_only = 0    # shadow flagged anomaly, champion did not
        self.errors = 0
        self._diff_sum = 0.0
        self._abs_diff_sum = 0.0

    def observe(self, champion_score: float, champion_anomaly: bool,
                shadow_score: float, shadow_anomaly: bool) -> None:
        with self._lock:
            self.n += 1
            if shadow_anomaly == champion_anomaly:
                self.agree += 1
            elif shadow_anomaly:
                self.shadow_only += 1
            else:
                self.champion_only += 1
            diff = shadow_score - champion_score
            self._diff_sum += diff
            self._abs_diff_sum += abs(diff)

    def error(self) -> None:
        with self._lock:
            self.errors += 1

    def report(self) -> dict:
        with self._lock:
            n = self.n
            return {
                "shadow_version": self.shadow_version,
                "champion_version": self.champion_version,
                "since": self.since,
                "n": n,
                "agree": self.agree,
                "champion_only": self.champion_only,
                "shadow_only": self.shadow_only,
                "errors": self.errors,
                "agreement_rate": round(self.agree / n, 5) if n else None,
                "score_mae": round(self._abs_diff_sum / n, 6) if n else None,
                "score_bias": round(self._diff_sum / n, 6) if n else None,
            }
