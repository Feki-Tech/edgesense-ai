"""Serving-side drift detection for the inference sidecar (MLOps phase 1).

The bundle already carries the training distribution (``scaler_mean`` /
``scaler_scale`` per feature), so drift needs no extra artifacts: the tracker
keeps a rolling window of the raw readings it scored and derives two cheap,
CPU-friendly signals per feature:

- **z-shift** — the rolling mean's distance from the training mean in training
  standard deviations: ``(rolling_mean - train_mean) / train_scale``. Signed,
  so a dashboard shows drift direction; |z-shift| ≳ 0.5 is a meaningful shift.
- **PSI** (population stability index) — compares the rolling distribution of
  the *standardized* feature against the training distribution over fixed bins
  spanning ±4σ (plus open-ended tail bins). Healthy training data is generated
  as N(mean, std) (``ml/train.py``), so the expected bin mass is the standard
  normal's. Usual reading: PSI < 0.1 stable, 0.1–0.25 moderate, > 0.25 major.

Everything is pure numpy over a small ring buffer (default 500 readings),
guarded by a lock — safe for the threaded FastAPI server and cheap at 2 Hz.
"""

from __future__ import annotations

import math
import threading

import numpy as np

DEFAULT_WINDOW = 500
MIN_SAMPLES = 50      # no drift signal until the window has this many readings
_N_INNER_BINS = 10    # inner bins spanning ±4σ; plus 2 open-ended tail bins
_EPS = 1e-4           # PSI flooring for empty bins

# fixed bin edges on the standardized scale: (-inf, -4, ..., +4, +inf)
_EDGES = np.linspace(-4.0, 4.0, _N_INNER_BINS + 1)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def expected_bin_probs() -> np.ndarray:
    """Standard-normal mass of each bin (tails included)."""
    cdf = _normal_cdf(_EDGES)
    return np.concatenate([[cdf[0]], np.diff(cdf), [1.0 - cdf[-1]]])


_EXPECTED = expected_bin_probs()


def psi(z_values: np.ndarray) -> float:
    """PSI of standardized samples vs the standard-normal training reference."""
    counts, _ = np.histogram(z_values, bins=np.concatenate([[-np.inf], _EDGES, [np.inf]]))
    actual = np.maximum(counts / max(len(z_values), 1), _EPS)
    expected = np.maximum(_EXPECTED, _EPS)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


class DriftTracker:
    """Rolling per-feature statistics of scored readings vs the training set."""

    def __init__(self, features: list[str], train_mean: np.ndarray,
                 train_scale: np.ndarray, window: int = DEFAULT_WINDOW,
                 min_samples: int = MIN_SAMPLES) -> None:
        self.features = list(features)
        self.window = max(int(window), 1)
        self.min_samples = min(min_samples, self.window)
        self._mean = np.asarray(train_mean, dtype=float)
        self._scale = np.where(np.asarray(train_scale, dtype=float) == 0, 1.0,
                               np.asarray(train_scale, dtype=float))
        self._buf = np.zeros((self.window, len(self.features)))
        self._n = 0          # readings seen since (re)start
        self._lock = threading.Lock()

    def observe(self, x: "np.ndarray | list[float]") -> None:
        """Record one raw reading (feature order must match the bundle's)."""
        arr = np.asarray(x, dtype=float).reshape(-1)
        with self._lock:
            self._buf[self._n % self.window] = arr
            self._n += 1

    def reset(self, train_mean: np.ndarray | None = None,
              train_scale: np.ndarray | None = None) -> None:
        """Clear the window (e.g. after a model reload swapped the scaler)."""
        with self._lock:
            if train_mean is not None:
                self._mean = np.asarray(train_mean, dtype=float)
            if train_scale is not None:
                scale = np.asarray(train_scale, dtype=float)
                self._scale = np.where(scale == 0, 1.0, scale)
            self._n = 0

    @property
    def size(self) -> int:
        return min(self._n, self.window)

    def signals(self) -> dict[str, dict[str, float]]:
        """Per-feature drift signals: ``{feature: {"zshift": .., "psi": ..}}``.

        Empty dict until ``min_samples`` readings have been observed.
        """
        with self._lock:
            n = min(self._n, self.window)
            if n < self.min_samples:
                return {}
            data = self._buf[:n].copy()
        z = (data - self._mean) / self._scale
        rolling = z.mean(axis=0)
        return {feat: {"zshift": float(rolling[i]), "psi": psi(z[:, i])}
                for i, feat in enumerate(self.features)}
