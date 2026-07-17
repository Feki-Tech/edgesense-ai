"""Prometheus metrics for the inference sidecar (MLOps phase 1).

Metrics live in a dedicated ``CollectorRegistry`` (not the global default) so
test suites that ``importlib.reload(inference.server)`` never hit duplicated
timeseries registration. Names are namespaced ``edgesense_model_*`` — the Go
edge agent already owns ``edgesense_readings_scored_total`` and friends on its
own /metrics endpoint.
"""

from __future__ import annotations

from prometheus_client import (CollectorRegistry, Counter, Gauge, Histogram,
                               make_asgi_app)

REGISTRY = CollectorRegistry()

SCORED = Counter(
    "edgesense_model_scored_total",
    "Readings scored by the inference sidecar",
    registry=REGISTRY)

ANOMALIES = Counter(
    "edgesense_model_anomalies_total",
    "Readings flagged anomalous by the sidecar, by trigger reason",
    ["reason"], registry=REGISTRY)

SCORE = Histogram(
    "edgesense_model_score",
    "Anomaly score distribution (mean squared reconstruction error)",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
             1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0),
    registry=REGISTRY)

DRIFT_ZSHIFT = Gauge(
    "edgesense_model_drift_zshift",
    "Rolling-mean shift vs the training mean, in training standard deviations",
    ["feature"], registry=REGISTRY)

DRIFT_PSI = Gauge(
    "edgesense_model_drift_psi",
    "Population stability index of the rolling window vs the training "
    "distribution (<0.1 stable, 0.1-0.25 moderate, >0.25 major drift)",
    ["feature"], registry=REGISTRY)

DRIFT_WINDOW = Gauge(
    "edgesense_model_drift_window_size",
    "Readings currently in the drift window",
    registry=REGISTRY)

MODEL_INFO = Gauge(
    "edgesense_model_info",
    "Live model metadata (value is always 1)",
    ["model_version", "kind", "backend"], registry=REGISTRY)

RELOADS = Counter(
    "edgesense_model_reloads_total",
    "Hot-reload attempts, by result",
    ["result"], registry=REGISTRY)


def set_model_info(version: str, kind: str, backend: str) -> None:
    """Point edgesense_model_info at the live model (single active label set)."""
    MODEL_INFO.clear()
    MODEL_INFO.labels(model_version=version, kind=kind, backend=backend).set(1)


def metrics_app():
    """ASGI app serving this registry, mountable at /metrics."""
    return make_asgi_app(registry=REGISTRY)
