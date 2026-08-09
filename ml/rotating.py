"""Load the KAIST rotating-machine dataset and reduce it to EdgeSense readings.

Dataset: "Vibration, Acoustic, Temperature, and Motor Current Dataset of
Rotating Machine Under Varying Load Conditions for Fault Diagnosis"
(Jung, Yun, Bae, Lim, Kim & Park — KAIST), Mendeley Data, CC BY 4.0:
https://data.mendeley.com/datasets/ztmf3m7h5x

Why this one: it is the only openly licensed set carrying all three signals
EdgeSense actually serves — vibration, temperature and motor current — on the
same machine at the same time. `ml/benchmark_public.py` (AI4I 2020) validates
the *architecture* on real data, but on five unrelated tabular features; this
module is what lets the benchmark exercise the real feature contract.

Raw layout (see the dataset's Description PDF):

    vibration/<load>Nm_<condition>.mat     25.6 kHz, unit g (1g = 9.80665 m/s^2)
        columns: Time Stamp, x_direction_housing_A, y_direction_housing_A,
                 x_direction_housing_B, y_direction_housing_B
    current,temp/<load>Nm_<condition>.tdms 25.6 kHz, Celsius and ampere
        columns: Time Stamp, Temperature_housing_A, Temperature_housing_B,
                 U-phase, V-phase, W-phase

    load      ∈ {0, 2, 4} Nm
    condition ∈ Normal | BPFI_{03,10,30} | BPFO_{03,10,30}
                | Misalign_{01,03,05} | Unbalance_{0583,1169,1751,2239,3318}mg
    duration  = 120 s for Normal, 60 s for every fault

Reduction to the reading contract: 25.6 kHz is the *sampling* rate; EdgeSense
reports at ~2 Hz. Each window collapses to one reading, exactly the
sensor-adapter behaviour docs/HARDWARE.md §4 specifies for real hardware:

    vibration   RMS over the window, pooled across the four accelerometer axes
    temperature mean of the two housing thermocouples (slow-varying)
    current     mean of the per-phase RMS over U/V/W

Windows are **non-overlapping** on purpose: overlapping them would inflate the
sample count while leaking the same milliseconds into both the training and
test splits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE_HZ = 25_600.0
READING_RATE_HZ = 2.0  # the contract in README/PLATFORM: ~2 Hz per machine

FEATURES = ["vibration", "temperature", "current"]

VIBRATION_CHANNELS = ("x_direction_housing_A", "y_direction_housing_A",
                      "x_direction_housing_B", "y_direction_housing_B")
TEMPERATURE_CHANNELS = ("Temperature_housing_A", "Temperature_housing_B")
CURRENT_CHANNELS = ("U-phase", "V-phase", "W-phase")

# condition stem -> the fault family EdgeSense would report
FAULT_FAMILIES = {
    "Normal": "healthy",
    "BPFI": "bearing inner race fault",
    "BPFO": "bearing outer race fault",
    "Misalign": "shaft misalignment",
    "Unbalance": "rotor unbalance",
}

_STEM_RE = re.compile(r"^(?P<load>\d+)Nm_(?P<condition>.+)$")


class DatasetError(RuntimeError):
    """Dataset missing, incomplete, or not in the documented layout."""


@dataclass(frozen=True)
class Recording:
    """One machine state: a load level plus a health condition."""

    load_nm: int
    condition: str  # e.g. "Normal", "BPFI_03", "Unbalance_1169mg"

    @property
    def family(self) -> str:
        head = self.condition.split("_")[0]
        return FAULT_FAMILIES.get(head, head)

    @property
    def is_healthy(self) -> bool:
        return self.condition == "Normal"

    @property
    def stem(self) -> str:
        return f"{self.load_nm}Nm_{self.condition}"

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.stem} ({self.family})"


def parse_stem(stem: str) -> Recording:
    """"0Nm_BPFI_03" -> Recording(load_nm=0, condition="BPFI_03")."""
    m = _STEM_RE.match(stem)
    if not m:
        raise DatasetError(f"unrecognised file stem {stem!r}")
    return Recording(load_nm=int(m["load"]), condition=m["condition"])


# --- raw readers ---------------------------------------------------------

def _columns_from_mapping(obj, names: "tuple[str, ...]") -> "dict[str, np.ndarray] | None":
    """Pull the named 1-D columns out of a dict-like of arrays."""
    try:
        keys = set(obj.keys())
    except AttributeError:
        return None
    if not set(names) <= keys:
        return None
    return {n: np.asarray(obj[n]).squeeze().astype(float) for n in names}


def read_mat_columns(path: Path,
                     names: "tuple[str, ...]" = VIBRATION_CHANNELS) -> "dict[str, np.ndarray]":
    """Read named columns from a MATLAB file, v7 or v7.3 (HDF5).

    The dataset's own docs describe partial ("matfile object") access, which
    implies v7.3 — but not every file in the wild is, so try the classic
    reader first and fall back to HDF5.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - scipy ships with sklearn
        raise DatasetError("scipy is required to read the .mat files") from exc

    try:
        mat = loadmat(str(path), squeeze_me=True)
    except (NotImplementedError, ValueError):
        return _read_mat_v73(path, names)

    cols = _columns_from_mapping(mat, names)
    if cols is not None:
        return cols

    # Fall back to a single 5-column matrix (Time Stamp + the four axes).
    for value in mat.values():
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[1] >= len(names) + 1:
            return {n: arr[:, i + 1].astype(float) for i, n in enumerate(names)}
    raise DatasetError(f"{path.name}: no {names} columns and no 5-column matrix")


def _read_mat_v73(path: Path, names: "tuple[str, ...]") -> "dict[str, np.ndarray]":
    try:
        import h5py
    except ImportError as exc:
        raise DatasetError(
            f"{path.name} is a MATLAB v7.3 file; install the 'benchmark' extra "
            "(h5py) to read it") from exc

    with h5py.File(path, "r") as fh:
        missing = [n for n in names if n not in fh]
        if missing:
            raise DatasetError(f"{path.name}: missing datasets {missing}")
        # h5py returns MATLAB arrays transposed; squeeze handles (1, N) / (N, 1).
        return {n: np.asarray(fh[n]).squeeze().astype(float) for n in names}


def read_tdms_columns(path: Path,
                      names: "tuple[str, ...]" = TEMPERATURE_CHANNELS + CURRENT_CHANNELS
                      ) -> "dict[str, np.ndarray]":
    """Read named channels from a TDMS file, searching every group."""
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise DatasetError(
            "npTDMS is required to read the .tdms files; install the "
            "'benchmark' extra") from exc

    found: dict[str, np.ndarray] = {}
    with TdmsFile.open(str(path)) as tdms:
        for group in tdms.groups():
            for channel in group.channels():
                if channel.name in names and channel.name not in found:
                    found[channel.name] = np.asarray(channel[:]).squeeze().astype(float)
    missing = [n for n in names if n not in found]
    if missing:
        raise DatasetError(f"{path.name}: missing channels {missing}")
    return found


# --- reduction to readings ----------------------------------------------

def _window_count(lengths: "list[int]", size: int) -> int:
    return min(lengths) // size


def reduce_to_readings(vibration: "dict[str, np.ndarray]",
                       temp_current: "dict[str, np.ndarray]",
                       *, sample_rate: float = SAMPLE_RATE_HZ,
                       reading_rate: float = READING_RATE_HZ) -> np.ndarray:
    """Collapse raw channels into (n_windows, 3) [vibration, temperature, current].

    Channels are truncated to a whole number of common windows — the vibration
    and temp/current files are recorded separately, so their lengths can differ
    by a few samples.
    """
    size = int(round(sample_rate / reading_rate))
    if size < 1:
        raise ValueError("reading_rate must be lower than sample_rate")

    vib = np.vstack([vibration[c] for c in VIBRATION_CHANNELS])
    temp = np.vstack([temp_current[c] for c in TEMPERATURE_CHANNELS])
    cur = np.vstack([temp_current[c] for c in CURRENT_CHANNELS])

    n = _window_count([vib.shape[1], temp.shape[1], cur.shape[1]], size)
    if n < 1:
        raise DatasetError(
            f"recording shorter than one {1 / reading_rate:.2f}s window")

    def windows(a: np.ndarray) -> np.ndarray:
        """(channels, n*size) -> (channels, n, size)"""
        return a[:, :n * size].reshape(a.shape[0], n, size)

    # RMS pooled over all four axes: one scalar vibration level per window.
    vib_rms = np.sqrt(np.mean(windows(vib) ** 2, axis=(0, 2)))
    # Temperature is slow-varying — the window mean is the natural reading.
    temp_mean = np.mean(windows(temp), axis=(0, 2))
    # Per-phase RMS first, then average: phases are 120° apart, so pooling the
    # raw samples the way vibration does would understate the true magnitude.
    cur_rms = np.mean(np.sqrt(np.mean(windows(cur) ** 2, axis=2)), axis=0)

    return np.column_stack([vib_rms, temp_mean, cur_rms])


# --- dataset assembly ----------------------------------------------------

def discover(root: Path) -> "list[Recording]":
    """Recordings that have BOTH a .mat and a .tdms file under root."""
    vib_dir, tc_dir = _resolve_dirs(root)
    stems = {p.stem for p in vib_dir.glob("*.mat")} & {p.stem for p in tc_dir.glob("*.tdms")}
    return sorted((parse_stem(s) for s in stems),
                  key=lambda r: (r.load_nm, r.condition))


def _resolve_dirs(root: Path) -> "tuple[Path, Path]":
    """Locate the vibration and current/temperature directories under root.

    The published archives unpack to 'vibration/' and 'current,temp/', but the
    comma makes that name easy to change by hand, so accept a few spellings.
    """
    root = Path(root)

    def first_with(candidates: "tuple[Path, ...]", pattern: str) -> "Path | None":
        return next((d for d in candidates
                     if d.is_dir() and any(d.glob(pattern))), None)

    vib = first_with((root / "vibration", root), "*.mat")
    tc = first_with(tuple(root / n for n in
                          ("current,temp", "current_temp", "current-temp",
                           "temp,current")) + (root,), "*.tdms")
    if vib is None or tc is None:
        raise DatasetError(
            f"{root} does not look like the rotating-machine dataset: expected "
            "a 'vibration/' directory of .mat files and a 'current,temp/' "
            "directory of .tdms files (see docs/BENCHMARK.md)")
    return vib, tc


def load_recording(root: Path, rec: Recording, *,
                   sample_rate: float = SAMPLE_RATE_HZ,
                   reading_rate: float = READING_RATE_HZ) -> np.ndarray:
    """Readings for one machine state, shape (n_windows, 3)."""
    vib_dir, tc_dir = _resolve_dirs(root)
    vibration = read_mat_columns(vib_dir / f"{rec.stem}.mat")
    temp_current = read_tdms_columns(tc_dir / f"{rec.stem}.tdms")
    return reduce_to_readings(vibration, temp_current, sample_rate=sample_rate,
                              reading_rate=reading_rate)


def load_dataset(root: Path, *, sample_rate: float = SAMPLE_RATE_HZ,
                 reading_rate: float = READING_RATE_HZ,
                 progress: bool = False) -> "tuple[np.ndarray, dict[str, np.ndarray]]":
    """(healthy readings, {condition: readings}) across every load level.

    Healthy rows from all three load levels are pooled: EdgeSense trains on
    "normal operation" as a whole, and load is an operating condition rather
    than a fault.
    """
    import sys

    recordings = discover(root)
    if not recordings:
        raise DatasetError(f"no complete recordings found under {root}")

    healthy: list[np.ndarray] = []
    faults: dict[str, list[np.ndarray]] = {}
    for rec in recordings:
        if progress:
            print(f"  reading {rec}", file=sys.stderr)
        rows = load_recording(root, rec, sample_rate=sample_rate,
                              reading_rate=reading_rate)
        if rec.is_healthy:
            healthy.append(rows)
        else:
            faults.setdefault(rec.condition, []).append(rows)

    if not healthy:
        raise DatasetError("no Normal recordings found — cannot train a baseline")
    return (np.vstack(healthy),
            {k: np.vstack(v) for k, v in sorted(faults.items())})
