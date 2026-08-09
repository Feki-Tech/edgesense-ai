"""Loading and reading-reduction for the KAIST rotating-machine dataset.

The published archives are ~4.3 GB, so these build small synthetic .mat/.tdms
files in the documented layout instead. That covers the parts that are ours —
filename parsing, channel discovery, the RMS/mean window reduction and the
split assembly — against signals whose correct answer is known analytically.
It does not prove the real files parse; `make benchmark-rotating` does that,
and docs/BENCHMARK.md says so.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from ml.rotating import (CURRENT_CHANNELS, TEMPERATURE_CHANNELS,
                         VIBRATION_CHANNELS, DatasetError, Recording,
                         discover, load_dataset, parse_stem,
                         read_mat_columns, read_tdms_columns,
                         reduce_to_readings)

nptdms = pytest.importorskip("nptdms", reason="benchmark extra not installed")
from nptdms import ChannelObject, TdmsWriter  # noqa: E402

FS = 1_000.0  # test sample rate; the real set is 25.6 kHz
RATE = 2.0    # 2 Hz readings -> 500-sample windows


def _write_mat(path, n, amplitude=1.0):
    savemat(str(path), {"Time Stamp": np.arange(n) / FS,
                        **{c: np.full(n, amplitude) for c in VIBRATION_CHANNELS}})


def _write_tdms(path, n, temp=40.0, current=5.0):
    channels = [ChannelObject("data", "Time Stamp", np.arange(n) / FS)]
    channels += [ChannelObject("data", c, np.full(n, temp))
                 for c in TEMPERATURE_CHANNELS]
    channels += [ChannelObject("data", c, np.full(n, current))
                 for c in CURRENT_CHANNELS]
    with TdmsWriter(str(path)) as w:
        w.write_segment(channels)


@pytest.fixture()
def dataset(tmp_path):
    """A miniature dataset in the published layout: 2 loads x (Normal + 1 fault)."""
    vib, tc = tmp_path / "vibration", tmp_path / "current,temp"
    vib.mkdir()
    tc.mkdir()
    n = int(FS * 2)  # 2 seconds -> 4 windows at 2 Hz
    for load in (0, 2):
        for cond, amp, temp in (("Normal", 1.0, 40.0), ("BPFI_03", 3.0, 55.0)):
            _write_mat(vib / f"{load}Nm_{cond}.mat", n, amplitude=amp)
            _write_tdms(tc / f"{load}Nm_{cond}.tdms", n, temp=temp)
    return tmp_path


# --- naming --------------------------------------------------------------

@pytest.mark.parametrize("stem,load,cond,family,healthy", [
    ("0Nm_Normal", 0, "Normal", "healthy", True),
    ("2Nm_BPFI_03", 2, "BPFI_03", "bearing inner race fault", False),
    ("4Nm_BPFO_30", 4, "BPFO_30", "bearing outer race fault", False),
    ("0Nm_Misalign_05", 0, "Misalign_05", "shaft misalignment", False),
    ("4Nm_Unbalance_3318mg", 4, "Unbalance_3318mg", "rotor unbalance", False),
])
def test_parse_stem(stem, load, cond, family, healthy) -> None:
    rec = parse_stem(stem)
    assert (rec.load_nm, rec.condition) == (load, cond)
    assert rec.family == family
    assert rec.is_healthy is healthy
    assert rec.stem == stem


def test_parse_stem_rejects_junk() -> None:
    with pytest.raises(DatasetError, match="unrecognised"):
        parse_stem("not-a-recording")


# --- readers -------------------------------------------------------------

def test_reads_mat_columns(tmp_path) -> None:
    _write_mat(tmp_path / "0Nm_Normal.mat", 100, amplitude=2.0)
    cols = read_mat_columns(tmp_path / "0Nm_Normal.mat")
    assert set(cols) == set(VIBRATION_CHANNELS)
    assert cols["x_direction_housing_A"].shape == (100,)
    assert np.allclose(cols["x_direction_housing_A"], 2.0)


def test_reads_mat_from_bare_matrix(tmp_path) -> None:
    """Falls back to a single 5-column matrix when named columns are absent."""
    n = 50
    matrix = np.column_stack([np.arange(n)] + [np.full(n, i + 1.0) for i in range(4)])
    savemat(str(tmp_path / "m.mat"), {"data": matrix})
    cols = read_mat_columns(tmp_path / "m.mat")
    assert np.allclose(cols["x_direction_housing_A"], 1.0)
    assert np.allclose(cols["y_direction_housing_B"], 4.0)


def test_reads_tdms_channels(tmp_path) -> None:
    _write_tdms(tmp_path / "0Nm_Normal.tdms", 80, temp=41.5, current=6.25)
    cols = read_tdms_columns(tmp_path / "0Nm_Normal.tdms")
    assert set(cols) == set(TEMPERATURE_CHANNELS + CURRENT_CHANNELS)
    assert np.allclose(cols["Temperature_housing_A"], 41.5)
    assert np.allclose(cols["U-phase"], 6.25)


def test_missing_tdms_channel_is_reported(tmp_path) -> None:
    with TdmsWriter(str(tmp_path / "partial.tdms")) as w:
        w.write_segment([ChannelObject("data", "U-phase", np.zeros(10))])
    with pytest.raises(DatasetError, match="missing channels"):
        read_tdms_columns(tmp_path / "partial.tdms")


# --- reduction -----------------------------------------------------------

def test_reduction_shapes_and_values() -> None:
    """Constant signals reduce to their own value: RMS(c) = c, mean(c) = c."""
    n = int(FS * 3)  # 3 s -> 6 windows at 2 Hz
    vib = {c: np.full(n, 2.0) for c in VIBRATION_CHANNELS}
    tc = {**{c: np.full(n, 45.0) for c in TEMPERATURE_CHANNELS},
          **{c: np.full(n, 7.0) for c in CURRENT_CHANNELS}}

    out = reduce_to_readings(vib, tc, sample_rate=FS, reading_rate=RATE)

    assert out.shape == (6, 3)
    assert np.allclose(out[:, 0], 2.0)   # vibration RMS
    assert np.allclose(out[:, 1], 45.0)  # temperature mean
    assert np.allclose(out[:, 2], 7.0)   # current RMS


def test_vibration_is_rms_not_mean() -> None:
    """A zero-mean oscillation must survive as amplitude, not average to 0."""
    n = int(FS)
    t = np.arange(n) / FS
    wave = 3.0 * np.sin(2 * np.pi * 50 * t)  # 50 Hz, amplitude 3 -> RMS 3/sqrt2
    vib = {c: wave for c in VIBRATION_CHANNELS}
    tc = {**{c: np.full(n, 40.0) for c in TEMPERATURE_CHANNELS},
          **{c: np.full(n, 5.0) for c in CURRENT_CHANNELS}}

    out = reduce_to_readings(vib, tc, sample_rate=FS, reading_rate=RATE)
    assert np.allclose(out[:, 0], 3.0 / np.sqrt(2), rtol=1e-2)


def test_ragged_channel_lengths_truncate_to_common_windows() -> None:
    """Vibration and temp/current are recorded separately; lengths can differ."""
    vib = {c: np.ones(int(FS * 2)) for c in VIBRATION_CHANNELS}          # 4 windows
    tc = {**{c: np.full(int(FS * 1.5), 40.0) for c in TEMPERATURE_CHANNELS},
          **{c: np.full(int(FS * 1.5), 5.0) for c in CURRENT_CHANNELS}}  # 3 windows

    out = reduce_to_readings(vib, tc, sample_rate=FS, reading_rate=RATE)
    assert out.shape == (3, 3)


def test_recording_shorter_than_one_window_is_an_error() -> None:
    vib = {c: np.ones(10) for c in VIBRATION_CHANNELS}
    tc = {**{c: np.full(10, 40.0) for c in TEMPERATURE_CHANNELS},
          **{c: np.full(10, 5.0) for c in CURRENT_CHANNELS}}
    with pytest.raises(DatasetError, match="shorter than one"):
        reduce_to_readings(vib, tc, sample_rate=FS, reading_rate=RATE)


# --- assembly ------------------------------------------------------------

def test_discover_pairs_mat_and_tdms(dataset) -> None:
    recs = discover(dataset)
    assert [r.stem for r in recs] == ["0Nm_BPFI_03", "0Nm_Normal",
                                      "2Nm_BPFI_03", "2Nm_Normal"]


def test_discover_ignores_unpaired_files(dataset) -> None:
    """A .mat with no matching .tdms cannot produce readings, so it is skipped."""
    _write_mat(dataset / "vibration" / "4Nm_Normal.mat", int(FS))
    assert "4Nm_Normal" not in {r.stem for r in discover(dataset)}


def test_load_dataset_pools_healthy_across_loads(dataset) -> None:
    healthy, faults = load_dataset(dataset, sample_rate=FS, reading_rate=RATE)

    # 2 loads x 2 s x 2 Hz = 8 healthy readings
    assert healthy.shape == (8, 3)
    assert list(faults) == ["BPFI_03"]
    assert faults["BPFI_03"].shape == (8, 3)

    # the fault rows really are separable on the fixture's amplitudes
    assert healthy[:, 0].mean() < faults["BPFI_03"][:, 0].mean()
    assert healthy[:, 1].mean() < faults["BPFI_03"][:, 1].mean()


def test_load_dataset_needs_a_baseline(tmp_path) -> None:
    vib, tc = tmp_path / "vibration", tmp_path / "current,temp"
    vib.mkdir()
    tc.mkdir()
    _write_mat(vib / "0Nm_BPFI_03.mat", int(FS))
    _write_tdms(tc / "0Nm_BPFI_03.tdms", int(FS))
    with pytest.raises(DatasetError, match="no Normal recordings"):
        load_dataset(tmp_path, sample_rate=FS, reading_rate=RATE)


def test_unrecognisable_root_is_reported(tmp_path) -> None:
    with pytest.raises(DatasetError, match="does not look like"):
        discover(tmp_path)


def test_flat_layout_is_accepted(tmp_path) -> None:
    """Both archives unpacked into one directory still works."""
    _write_mat(tmp_path / "0Nm_Normal.mat", int(FS))
    _write_tdms(tmp_path / "0Nm_Normal.tdms", int(FS))
    assert [r.stem for r in discover(tmp_path)] == ["0Nm_Normal"]
    assert load_dataset(tmp_path, sample_rate=FS, reading_rate=RATE)[0].shape == (2, 3)


def test_recording_dataclass_is_hashable() -> None:
    assert len({Recording(0, "Normal"), Recording(0, "Normal")}) == 1


# --- benchmark end to end ------------------------------------------------

@pytest.fixture()
def separable_dataset(tmp_path):
    """Enough readings to actually train, with faults that differ from healthy."""
    vib, tc = tmp_path / "vibration", tmp_path / "current,temp"
    vib.mkdir()
    tc.mkdir()
    rng = np.random.default_rng(0)

    def write(stem, secs, amp, temp, cur):
        n = int(FS * secs)
        t = np.arange(n) / FS
        savemat(str(vib / f"{stem}.mat"), {
            "Time Stamp": t,
            **{c: amp * np.sin(2 * np.pi * 50 * t) + rng.normal(0, amp * 0.1, n)
               for c in VIBRATION_CHANNELS}})
        chans = [ChannelObject("d", "Time Stamp", t)]
        chans += [ChannelObject("d", c, temp + rng.normal(0, 0.3, n))
                  for c in TEMPERATURE_CHANNELS]
        chans += [ChannelObject("d", c, cur + rng.normal(0, 0.05, n))
                  for c in CURRENT_CHANNELS]
        with TdmsWriter(str(tc / f"{stem}.tdms")) as w:
            w.write_segment(chans)

    for load in (0, 2):
        write(f"{load}Nm_Normal", 60, 1.0, 40.0 + load, 5.0)
        write(f"{load}Nm_BPFI_30", 30, 3.0, 48.0 + load, 6.0)
    return tmp_path


def test_benchmark_runs_and_reports(separable_dataset) -> None:
    from ml.benchmark_rotating import run_benchmark, to_markdown

    healthy, faults = load_dataset(separable_dataset, sample_rate=FS,
                                   reading_rate=RATE)
    results = run_benchmark(healthy, faults, "sklearn", seed=7, epochs=200)

    assert set(results["conditions"]) == {"BPFI_30"}
    r = results["conditions"]["BPFI_30"]
    assert r["family"] == "bearing inner race fault"
    # the fault is unmistakable in this fixture; a broken pipeline would not see it
    assert r["auc"] > 0.9
    assert 0.0 <= results["healthy"]["model_fp"] <= 0.2

    md = to_markdown(results, {"seed": 7, "date": "2026-01-01",
                               "sample_rate": FS, "reading_rate": RATE})
    assert "BPFI_30" in md and "bearing inner race fault" in md
    assert "ROC-AUC" in md
