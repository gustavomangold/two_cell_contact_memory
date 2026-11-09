"""Compact sweep storage: block means in logarithmically spaced time windows.

Window counts preserve the uniformly sampled run mean. Tau values are unchanged.
"""

import csv
from pathlib import Path

import numpy as np


RAW_HEADER = ["L", "tau", "sample", "seed", "step", "time", "separation", "separation_over_L"]
HEADER = ["L", "tau", "sample", "seed", "start_step", "end_step", "time",
          "observations", "mean_separation", "separation_over_L"]
LOG_WINDOWS = 64
CHUNK_ROWS = 250_000


def window_edges(count):
    if count < 1:
        raise ValueError("Cannot average an empty run")
    return np.unique(np.r_[0, np.ceil(np.geomspace(1, count, LOG_WINDOWS)).astype(np.int64)])


def mean_rows(distances, steps, times, normalized):
    edges = window_edges(len(distances))
    for start, stop in zip(edges[:-1], edges[1:]):
        yield [int(steps[start]), int(steps[stop - 1]), float(times[stop - 1]),
               int(stop - start), float(distances[start:stop].mean()),
               float(normalized[start:stop].mean())]


def compact_sample(distances, length, first_step, stride, dt):
    steps = first_step + np.arange(len(distances)) * stride
    return list(mean_rows(distances, steps, (steps - 1) * dt, distances / length))


def compact_file(path, expected_count=None):
    """Atomically replace a raw CSV after validating all samples and their means."""
    path = Path(path)
    before = path.stat()
    temporary = path.with_name(path.name + ".compact.tmp")
    samples, rows_per_sample, initial_seed = 0, None, None

    def write_sample(data, writer):
        nonlocal samples, rows_per_sample, initial_seed
        length, tau, sample, seed = data[0, :4]
        if initial_seed is None:
            initial_seed = seed
            rows_per_sample = len(data)
        if ((expected_count is not None and len(data) != expected_count)
                or len(data) != rows_per_sample or sample != samples + 1
                or seed != initial_seed + samples or not np.isfinite(data).all()
                or not (data[:, :4] == data[0, :4]).all()
                or not np.isclose(data[:, 7], data[:, 6] / length).all()
                or (np.diff(data[:, 4]) <= 0).any()):
            raise ValueError(f"Incomplete or inconsistent sample {sample:g} in {path}")
        if len(data) > 1 and not (np.diff(data[:, 4]) == data[1, 4] - data[0, 4]).all():
            raise ValueError(f"Nonuniform original sampling in {path}")
        rows = list(mean_rows(data[:, 6], data[:, 4], data[:, 5], data[:, 7]))
        weights = np.array([row[3] for row in rows])
        # Check preservation before allowing the atomic replacement.
        for column, original in [(4, data[:, 6]), (5, data[:, 7])]:
            averaged = np.average([row[column] for row in rows], weights=weights)
            if not np.isclose(averaged, original.mean(), rtol=1e-12, atol=1e-14):
                raise ValueError(f"Mean changed during compaction of {path}")
        writer.writerows([[length, tau, int(sample), int(seed), *row] for row in rows])
        samples += 1

    try:
        with path.open("rb") as source, temporary.open("w", newline="") as target:
            header = source.readline().decode().strip().split(",")
            if header == HEADER:
                return before.st_size, before.st_size
            if header != RAW_HEADER:
                raise ValueError(f"Unexpected CSV header: {path}")
            writer = csv.writer(target)
            writer.writerow(HEADER)
            pending = np.empty((0, len(RAW_HEADER)))
            while True:
                position = source.tell()
                if not source.readline():
                    break
                source.seek(position)
                chunk = np.loadtxt(source, delimiter=",", max_rows=CHUNK_ROWS, ndmin=2)
                if chunk.shape[1] != len(RAW_HEADER):
                    raise ValueError(f"Incomplete raw row in {path}")
                data = np.concatenate((pending, chunk))
                boundaries = np.flatnonzero(np.diff(data[:, 2])) + 1
                start = 0
                for end in boundaries:
                    write_sample(data[start:end], writer)
                    start = end
                pending = data[start:].copy()
            if len(pending):
                write_sample(pending, writer)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"{path} changed during conversion; original kept")
        temporary.replace(path)
        return before.st_size, path.stat().st_size
    finally:
        temporary.unlink(missing_ok=True)


def completed_means(path, length, tau, seed, first_step, stride, last_step, samples, dt):
    """Resume at the last complete run, trimming only an interrupted trailing run."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = (last_step - first_step) // stride + 1
    edges = window_edges(count)
    completed, window = 0, 0
    with path.open("rb+") as file:
        if file.readline().decode().strip().split(",") != HEADER:
            raise ValueError(f"Unexpected compact header in {path}")
        keep = file.tell()
        end = path.stat().st_size
        while line := file.readline():
            if not line.endswith(b"\n"):
                break
            try:
                row = [float(value) for value in line.decode().strip().split(",")]
                if len(row) != len(HEADER):
                    raise ValueError("incomplete row")
            except (ValueError, UnicodeDecodeError):
                if file.tell() != end:
                    raise ValueError(f"Invalid row in {path}")
                break
            start, stop = edges[window:window + 2]
            start_step = first_step + start * stride
            end_step = first_step + (stop - 1) * stride
            if (row[:6] != [length, tau, completed + 1, seed + completed, start_step, end_step]
                    or completed >= samples or not np.isfinite(row).all()
                    or not np.isclose(row[6], (end_step - 1) * dt)
                    or row[7] != stop - start or not np.isclose(row[9], row[8] / length)):
                raise ValueError(f"{path} does not match current settings")
            window += 1
            if window == len(edges) - 1:
                completed += 1
                window = 0
                keep = file.tell()
        if keep != end:
            file.truncate(keep)
    return completed


def _recover_raw(path, length, tau, seed, first, stride, last, samples, dt):
    """Keep whole legacy samples and trim an interrupted trailing sample."""
    rows_per_sample = (last - first) // stride + 1
    completed, count = 0, 0
    with path.open("rb+") as file:
        if file.readline().decode().strip().split(",") != RAW_HEADER:
            raise ValueError(f"Unexpected raw header in {path}")
        keep, end = file.tell(), path.stat().st_size
        while line := file.readline():
            if not line.endswith(b"\n"):
                break
            try:
                row = [float(value) for value in line.decode().strip().split(",")]
                if len(row) != len(RAW_HEADER):
                    raise ValueError("incomplete row")
            except (ValueError, UnicodeDecodeError):
                if file.tell() != end:
                    raise ValueError(f"Invalid row in {path}")
                break
            step = first + count * stride
            if (row[:5] != [length, tau, completed + 1, seed + completed, step]
                    or completed >= samples or not np.isfinite(row).all()
                    or not np.isclose(row[5], (step - 1) * dt)
                    or not np.isclose(row[7], row[6] / length)):
                raise ValueError(f"{path} does not match current settings")
            count += 1
            if count == rows_per_sample:
                completed, count, keep = completed + 1, 0, file.tell()
        if keep != end:
            file.truncate(keep)


def completed_samples(path, length, tau, seed, first, stride, last, samples, dt):
    """Resume compact CSVs, converting legacy raw files when necessary."""
    if path.exists() and path.stat().st_size:
        with path.open() as file:
            raw = file.readline().strip().split(",") == RAW_HEADER
        if raw:
            expected = (last - first) // stride + 1
            try:
                compact_file(path, expected_count=expected)
            except ValueError:
                _recover_raw(path, length, tau, seed, first, stride, last, samples, dt)
                compact_file(path, expected_count=expected)
    return completed_means(path, length, tau, seed, first, stride, last, samples, dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compact legacy sweep CSVs without rerunning.")
    parser.add_argument("data_dir", type=Path, help="Search recursively for L*_tau*.csv")
    args = parser.parse_args()
    for path in sorted(args.data_dir.rglob("L*_tau*.csv")):
        with path.open() as file:
            raw = file.readline().strip().split(",") == RAW_HEADER
        if raw:
            old, new = compact_file(path)
            print(f"{path}: {old / 1e6:.2f} MB -> {new / 1e6:.2f} MB", flush=True)
