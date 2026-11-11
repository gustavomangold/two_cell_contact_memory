"""Paper Eq. (4) on a saved configuration, using neighbors within 1.1 diameters.

The first half of the disks are green. Isolated disks have no defined index and
are excluded, as in the authors' analysis. Each non-isolated disk has equal weight.
"""

import numpy as np


FINAL_METRICS = ("demixing_index", "demixing_index_green", "demixing_index_red",
                 "neighbor_coverage", "neighbor_coverage_green", "neighbor_coverage_red")


def snapshot_demixing(positions, box_size=None):
    """Return raw M and non-isolated fractions for all, green and red disks."""
    positions = np.asarray(positions, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 2
            or len(positions) % 2 or not np.isfinite(positions).all()):
        raise ValueError("Expected finite (N, 2) positions with even N>=2")
    number = len(positions)
    green = np.arange(number) < number // 2
    if box_size is not None and (not np.isfinite(box_size) or box_size <= 0):
        raise ValueError("box_size must be finite and positive")
    count, same = np.zeros(number, dtype=int), np.zeros(number, dtype=int)
    # Bound memory use for the paper's 4,000 particles.
    for start in range(0, number, 256):
        stop = min(start + 256, number)
        displacement = positions[start:stop, None, :] - positions[None, :, :]
        if box_size is not None:
            displacement -= box_size * np.floor(displacement / box_size + .5)
        neighbors = np.sum(displacement ** 2, axis=-1) < 1.1 ** 2
        neighbors[np.arange(stop - start), np.arange(start, stop)] = False
        count[start:stop] = neighbors.sum(axis=1)
        same[start:stop] = (neighbors & (green[start:stop, None] == green[None, :])).sum(axis=1)
    index = np.divide(same, count, out=np.full(number, np.nan), where=count > 0)
    groups = (np.ones(number, dtype=bool), green, ~green)
    means, coverage = [], []
    for group in groups:
        eligible = group & (count > 0)
        means.append(float(index[eligible].mean()) if eligible.any() else np.nan)
        coverage.append(float((count[group] > 0).mean()))
    return np.array(means + coverage)
