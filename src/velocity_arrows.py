"""Velocity arrows from actual displacements between recorded frames.

Brownian motion has no instantaneous velocity: these are interval averages.
Arrow lengths are scaled and capped for readability; directions are preserved.
"""

import numpy as np


def motion_velocities(times, tracks):
    times = np.asarray(times, dtype=float)
    tracks = np.asarray(tracks, dtype=float)
    if (times.ndim != 1 or tracks.ndim != 3 or tracks.shape[0] != len(times)
            or tracks.shape[-1] != 2 or not np.isfinite(times).all()
            or not np.isfinite(tracks).all() or (np.diff(times) <= 0).any()):
        raise ValueError("Need finite positions and strictly increasing frame times")
    velocities = np.zeros_like(tracks)
    if len(times) > 1:
        velocities[:-1] = np.diff(tracks, axis=0) / np.diff(times)[:, None, None]
        velocities[-1] = velocities[-2]
    return velocities


def arrow_components(velocities):
    arrows = .35 * np.asarray(velocities)
    lengths = np.linalg.norm(arrows, axis=-1, keepdims=True)
    arrows *= np.minimum(1., .65 / np.maximum(lengths, 1e-12))
    stationary = np.linalg.norm(velocities, axis=-1) == 0
    return (np.ma.masked_where(stationary, arrows[..., 0]),
            np.ma.masked_where(stationary, arrows[..., 1]))


def add_velocity_arrows(ax, positions, velocities):
    return ax.quiver(positions[:, 0], positions[:, 1], *arrow_components(velocities),
                     angles="xy", scale_units="xy", scale=1, color="#111827",
                     edgecolor="white", linewidth=.45, width=.007,
                     headwidth=3.8, headlength=4.8, zorder=10)
