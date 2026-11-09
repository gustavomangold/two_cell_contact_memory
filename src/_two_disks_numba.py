"""Optional compiled time loop. Parameters and the NumPy RNG are passed explicitly.

No fast-math or threaded random draws: preserve the original update order.
The readable NumPy implementation remains in two_disks.py as the fallback.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def simulate_kernel(rng, arena, speed, repulsion, angular_diffusion,
                    persistence_time, dt, duration, steady_steps,
                    step_separation, save_every, mobility,
                    translational_diffusion=0.0, persistent_diffusion=0.0,
                    reorient_on_contact=True, wall_repulsion=30.0):
    observation_steps = round(duration / dt)
    total_steps = observation_steps + steady_steps
    # Allocate once, rather than creating small arrays on every time step.
    times = np.empty(total_steps // save_every + 2)
    tracks = np.empty((len(times), 2, 2))
    persistent = np.empty(len(times), dtype=np.bool_)
    separations = np.empty(total_steps // step_separation + 1)
    positions = np.array([[-1.0, 0.0], [1.0, 0.0]])
    angles = np.array([0.0, np.pi])
    forces = np.zeros((2, 2))
    persistent_left = 0.0
    center_limit = arena / 2 - 0.5
    times[0], tracks[0], persistent[0] = 0.0, positions, False
    frame, sample = 1, 0

    for step in range(1, total_steps + 1):
        persistent_left = max(0.0, persistent_left - dt)
        dx = positions[0, 0] - positions[1, 0]
        dy = positions[0, 1] - positions[1, 1]
        distance = np.sqrt(dx * dx + dy * dy)
        forces[:, :] = 0.0

        if distance < 1.0:
            nx, ny = (dx / distance, dy / distance) if distance > 1e-12 else (1., 0.)
            strength = repulsion * max(0.0, 1.0 - distance)
            forces[0, 0], forces[0, 1] = strength * nx, strength * ny
            forces[1, 0], forces[1, 1] = -forces[0, 0], -forces[0, 1]
            if reorient_on_contact:
                away = np.arctan2(ny, nx)
                angles[0], angles[1] = away, away + np.pi
            if persistent_left == 0.0:
                persistent_left = persistence_time

        for i in range(2):
            x, y = positions[i, 0], positions[i, 1]
            radius = np.sqrt(x * x + y * y)
            if radius > center_limit:
                strength = wall_repulsion * max(0.0, radius - center_limit)
                forces[i, 0] += strength * (-x / radius)
                forces[i, 1] += strength * (-y / radius)
            positions[i, 0] += (speed * np.cos(angles[i]) + mobility * forces[i, 0]) * dt
            positions[i, 1] += (speed * np.sin(angles[i]) + mobility * forces[i, 1]) * dt

        if translational_diffusion > 0:
            position_noise = rng.normal(size=(2, 2))
            positions += np.sqrt(2 * translational_diffusion * dt) * position_noise

        # Use the same Generator and draw order as the NumPy implementation.
        noise = rng.normal(size=2)
        green_diffusion = persistent_diffusion if persistent_left > 0 else angular_diffusion
        angles[0] += np.sqrt(2 * green_diffusion * dt) * noise[0]
        angles[1] += np.sqrt(2 * angular_diffusion * dt) * noise[1]

        if step >= steady_steps and step % step_separation == 0:
            separations[sample] = distance
            sample += 1
        if step % save_every == 0 or step == observation_steps:
            times[frame] = step * dt
            tracks[frame] = positions
            persistent[frame] = persistent_left > 0
            frame += 1

    return times[:frame], tracks[:frame], persistent[:frame], separations[:sample]
