"""Equal green/red populations in the same soft circular confinement.

No reorientation: unlike contact starts an expired green memory timer. Other
contacts do not refresh it. Lengths use disk diameters, as in two_disks.py.
"""

from collections import namedtuple

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ModuleNotFoundError:
    HAS_NUMBA = False

    def njit(**kwargs):
        return lambda function: function


Model = namedtuple("Model", "arena speed repulsion wall diffusion angular persistent tau dt mobility warmup steps save_every")
METRICS = ("contact_fraction", "red_contact_fraction", "unlike_contacts_per_green",
           "persistent_fraction", "demixing_index", "neighbor_coverage",
           "wall_fraction_green", "wall_fraction_red")
NEIGHBOR_DISTANCE = 1.1
SKIN = .3


def make_model(settings, arena, tau):
    if settings["REORIENT_ON_CONTACT"]:
        raise ValueError("This size study isolates memory without reorientation")
    return Model(float(arena), float(settings["SPEED"]), float(settings["REPULSION"]),
                 float(settings["WALL_REPULSION"]), float(settings["TRANSLATIONAL_DIFFUSION"]),
                 float(settings["ANGULAR_DIFFUSION"]), float(settings["PERSISTENT_DIFFUSION"]),
                 float(tau), float(settings["DT"]), float(settings["MU"]),
                 int(settings["STEADY_STATE_STEPS"]), round(settings["DURATION"] / settings["DT"]),
                 int(settings["SAVE_EVERY"]))


def initial_state(number, arena, rng):
    if number < 2 or number % 2:
        raise ValueError("Use an even particle count >= 2 for equal populations")
    if number == 2:
        # Preserve the exact initial state and RNG stream of the original pair.
        return np.array([[-1., 0.], [1., 0.]]), np.array([0., np.pi])
    positions = np.empty((number, 2))
    limit = arena / 2 - .5
    for i in range(number):
        for _ in range(10_000):
            radius, angle = limit * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
            point = radius * np.array([np.cos(angle), np.sin(angle)])
            if not i or (np.sum((positions[:i] - point) ** 2, axis=1) >= 1.).all():
                positions[i] = point
                break
        else:
            raise ValueError("Could not place non-overlapping disks; increase the arena")
    # Shuffle before assigning the first half green, avoiding placement-order bias.
    positions = positions[rng.permutation(number)]
    return positions, rng.uniform(0, 2 * np.pi, number)


@njit(cache=True)
def _build_pairs(positions, pairs, cutoff):
    count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx, dy = positions[i, 0] - positions[j, 0], positions[i, 1] - positions[j, 1]
            if dx * dx + dy * dy < cutoff * cutoff:
                pairs[count, 0], pairs[count, 1] = i, j
                count += 1
    return count


def _simulate_python(rng, positions, angles, model, edges, record=False, neighbor_list=True):
    """Readable time loop; compiled below when Numba is available."""
    positions, angles = positions.copy(), angles.copy()
    number, greens = len(positions), len(positions) // 2
    memory = np.zeros(greens)
    forces = np.zeros_like(positions)
    contact = np.zeros(number, dtype=np.bool_)
    neighbors, same = np.zeros(number, dtype=np.int64), np.zeros(number, dtype=np.int64)
    sums = np.zeros((len(edges) - 1, len(METRICS)))
    total = model.warmup + model.steps
    frames = total // model.save_every + 2 if record else 1
    times, tracks = np.zeros(frames), np.empty((frames, number, 2))
    states = np.zeros((frames, greens), dtype=np.bool_)
    tracks[0] = positions
    frame, window = 1, 0
    limit = model.arena / 2 - .5

    # A Verlet list is rebuilt before motion can bring an omitted pair into range.
    pairs = np.empty((number * (number - 1) // 2, 2), dtype=np.int64)
    cutoff = NEIGHBOR_DISTANCE + SKIN if neighbor_list else np.inf
    pair_count = _build_pairs(positions, pairs, cutoff)
    reference = positions.copy()
    for step in range(1, total + 1):
        for i in range(greens):
            memory[i] = max(0., memory[i] - model.dt)
        if neighbor_list:
            rebuild = False
            for i in range(number):
                dx, dy = positions[i, 0] - reference[i, 0], positions[i, 1] - reference[i, 1]
                if dx * dx + dy * dy >= (SKIN / 2) ** 2:
                    rebuild = True
                    break
            if rebuild:
                pair_count = _build_pairs(positions, pairs, cutoff)
                reference[:, :] = positions

        forces[:, :] = 0.
        contact[:] = False
        neighbors[:], same[:] = 0, 0
        unlike_pairs = 0
        for pair in range(pair_count):
            i, j = pairs[pair, 0], pairs[pair, 1]
            dx, dy = positions[i, 0] - positions[j, 0], positions[i, 1] - positions[j, 1]
            squared = dx * dx + dy * dy
            if squared >= NEIGHBOR_DISTANCE ** 2:
                continue
            neighbors[i] += 1
            neighbors[j] += 1
            unlike = i < greens and j >= greens
            if not unlike:
                same[i] += 1
                same[j] += 1
            distance = np.sqrt(squared)
            if distance < 1.:
                nx, ny = (dx / distance, dy / distance) if distance > 1e-12 else (1., 0.)
                strength = model.repulsion * max(0., 1. - distance)
                forces[i, 0] += strength * nx
                forces[i, 1] += strength * ny
                forces[j, 0] -= strength * nx
                forces[j, 1] -= strength * ny
                if unlike:
                    contact[i], contact[j] = True, True
                    unlike_pairs += 1
                    if memory[i] == 0.:
                        memory[i] = model.tau

        observed = step - model.warmup - 1
        if observed >= 0:
            if observed == edges[window + 1]:
                window += 1
            sums[window, 2] += unlike_pairs
        for i in range(number):
            x, y = positions[i, 0], positions[i, 1]
            radius = np.sqrt(x * x + y * y)
            outside = radius > limit
            if outside:
                strength = model.wall * max(0., radius - limit)
                forces[i, 0] += strength * (-x / radius)
                forces[i, 1] += strength * (-y / radius)
            if observed >= 0:
                if i < greens:
                    sums[window, 0] += 1. if contact[i] else 0.
                    sums[window, 3] += 1. if memory[i] > 0 else 0.
                    sums[window, 6] += 1. if outside and model.wall > 0 else 0.
                else:
                    sums[window, 1] += 1. if contact[i] else 0.
                    sums[window, 7] += 1. if outside and model.wall > 0 else 0.
                if neighbors[i]:
                    sums[window, 4] += same[i] / neighbors[i]
                    sums[window, 5] += 1

        for i in range(number):
            positions[i, 0] += (model.speed * np.cos(angles[i]) + model.mobility * forces[i, 0]) * model.dt
            positions[i, 1] += (model.speed * np.sin(angles[i]) + model.mobility * forces[i, 1]) * model.dt
        if model.diffusion > 0:
            positions += np.sqrt(2 * model.diffusion * model.dt) * rng.normal(size=(number, 2))
        noise = rng.normal(size=number)
        for i in range(number):
            diffusion = model.persistent if i < greens and memory[i] > 0 else model.angular
            angles[i] += np.sqrt(2 * diffusion * model.dt) * noise[i]
        if record and (step % model.save_every == 0 or step == model.steps):
            times[frame], tracks[frame] = step * model.dt, positions
            for i in range(greens):
                states[frame, i] = memory[i] > 0
            frame += 1
    return sums, times[:frame], tracks[:frame], states[:frame], positions, angles, memory


simulate_kernel = njit(cache=True)(_simulate_python)


def normalize(sums, number, observations):
    sums = np.asarray(sums, dtype=float)
    values = sums.copy()
    greens = number // 2
    for column in (0, 1, 2, 3, 6, 7):
        values[..., column] /= observations * greens
    values[..., 4] = np.divide(sums[..., 4], sums[..., 5],
                              out=np.full(values[..., 4].shape, np.nan), where=sums[..., 5] > 0)
    values[..., 5] /= observations * number
    return values
