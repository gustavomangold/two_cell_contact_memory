"""Optional post-separation counters. No dynamics or random draws are changed."""

from collections import namedtuple

import numpy as np

try:
    from numba import njit, types
    from numba.typed import List
except ModuleNotFoundError:
    List = None

    def njit(**kwargs):
        return lambda function: function


PostRecorder = namedtuple("PostRecorder", "window state events fixed")
POST_COLUMNS = ["time", "duration", "memory_at_start", "green_displacement",
                "max_green_displacement", "max_gap_increase", "final_gap_change",
                "end_reason", "near_wall_at_start", "wall_reached", "initial_heading_cosine",
                "initial_pair_distance"]
FIXED_COLUMNS = ["time", "duration", "memory_at_start", "green_displacement", "gap_change",
                 "near_wall_at_start", "initial_heading_cosine", "recontacted", "wall_reached",
                 "initial_pair_distance", "complete"]
POST_STATE_DTYPE = [("active", "?"), ("start", "i8"), ("x", "f8"), ("y", "f8"),
                    ("distance", "f8"), ("memory", "f8"), ("heading", "f8"),
                    ("near_wall", "?"), ("wall", "?"), ("net", "f8"),
                    ("max_move", "f8"), ("max_gap", "f8"), ("gap_change", "f8")]


def make_post_recorder(window):
    events = List.empty_list(types.UniTuple(types.float64, 12)) if List is not None else []
    fixed = List.empty_list(types.UniTuple(types.float64, 11)) if List is not None else []
    return PostRecorder(window, np.zeros(1, dtype=POST_STATE_DTYPE), events, fixed)


@njit(cache=True)
def _close_excursion(recorder, index, reason):
    state, dt = recorder.post.state[0], recorder.parameters.dt
    recorder.post.events.append((state["start"] * dt, (index - state["start"]) * dt,
                                 state["memory"], state["net"], state["max_move"], state["max_gap"],
                                 state["gap_change"], float(reason),
                                 1. if state["near_wall"] else 0., 1. if state["wall"] else 0.,
                                 state["heading"], state["distance"]))
    state["active"] = False


@njit(cache=True)
def _append_fixed(recorder, start, end, complete):
    history = recorder.history
    reference, current = history[start % len(history)], history[end % len(history)]
    dx, dy = current[0] - reference[0], current[1] - reference[1]
    counts, dt = recorder.counts[0], recorder.parameters.dt
    recorder.post.fixed.append((start * dt, (end - start) * dt, reference[12],
                                np.sqrt(dx * dx + dy * dy), current[4] - reference[4],
                                reference[8], reference[10],
                                1. if counts["new_contacts"] > reference[13] else 0.,
                                1. if counts["wall_green"] + counts["wall_red"] > reference[14] else 0.,
                                reference[4], 1. if complete else 0.))


@njit(cache=True)
def observe_post(recorder, index, positions, angles, memory, distance, separated,
                 entered, near_wall, wall_reached):
    """Track one memory excursion, plus possibly overlapping fixed-time windows."""
    state = recorder.post.state[0]
    if state["active"]:
        dx, dy = positions[0, 0] - state["x"], positions[0, 1] - state["y"]
        state["net"] = np.sqrt(dx * dx + dy * dy)
        state["gap_change"] = distance - state["distance"]
        state["max_move"] = max(state["max_move"], state["net"])
        state["max_gap"] = max(state["max_gap"], state["gap_change"])
        state["wall"] = state["wall"] or wall_reached
        # Stop before a new contact can renew memory and extend this excursion.
        if entered:
            _close_excursion(recorder, index, 1)
        elif memory <= 0:
            _close_excursion(recorder, index, 0)

    current = recorder.history[index % len(recorder.history)]
    current[11] = 1. if separated else 0.
    if separated:
        dx, dy = positions[0, 0] - positions[1, 0], positions[0, 1] - positions[1, 1]
        u = (np.cos(angles[0]) * dx + np.sin(angles[0]) * dy) / distance
        current[8], current[10], current[12] = (1. if near_wall else 0.), u, memory
        current[13] = recorder.counts[0]["new_contacts"]
        # Prefix counts let many overlapping windows detect wall visits cheaply.
        current[14] = recorder.counts[0]["wall_green"] + recorder.counts[0]["wall_red"]
        if wall_reached:
            current[14] -= 1.
        if memory > 0:
            state["active"], state["start"] = True, index
            state["x"], state["y"] = positions[0, 0], positions[0, 1]
            state["distance"], state["memory"], state["heading"] = distance, memory, u
            state["near_wall"], state["wall"] = near_wall, wall_reached
            state["net"], state["max_move"], state["max_gap"], state["gap_change"] = 0., 0., 0., 0.

    start = index - recorder.post.window
    if start >= 0 and recorder.history[start % len(recorder.history), 11]:
        # This window always has the same length, even after recontact/expiry.
        _append_fixed(recorder, start, index, True)


@njit(cache=True)
def finish_post(recorder):
    last = recorder.parameters.steps - 1
    if recorder.post.state[0]["active"]:
        _close_excursion(recorder, last, 2)  # Recording ended; endpoint is censored.
    for start in range(max(0, last + 1 - recorder.post.window), last + 1):
        if recorder.history[start % len(recorder.history), 11]:
            _append_fixed(recorder, start, last, False)


def post_summary(events, fixed):
    complete = events[events[:, 7] != 2]
    observed_fixed = fixed[fixed[:, 10] == 1]
    average = lambda rows, column: rows[:, column].mean() if len(rows) else np.nan
    return dict(
        post_separation_active_fraction=np.mean(fixed[:, 2] > 0) if len(fixed) else np.nan,
        post_separation_complete_events=float(len(complete)),
        post_separation_censored_fraction=np.mean(events[:, 7] == 2) if len(events) else np.nan,
        post_separation_net_displacement=average(complete, 3),
        post_separation_max_displacement=average(complete, 4),
        post_separation_max_gap_increase=average(complete, 5),
        post_separation_duration=average(complete, 1),
        fixed_separation_complete_events=float(len(observed_fixed)),
        fixed_separation_censored_fraction=np.mean(fixed[:, 10] == 0) if len(fixed) else np.nan,
        fixed_separation_net_displacement=average(observed_fixed, 3),
        fixed_separation_gap_change=average(observed_fixed, 4),
        fixed_separation_recontact_fraction=average(observed_fixed, 7),
    )
