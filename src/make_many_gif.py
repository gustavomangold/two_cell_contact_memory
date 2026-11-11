"""Animate the size-study model; default: N=32, L=48, tau=0.001.

Save a GIF and a final-frame PNG without changing simulation defaults or sweeps.
Purple marks active green memory. Arrows follow displacement between shown frames.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import two_disks as simulation
from many_disks import HAS_NUMBA, initial_state, make_model, normalize, simulate_kernel
from sweep_size_scaling import arena_size
from velocity_arrows import motion_velocities, arrow_components


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--L", type=float, help="Default follows the chosen size-study scaling")
    parser.add_argument("--scaling", choices=["diameter", "constant-density"], default="diameter")
    parser.add_argument("--tau", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=simulation.SEED)
    parser.add_argument("--duration", type=float, default=simulation.DURATION)
    parser.add_argument("--dt", type=float, default=simulation.DT)
    parser.add_argument("--steady-state-steps", type=int, default=simulation.STEADY_STATE_STEPS)
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--arrow-scale", type=float, default=10., help="Visual enlargement only; arrow directions are unchanged")
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    length = args.L if args.L is not None else arena_size(args.N, scaling=args.scaling)
    if (args.N < 2 or args.N % 2 or not np.isfinite(length) or length <= 1
            or not all(np.isfinite(value) for value in [args.tau, args.dt, args.duration, args.arrow_scale])
            or args.tau < 0 or min(args.dt, args.duration, args.arrow_scale) <= 0
            or min(args.seed, args.steady_state_steps) < 0 or min(args.save_every, args.fps) < 1
            or args.max_frames < 2 or round(args.duration / args.dt) < 1):
        parser.error("Require even N>=2, L>1, positive times/frame settings and nonnegative tau/seed/warm-up")
    output = args.output or ROOT / "figures" / "size_scaling" / f"disks_N{args.N}_L{length:g}_tau{args.tau:g}.gif"
    if output.suffix.lower() != ".gif":
        parser.error("output must have a .gif extension")
    settings = {name: value for name, value in vars(simulation).items()
                if name.isupper() and isinstance(value, (int, float))}
    settings.update(ARENA_DIAMETER=length, PERSISTENCE_TIME=args.tau, SEED=args.seed,
                    DT=args.dt, DURATION=args.duration, STEADY_STATE_STEPS=args.steady_state_steps,
                    SAVE_EVERY=args.save_every, REORIENT_ON_CONTACT=False)
    model = make_model(settings, length, args.tau)
    rng = np.random.default_rng(args.seed)
    positions, angles = initial_state(args.N, length, rng)
    print(f"Simulating N={args.N}, L={length:g}, tau={args.tau:g}, seed={args.seed}; "
          f"backend={'Numba' if HAS_NUMBA else 'NumPy'}", flush=True)
    sums, times, tracks, states, final, _, memory = simulate_kernel(
        rng, positions, angles, model, np.array([0, model.steps]), record=True)
    end_time = (model.warmup + model.steps) * model.dt
    if times[-1] < end_time:
        times = np.r_[times, end_time]
        tracks = np.concatenate((tracks, final[None]))
        states = np.concatenate((states, (memory > 0)[None]))
    selected = np.flatnonzero(np.ones(len(times), dtype=bool) if args.include_warmup
                              else times >= model.warmup * model.dt)
    selected = selected[np.unique(np.linspace(0, len(selected) - 1, min(args.max_frames, len(selected)), dtype=int))]
    times, tracks, states = times[selected], tracks[selected], states[selected]
    velocities = motion_velocities(times, tracks) * args.arrow_scale
    greens = args.N // 2
    separation = tracks[:, :greens, None, :] - tracks[:, None, greens:, :]
    contacts = (np.sum(separation ** 2, axis=-1) < 1.).any(axis=-1).sum(axis=-1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Patch

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=.14, top=.9)
    radius, margin = length / 2, max(.5, .02 * length)
    ax.add_patch(Circle((0, 0), radius, fill=False, color="#334155", linewidth=1.3))
    ax.set(xlim=(-radius - margin, radius + margin), ylim=(-radius - margin, radius + margin),
           xlabel="x / disk diameter", ylabel="y / disk diameter", aspect="equal")
    disks = []
    for i in range(args.N):
        disk = Circle(tracks[0, i], .5, facecolor="tab:green" if i < greens else "tab:red",
                      edgecolor="#334155", linewidth=.45, alpha=.85)
        ax.add_patch(disk)
        disks.append(disk)
    arrows = ax.quiver(tracks[0, :, 0], tracks[0, :, 1], *arrow_components(velocities[0]),
                       angles="xy", scale_units="xy", scale=1, color="#111827", edgecolor="white",
                       linewidth=.25, width=min(.007, .05 / length), headwidth=3.8, headlength=4.8, zorder=10)
    title = ax.set_title("")
    fig.legend(handles=[Patch(color="tab:green", label="Green"), Patch(color="tab:purple", label="Memory active"),
                        Patch(color="tab:red", label="Red")], loc="lower center", ncol=3, frameon=False)

    def update(frame):
        for i, disk in enumerate(disks):
            disk.center = tracks[frame, i]
            if i < greens:
                disk.set_facecolor("tab:purple" if states[frame, i] else "tab:green")
        arrows.set_offsets(tracks[frame])
        arrows.set_UVC(*arrow_components(velocities[frame]))
        title.set_text(f"N={args.N} ({greens} green + {greens} red) · L={length:g} · τ={args.tau:g}\n"
                       f"t={times[frame]:.2f} · {contacts[frame]}/{greens} green currently touching red")
        return *disks, arrows, title

    output.parent.mkdir(parents=True, exist_ok=True)
    update(len(times) - 1)
    fig.savefig(output.with_suffix(".png"), dpi=120)
    animation = FuncAnimation(fig, update, frames=len(times), interval=1000 / args.fps, blit=True)
    print(f"Rendering {len(times)} frames, t={times[0]:g}..{times[-1]:g}", flush=True)

    def progress(frame, total):
        if frame % 50 == 0 or frame == total - 1:
            print(f"Frame {frame + 1}/{total}", flush=True)

    temporary = output.with_name(output.stem + ".rendering.gif")
    try:
        animation.save(temporary, writer=PillowWriter(fps=args.fps), dpi=80, progress_callback=progress)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)
    metrics = normalize(sums.sum(axis=0), args.N, model.steps)
    output.with_suffix(".json").write_text(json.dumps(dict(N=args.N, L=length, tau=args.tau,
        settings=settings, frames=len(times), first_time=float(times[0]), last_time=float(times[-1]),
        contact_fraction=float(metrics[0]), arrow_scale=args.arrow_scale), indent=2) + "\n")
    print(f"Saved {output} and {output.with_suffix('.png')}; integrated green contact fraction={metrics[0]:.4g}")


if __name__ == "__main__":
    main()
