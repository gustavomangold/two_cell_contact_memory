"""Export a single simulation, without changing two_disks.py defaults.

Run: python src/make_gif.py --L 3 --tau 0.001 --save-every 2000
Show the observation interval after warm-up; --include-warmup shows everything.
Use --contact-window to show an actual encounter. If the default interval has
no contacts, automatically use a contact window instead.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Patch
import numpy as np
import two_disks as simulation
from velocity_arrows import motion_velocities, add_velocity_arrows, arrow_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=float, default=3.)
    parser.add_argument("--tau", type=float, default=.001)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=simulation.DURATION)
    parser.add_argument("--dt", type=float, default=simulation.DT)
    parser.add_argument("--steady-state-steps", type=int, default=simulation.STEADY_STATE_STEPS)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=simulation.SEED)
    parser.add_argument("--reorient", action="store_true")
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--contact-window", action="store_true")
    parser.add_argument("--pre-contact", type=float, default=3.)
    parser.add_argument("--post-contact", type=float, default=15.)
    parser.add_argument("--hold-contact", type=float, default=0.,
                        help="Pause on the first contact frame for this many playback seconds")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (not np.isfinite(args.L) or args.L <= 1 or not np.isfinite(args.tau)
            or args.tau < 0 or not np.isfinite(args.duration) or args.duration <= 0
            or not np.isfinite(args.dt) or args.dt <= 0 or args.steady_state_steps < 0
            or args.seed < 0
            or args.save_every < 1 or args.fps < 1 or not np.isfinite(args.pre_contact)
            or args.pre_contact < 0 or not np.isfinite(args.post_contact) or args.post_contact <= 0
            or not np.isfinite(args.hold_contact) or args.hold_contact < 0):
        parser.error("Require L>1, tau/pre-contact>=0 and positive duration/save-every/fps/post-contact")

    simulation.ARENA_DIAMETER = args.L
    simulation.PERSISTENCE_TIME = args.tau
    simulation.SAVE_EVERY = args.save_every
    simulation.DURATION = args.duration
    simulation.SEED = args.seed
    simulation.REORIENT_ON_CONTACT = args.reorient
    simulation.DT = args.dt
    simulation.STEADY_STATE_STEPS = args.steady_state_steps
    times, tracks, persistent, _ = simulation.simulate()
    velocities = motion_velocities(times, tracks)
    distances = np.linalg.norm(tracks[:, 0] - tracks[:, 1], axis=1)
    selected = np.ones(len(times), dtype=bool) if args.include_warmup else (
        times >= simulation.STEADY_STATE_STEPS * simulation.DT)
    if args.contact_window or not (distances[selected] < 1).any():
        contacts = np.flatnonzero(distances < 1)
        if not len(contacts):
            parser.error("No recorded contacts; increase --duration or reduce --save-every")
        contact_time = times[contacts[0]]
        selected = ((times >= contact_time - args.pre_contact)
                    & (times <= contact_time + args.post_contact))
        print(f"Selected actual encounter around t={contact_time:g}", flush=True)
    times, tracks, persistent, distances = (
        times[selected], tracks[selected], persistent[selected], distances[selected])
    velocities = velocities[selected]
    if not len(times):
        parser.error("No animation frames; reduce --save-every")
    contact_frames = np.count_nonzero(distances < 1)
    if not contact_frames:
        parser.error("Selected clip has no contact frames")

    radius = args.L / 2
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(bottom=.18, top=.88)
    ax.add_patch(Circle((0, 0), radius, fill=False, color="#334155", linewidth=1.5))
    ax.set(xlim=(-radius - .2, radius + .2), ylim=(-radius - .2, radius + .2),
           xlabel="x / disk diameter", ylabel="y / disk diameter", aspect="equal")
    disks = [Circle(tracks[0, i], .5, color=color, alpha=.85)
             for i, color in enumerate(["tab:green", "tab:red"])]
    for disk in disks:
        disk.set_edgecolor("#334155")
        disk.set_linewidth(1.)
        ax.add_patch(disk)
    arrows = add_velocity_arrows(ax, tracks[0], velocities[0])
    title = ax.set_title("")
    mode = "Re-orient" if args.reorient else "Do not re-orient"
    fig.legend(handles=[Patch(color="tab:green", label="Green"),
                        Patch(color="tab:purple", label="Memory active"),
                        Patch(color="tab:red", label="Red")],
               loc="lower center", ncol=3, frameon=False)

    def update(frame):
        for i, disk in enumerate(disks):
            disk.center = tracks[frame, i]
        disks[0].set_color("tab:purple" if persistent[frame] else "tab:green")
        arrows.set_offsets(tracks[frame])
        arrows.set_UVC(*arrow_components(velocities[frame]))
        state = "Contact" if distances[frame] < 1 else "Separated"
        title.set_text(f"{mode} · L={args.L:g} · τ={args.tau:g}\n"
                       f"t={times[frame]:.2f} · d/D={distances[frame]:.3f} · {state}")
        return *disks, arrows, title

    frames = list(range(len(times)))
    if args.hold_contact:
        contact = int(np.flatnonzero(distances < 1)[0])
        frames[contact:contact] = [contact] * round(args.hold_contact * args.fps)
    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps,
                              blit=True)
    suffix = "reorient" if args.reorient else "no_reorient"
    output = args.output or (Path(__file__).resolve().parents[1] / "figures" /
                            f"two_disks_L{args.L:g}_tau{args.tau:g}_{suffix}.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(times)} frames, t={times[0]:g}..{times[-1]:g}, "
          f"every {args.save_every} steps", flush=True)
    print(f"Verified {contact_frames} contact frames; minimum d/D={distances.min():.3f}", flush=True)

    def progress(frame, total):
        if frame % 100 == 0 or frame == total - 1:
            print(f"Frame {frame + 1}/{total}", flush=True)

    temporary = output.with_name(output.stem + ".rendering.gif")
    try:
        animation.save(temporary, writer=PillowWriter(fps=args.fps), dpi=75,
                       progress_callback=progress)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
