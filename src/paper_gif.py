"""Animate the paper's periodic-square dynamics, with displacement-velocity arrows."""

import argparse
import json
from pathlib import Path

import numpy as np
from paper_engine import ROOT, DEFAULTS, neighbor_method, source_hash
from paper_movie import record_movie
from velocity_arrows import motion_velocities, arrow_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--D", type=float, default=360000.)
    parser.add_argument("--tau", type=float, default=.001)
    parser.add_argument("--packing-fraction", type=float, default=DEFAULTS["packing_fraction"])
    parser.add_argument("--duration", type=float, default=DEFAULTS["duration"])
    parser.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    parser.add_argument("--start", type=float, help="First displayed time; default: final 0.05 paper time units")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--gsl-prefix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = max(0., args.duration-.05) if args.start is None else args.start
    if (args.N < 2 or args.N % 2 or not all(np.isfinite(v) for v in
            [args.D,args.tau,args.packing_fraction,args.dt,args.duration,start])
            or min(args.D,args.tau,start) < 0 or not 0 < args.packing_fraction <= 1
            or min(args.dt,args.duration) <= 0 or not 0 <= start < args.duration
            or min(args.save_every,args.fps) < 1 or args.max_frames < 2
            or not 0 <= args.seed < 2**32 or not 20 <= round(args.duration/args.dt) < 2**31-1):
        parser.error("Use even N>=2, nonnegative D/tau, 0<packing fraction<=1, 0<=start<duration, positive frame settings and >=20 steps")
    output = args.output or ROOT / "figures/exactly-like-paper" / f"disks_N{args.N}_tau{args.tau:g}.gif"
    if output.suffix.lower() != ".gif":
        parser.error("output must end in .gif")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.collections import EllipseCollection
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Patch
        import PIL
    except ModuleNotFoundError as error:
        parser.error(f"Missing {error.name}; install numpy matplotlib pillow using this Python environment")
    settings = {**DEFAULTS, "dt":args.dt, "duration":args.duration,
                "packing_fraction":args.packing_fraction, "seed":args.seed}
    output.parent.mkdir(parents=True,exist_ok=True)
    log = output.with_suffix(".log")
    print(f"Simulating paper C engine: N={args.N}, tau={args.tau:g}, final time={args.duration:g}; log: {log}",flush=True)
    try:
        movie = record_movie(args.N,args.D,args.tau,args.seed,settings,start,args.save_every,args.max_frames,log,args.gsl_prefix)
    except (RuntimeError,ValueError) as error:
        parser.error(str(error))
    times, tracks, length = movie["times"], movie["tracks"], movie["L"]
    # Normalize lengths only for arrow display; directions use true unwrapped motion.
    velocities = motion_velocities(times,movie["unwrapped"]) / settings["Pe"]
    shifts = length * np.array([(x,y) for x in [-1,0,1] for y in [-1,0,1]])
    palette = np.array([to_rgba(c) for c in ["tab:red","tab:green","tab:purple"]])

    def offsets(frame):
        return (tracks[frame][None,:,:]+shifts[:,None,:]).reshape(-1,2)

    def components(frame):
        u,v = arrow_components(velocities[frame])
        return np.ma.concatenate([u]*9), np.ma.concatenate([v]*9)

    fig,ax = plt.subplots(figsize=(7,7))
    fig.subplots_adjust(bottom=.14,top=.89)
    ax.set(xlim=(0,length),ylim=(0,length),aspect="equal",xlabel="x / disk diameter",ylabel="y / disk diameter")
    count = 9 * args.N
    disks = EllipseCollection(np.ones(count),np.ones(count),np.zeros(count),units="xy",
                              offsets=offsets(0),offset_transform=ax.transData,
                              facecolors=np.tile(palette[movie["colors"][0]],(9,1)),
                              edgecolors="#334155",linewidths=.5,alpha=.85)
    ax.add_collection(disks)
    arrows = ax.quiver(*offsets(0).T,*components(0),angles="xy",scale_units="xy",scale=1,
                       color="#111827",edgecolor="white",linewidth=.25,
                       width=min(.006,.025/length),zorder=3)
    title = ax.set_title("")
    fig.legend(handles=[Patch(color=c,label=t) for c,t in
                        [("tab:green","Green"),("tab:purple","Memory active"),("tab:red","Red")]],
               loc="lower center",ncol=3,frameon=False)

    def update(frame):
        disks.set_offsets(offsets(frame))
        disks.set_facecolors(np.tile(palette[movie["colors"][frame]],(9,1)))
        arrows.set_offsets(offsets(frame))
        arrows.set_UVC(*components(frame))
        title.set_text(f"N={args.N} · φ={args.packing_fraction:g} · τ={args.tau:g}\n"
                       f"t={times[frame]:.5f} (paper units) · periodic square")
        return disks,arrows,title

    update(len(times)-1)
    fig.savefig(output.with_suffix(".png"),dpi=120)
    animation = FuncAnimation(fig,update,frames=len(times),interval=1000/args.fps,blit=True)
    temporary = output.with_name(output.stem+".rendering.gif")
    print(f"Rendering {len(times)} frames, t={times[0]:g}..{times[-1]:g}",flush=True)

    def progress(frame,total):
        if frame % 25 == 0 or frame == total-1:
            print(f"Frame {frame+1}/{total}",flush=True)

    try:
        animation.save(temporary,writer=PillowWriter(fps=args.fps),dpi=80,progress_callback=progress)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)
    metadata = dict(N=args.N,D=args.D,tau=args.tau,L=length,settings=settings,source_hash=source_hash(),
                    neighbor_method=neighbor_method(args.N,args.packing_fraction),frames=len(times),
                    first_time=float(times[0]),last_time=float(times[-1]),save_every=args.save_every,
                    arrows="mean unwrapped displacement velocity, normalized by Pe for display")
    output.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Saved {output} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
