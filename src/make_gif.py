"""Animate the same paper engine used by run.py, with velocity arrows."""

import argparse
import json
from pathlib import Path

import numpy as np
import engine


def velocities(times, tracks, speed):
    """Interval-averaged unwrapped velocity; Brownian motion has no instantaneous velocity."""
    values = np.zeros_like(tracks)
    values[:-1] = np.diff(tracks,axis=0)/np.diff(times)[:,None,None]
    values[-1] = values[-2]
    return values/speed


def arrow_components(values):
    arrows = .35*values
    lengths = np.linalg.norm(arrows,axis=-1,keepdims=True)
    arrows *= np.minimum(1.,.65/np.maximum(lengths,1e-12))
    stationary = np.linalg.norm(values,axis=-1)==0
    return np.ma.masked_where(stationary,arrows[...,0]), np.ma.masked_where(stationary,arrows[...,1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary",choices=engine.BOUNDARIES,required=True)
    parser.add_argument("--N",type=int,default=32)
    parser.add_argument("--D",type=float,default=engine.DEFAULT_DIFFUSION)
    parser.add_argument("--D-plus",type=float,default=engine.DEFAULTS["D_plus"])
    parser.add_argument("--tau",type=float,default=.001)
    parser.add_argument("--packing-fraction",type=float,default=engine.DEFAULTS["packing_fraction"])
    parser.add_argument("--wall-stiffness",type=float,default=30000.)
    parser.add_argument("--dt",type=float,default=engine.DEFAULTS["dt"])
    parser.add_argument("--duration",type=float,default=engine.DEFAULTS["duration"])
    parser.add_argument("--seed",type=int,default=engine.DEFAULTS["seed"])
    parser.add_argument("--start",type=float,help="First displayed time; default: final 0.05 time units")
    parser.add_argument("--save-every",type=int,default=50)
    parser.add_argument("--max-frames",type=int,default=300)
    parser.add_argument("--fps",type=int,default=25)
    parser.add_argument("--gsl-prefix",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    start = max(0.,args.duration-.05) if args.start is None else args.start
    if (args.N<2 or args.N%2 or not all(np.isfinite(v) for v in
            [args.D,args.D_plus,args.tau,args.packing_fraction,args.wall_stiffness,args.dt,args.duration,start])
            or min(args.D,args.D_plus,args.tau,start)<0 or min(args.dt,args.duration,args.wall_stiffness)<=0
            or not 0<args.packing_fraction<=1 or not 0<=start<args.duration
            or min(args.save_every,args.fps)<1 or args.max_frames<2 or not 0<=args.seed<2**32
            or not 20<=round(args.duration/args.dt)<2**31-1):
        parser.error("Use even N>=2, finite valid physics, 0<=start<duration, positive frame settings, uint32 seed and >=20 steps")
    boundary = engine.BOUNDARIES[args.boundary]
    output = args.output or engine.figure_directory(boundary)/f"disks_N{args.N}_tau{args.tau:g}.gif"
    if output.suffix.lower()!=".gif":parser.error("Output must end in .gif")
    output.parent.mkdir(parents=True,exist_ok=True)
    settings = engine.make_settings(boundary,args.dt,args.duration,args.packing_fraction,args.seed,args.wall_stiffness,args.D_plus)
    print(f"Recording {args.boundary}, N={args.N}, tau={args.tau:g}; log: {output.with_suffix('.log')}",flush=True)
    try:
        movie = engine.record_motion(args.N,args.D,args.tau,args.seed,settings,start,args.save_every,
                                     args.max_frames,output.with_suffix(".log"),args.gsl_prefix)
    except (ValueError,RuntimeError) as error:
        parser.error(str(error))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import EllipseCollection
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Circle, Patch

    times,tracks,length = movie["times"],movie["tracks"],movie["L"]
    motion = velocities(times,movie["unwrapped"],settings["Pe"])
    shifts = length*np.array([(x,y) for x in [-1,0,1] for y in [-1,0,1]]) if args.boundary=="pbc" else np.zeros((1,2))
    palette = np.array([to_rgba(c) for c in ["tab:red","tab:green","tab:purple"]])
    fig,ax = plt.subplots(figsize=(7,7))
    fig.subplots_adjust(bottom=.14,top=.89)
    ax.set(xlim=(0,length),ylim=(0,length),aspect="equal",xlabel="x / disk diameter",ylabel="y / disk diameter")
    if args.boundary=="arena":ax.add_patch(Circle((length/2,length/2),length/2,fill=False,color="#334155"))

    def offsets(frame):
        return (tracks[frame][None,:,:]+shifts[:,None,:]).reshape(-1,2)

    def components(frame):
        u,v = arrow_components(motion[frame])
        return np.ma.concatenate([u]*len(shifts)), np.ma.concatenate([v]*len(shifts))

    count = len(shifts)*args.N
    disks = EllipseCollection(np.ones(count),np.ones(count),np.zeros(count),units="xy",
                              offsets=offsets(0),offset_transform=ax.transData,
                              facecolors=np.tile(palette[movie["colors"][0]],(len(shifts),1)),
                              edgecolors="#334155",linewidths=.5,alpha=.85)
    ax.add_collection(disks)
    arrows = ax.quiver(*offsets(0).T,*components(0),angles="xy",scale_units="xy",scale=1,
                       color="#111827",edgecolor="white",linewidth=.25,width=min(.006,.025/length),zorder=3)
    title = ax.set_title("")
    fig.legend(handles=[Patch(color=c,label=t) for c,t in
                        [("tab:green","Green"),("tab:purple","Memory active"),("tab:red","Red")]],
               loc="lower center",ncol=3,frameon=False)

    def update(frame):
        disks.set_offsets(offsets(frame))
        disks.set_facecolors(np.tile(palette[movie["colors"][frame]],(len(shifts),1)))
        arrows.set_offsets(offsets(frame));arrows.set_UVC(*components(frame))
        title.set_text(f"{args.boundary} · N={args.N} · φ={args.packing_fraction:g} · τ={args.tau:g}\nt={times[frame]:.5f} (paper units)")
        return disks,arrows,title

    update(len(times)-1)
    fig.savefig(output.with_suffix(".png"),dpi=120)
    animation = FuncAnimation(fig,update,frames=len(times),interval=1000/args.fps,blit=True)
    temporary = output.with_name(output.stem+".rendering.gif")
    print(f"Rendering {len(times)} frames, t={times[0]:g}..{times[-1]:g}",flush=True)
    try:
        animation.save(temporary,writer=PillowWriter(fps=args.fps),dpi=80)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)
    metadata = dict(N=args.N,D=args.D,tau=args.tau,L=length,settings=settings,frames=len(times),
                    first_time=float(times[0]),last_time=float(times[-1]),save_every=args.save_every,
                    arrows="mean unwrapped displacement velocity, normalized by Pe for display")
    output.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Saved {output} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
