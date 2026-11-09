"""Two active disks in a circular arena, inspired by cell_simulation.

Run: python3 src/two_disks.py --PERSISTENCE_TIME 2 (requires numpy and matplotlib).
Lengths below are in disk diameters: D=10 physically, arena diameter=L*D.
Fig. 4b parameters, with time rescaled by Pe=1000 so that SPEED=1.
Paper: Pe=k=1000, D=360000, D+=0, tau=0.002, packing fraction=0.4.
The two-particle circular confinement remains an approximation to the paper.
Optional acceleration: install numba>=0.59; otherwise the NumPy version runs.

Both disks propel themselves and diffuse. Contact activates persistence in
the green disk, without turning either disk (Fig. 4 isolates persistence).
Disk and wall forces are soft springs, with separate stiffnesses.
"""

import argparse
from pathlib import Path

import numpy as np

try:
    from _two_disks_numba import simulate_kernel as _simulate_kernel
except ModuleNotFoundError as error:
    if error.name != "numba":
        raise
    _simulate_kernel = None


# Edit these parameters to explore the model. Mobility is set to one.
DISK_DIAMETER = 10.0      # Physical length represented by one unit
ARENA_DIAMETER = np.sqrt(5.0)        # Nominal area fraction 2/L^2 = 0.4
SPEED = 1.0                         # Disk diameters per time unit
REPULSION = 1.0                     # Paper k/Pe; mobility remains one
WALL_REPULSION = 30.0               # Preserve confinement when reducing disk stiffness
TRANSLATIONAL_DIFFUSION = 0.001     # Paper translational diffusion / Pe
ANGULAR_DIFFUSION = 360.0           # Paper D/Pe
PERSISTENT_DIFFUSION = 0.0          # Ballistic activated green disk
REORIENT_ON_CONTACT = False         # Fig. 4 excludes contact-induced turning
PERSISTENCE_TIME = 2.0
DT = 0.00001                        # Resolve the faster angular diffusion

STEP_SEPARATION = 50                # Separation sampling stride, in integration steps
STEADY_STATE_STEPS = 2000000        # Steps discarded before measurements

DURATION = 100.0
SAVE_EVERY = 2000                    # Animation frame stride, in integration steps
SEED = 7
MU = 1


def repulsive_force(overlap, normal, strength=None):
    """Soft spring, with disk stiffness unless another strength is supplied."""
    if strength is None:
        strength = REPULSION
    return strength * max(0.0, overlap) * normal


def simulate(diagnostics=None):
    """Return times, positions, persistence flags, separations; optionally collect statistics."""
    if _simulate_kernel is None:
        return _simulate_python(diagnostics)
    return _simulate_kernel(
        np.random.default_rng(SEED), ARENA_DIAMETER, SPEED, REPULSION,
        ANGULAR_DIFFUSION, PERSISTENCE_TIME, DT, DURATION,
        STEADY_STATE_STEPS, STEP_SEPARATION, SAVE_EVERY, MU,
        TRANSLATIONAL_DIFFUSION, PERSISTENT_DIFFUSION,
        REORIENT_ON_CONTACT, WALL_REPULSION, diagnostics,
    )


def _simulate_python(diagnostics=None):
    """Readable NumPy reference and fallback for the compiled version."""
    rng = np.random.default_rng(SEED)
    if diagnostics is not None:
        from contact_stats import _observe_python
    # Disk 0 is green, disk 1 is red; initially they face each other.
    positions = np.array([[-1.0, 0.0], [1.0, 0.0]])
    angles = np.array([0.0, np.pi])
    persistent_left = 0.0
    # Wall contact begins half a disk diameter inside the arena boundary.
    center_limit = ARENA_DIAMETER / 2 - 0.5
    times, tracks, persistent = [0.0], [positions.copy()], [False]

    separations = []

    for step in range(1, round(DURATION / DT) + STEADY_STATE_STEPS + 1):
        activated = False
        persistent_left = max(0.0, persistent_left - DT)
        separation = positions[0] - positions[1]
        distance = np.linalg.norm(separation)
        # A fallback direction avoids division by zero at coincident centers.
        normal = separation / distance if distance > 1e-12 else np.array([1., 0.])
        forces = np.zeros((2, 2))

        if distance < 1.0:  # Contact: one disk diameter between centers
            forces[0] = repulsive_force(1.0 - distance, normal)
            forces[1] = - forces[0]
            if REORIENT_ON_CONTACT:
                away = np.arctan2(normal[1], normal[0])
                angles[:] = [away, away + np.pi]
            # Like the repo, contacts do not refresh an active persistence timer.
            if persistent_left == 0.0:
                persistent_left = PERSISTENCE_TIME
                activated = PERSISTENCE_TIME > 0

        if diagnostics is not None:
            _observe_python(diagnostics, step, positions, angles, persistent_left, activated)

        disk_distances = np.linalg.norm(positions, axis=1)

        for i in range(2):
            if disk_distances[i] > center_limit:
                penetration = disk_distances[i] - center_limit
                inward = - positions[i] / disk_distances[i]
                forces[i] += repulsive_force(penetration, inward, WALL_REPULSION)

        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        # Euler-Maruyama: drift * dt + sqrt(2 * diffusion * dt) * normal noise.
        positions += (SPEED * directions + MU * forces) * DT
        if TRANSLATIONAL_DIFFUSION > 0:
            positions += np.sqrt(2 * TRANSLATIONAL_DIFFUSION * DT) * rng.normal(size=(2, 2))

        # Project centers back onto the allowed circle: a sliding hard wall.
        
        #outside = radii > center_limit
        #positions[outside] *= (center_limit / radii[outside])[:, None 

        angular_diffusion_green = ANGULAR_DIFFUSION
        if persistent_left > 0:
            angular_diffusion_green = PERSISTENT_DIFFUSION

        angular_diffusion = np.array([
            angular_diffusion_green,
            ANGULAR_DIFFUSION
        ])
        angles += np.sqrt(2 * angular_diffusion * DT) * rng.normal(size=2)

        if step >= STEADY_STATE_STEPS:
            if step % STEP_SEPARATION == 0:
                separations.append(distance)

        if step % SAVE_EVERY == 0 or step == round(DURATION / DT):
            times.append(step * DT)
            tracks.append(positions.copy())
            persistent.append(persistent_left > 0)

    return np.array(times), np.array(tracks), np.array(persistent), np.array(separations)


def animate(times, tracks, persistent, separations):
    """Show the arena; purple means the green disk has increased persistence."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle
    from velocity_arrows import motion_velocities, add_velocity_arrows, arrow_components

    velocities = motion_velocities(times, tracks)
    fig, ax = plt.subplots(figsize=(6, 6))
    radius = ARENA_DIAMETER / 2
    ax.add_patch(Circle((0, 0), radius, fill=False, color="black"))
    ax.set(xlim=(-radius - .2, radius + .2), ylim=(-radius - .2, radius + .2),
           xlabel="x / disk diameter", ylabel="y / disk diameter", aspect="equal")
    disks = [Circle(tracks[0, i], .5, color=color, alpha=.8)
             for i, color in enumerate(["tab:green", "tab:red"])]
    for disk in disks:
        ax.add_patch(disk)
    arrows = add_velocity_arrows(ax, tracks[0], velocities[0])
    title = ax.set_title("")

    def update(frame):
        for i, disk in enumerate(disks):
            disk.center = tracks[frame, i]
        disks[0].set_color("tab:purple" if persistent[frame] else "tab:green")
        arrows.set_offsets(tracks[frame])
        arrows.set_UVC(*arrow_components(velocities[frame]))
        title.set_text(f"t = {times[frame]:.2f} | arena diameter = {ARENA_DIAMETER:g}D, D = {DISK_DIAMETER:g}")
        return *disks, arrows, title
    
    animation = FuncAnimation(fig, update, frames=len(times), interval=40, blit=True)
    plt.show()
    return animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=float, default=ARENA_DIAMETER,
                        help="Arena diameter in disk diameters (default: %(default)s)")
    parser.add_argument("--PERSISTENCE_TIME", "--persistence-time", type=float,
                        default=PERSISTENCE_TIME,
                        help="Persistence duration after contact (default: %(default)s)")
    parser.add_argument("--ANIMATE", "--animate", type=bool,
                            default=False,
                            help="Display animation (default: %(default)s)")
    parser.add_argument("--STEADY_STATE_STEPS", "--steady-state-steps", type=int,
                        default=STEADY_STATE_STEPS,
                        help="Steps discarded before sampling (default: %(default)s)")
    parser.add_argument("--DURATION", "--duration", type=float, default=DURATION,
                        help="Simulation duration (default: %(default)s)")
    args = parser.parse_args()
    if not np.isfinite(args.L) or args.L <= 1:
        parser.error("L must be finite and greater than 1 disk diameter")
    if not np.isfinite(args.PERSISTENCE_TIME) or args.PERSISTENCE_TIME < 0:
        parser.error("PERSISTENCE_TIME must be finite and nonnegative")
    if not np.isfinite(args.DURATION) or args.DURATION <= 0:
        parser.error("DURATION must be finite and positive")
    if args.STEADY_STATE_STEPS < 0:
        parser.error("STEADY_STATE_STEPS must be nonnegative")
    ARENA_DIAMETER = args.L
    PERSISTENCE_TIME = args.PERSISTENCE_TIME
    STEADY_STATE_STEPS = args.STEADY_STATE_STEPS
    DURATION = args.DURATION
    results = simulate()
    if args.ANIMATE:
        animation = animate(*results)
    separations = results[3]
    if separations.size:
        import matplotlib.pyplot as plt

        plt.figure()
        bins = np.arange(0, separations.max() + 0.05, 0.05)
        plt.hist(separations, bins=bins, edgecolor="black")
        plt.xlabel("Separation / disk diameter")
        plt.ylabel("Count")
        plt.figtext(0.5, 0.02, f"Mean separation = {separations.mean():.3f} disk diameters",
                    ha="center")
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        data_dir = Path(__file__).resolve().parents[1] / "data" / "paper_params"
        data_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(data_dir / f"separations_L{ARENA_DIAMETER:g}_tau{PERSISTENCE_TIME:g}.png")
        #plt.show()
