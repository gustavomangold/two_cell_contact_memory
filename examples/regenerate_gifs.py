"""Regenerate both contact GIFs, the duplicate tau=2 GIF, and first_part.gif.

Run: .venv/bin/python examples/regenerate_gifs.py
All show arrows for the actual displacement velocity between recorded frames.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import warnings

# Run from the repository root; make the source modules importable here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import two_disks as simulation


ROOT = Path(__file__).resolve().parents[1]


def historical_gif():
    # Recovered from the historical code and checked against first_part.gif.
    settings = dict(ARENA_DIAMETER=6., SPEED=1., REPULSION=30., WALL_REPULSION=30.,
                    TRANSLATIONAL_DIFFUSION=0., ANGULAR_DIFFUSION=10.,
                    PERSISTENT_DIFFUSION=0., REORIENT_ON_CONTACT=True,
                    PERSISTENCE_TIME=0., DT=.001, DURATION=100.,
                    STEADY_STATE_STEPS=0, SAVE_EVERY=100, STEP_SEPARATION=100,
                    SEED=7, MU=1.)
    previous = {name: getattr(simulation, name) for name in settings}
    output = ROOT / "first_part.gif"
    temporary = ROOT / "first_part.rendering.gif"
    try:
        for name, value in settings.items():
            setattr(simulation, name, value)
        result = simulation.simulate()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*non-GUI backend.*")
            animation = simulation.animate(*result)

        def progress(frame, total):
            if frame % 200 == 0 or frame == total - 1:
                print(f"first_part.gif: frame {frame + 1}/{total}", flush=True)

        animation.save(temporary, writer="pillow", fps=25, dpi=100,
                       progress_callback=progress)
        temporary.replace(output)
        print(f"Saved {output}", flush=True)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close("all")
        for name, value in previous.items():
            setattr(simulation, name, value)


def main():
    exporter = [sys.executable, str(ROOT / "src" / "make_gif.py")]
    subprocess.run(exporter + ["--L", "3", "--tau", "0.001", "--seed", "7",
                               "--save-every", "2000", "--contact-window", "--fps", "10"],
                   check=True)
    subprocess.run(exporter + ["--L", "3", "--tau", "2", "--seed", "8", "--reorient",
                               "--save-every", "50", "--contact-window", "--pre-contact", "0.1",
                               "--post-contact", "3", "--hold-contact", "0.8", "--fps", "25"],
                   check=True)
    shutil.copyfile(ROOT / "figures" / "two_disks_L3_tau2_reorient.gif",
                    ROOT / "two_disks_L3_tau2_reorient.gif")
    historical_gif()


if __name__ == "__main__":
    main()
