#!/usr/bin/env bash
# No-memory control: vary confinement and baseline angular diffusion.
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$source_dir/.." && pwd)"
simulation_python="${SIM_PYTHON:-$repo_dir/.venv/bin/python}"
if [[ -z "${SIM_PYTHON:-}" && ! -x "$simulation_python" ]]; then
    simulation_python=python3
fi

# Edit these two grids to change the experiment.
lengths=(2.24 3 4 6)
diffusions=(1 10 100 360)

# Keep this control free of memory and reorientation; other options pass through.
for option in "$@"; do
    case "$option" in
        --L|--L=*|--tau*|--angular-diffusion*|--output*|--reorient*)
            echo "The sweep fixes $option; edit the grids above instead." >&2
            exit 2
            ;;
    esac
done

for diffusion in "${diffusions[@]}"; do
    for length in "${lengths[@]}"; do
        echo "No memory: L=$length, angular diffusion=$diffusion"
        "$simulation_python" "$source_dir/contact_stats.py" \
            --samples 50 --workers 10 "$@" \
            --L "$length" --tau 0 --angular-diffusion "$diffusion" \
            --output "$repo_dir/data/contact_stats/no_memory/Dr$diffusion"
    done
done
