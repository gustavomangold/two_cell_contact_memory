#!/usr/bin/env bash
# Test timestep convergence with the existing arena engine; plotting is separate.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python=${PYTHON:-.venv/bin/python}
duration=${DURATION:-0.12}
output=${OUTPUT_ROOT:-data/timestep_sweep/arena_phi2over9_T${duration}}
read -r -a timesteps <<< "${TIMESTEPS:-1e-5 2e-6 5e-7 1e-7 2e-8 1e-8}"

for dt in "${timesteps[@]}"; do
  "$python" src/run.py --boundary arena \
    --sizes 2 4 8 16 32 64 128 256 512 \
    --tau 0 5e-7 1e-6 1e-5 1e-4 1e-3 2e-3 \
    --packing-fraction 0.2222222222222222 --duration "$duration" \
    --samples 50 --workers 10 "$@" --dt "$dt" --output "$output/dt$dt"
done

echo "Plot separately: $python src/plot.py --data-dir $output --timestep-sweep"
