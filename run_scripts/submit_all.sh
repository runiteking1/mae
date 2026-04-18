#!/bin/bash
# Submit each ablation script twice, chained via --dependency=afterany so the
# second job resumes when the first finishes (or hits its 4-day walltime).
# Total wall: 8 days per ablation.
#
# Requires: each script auto-appends `--resume ${OUTPUT_DIR}/checkpoint-latest.pth`
# when that file exists. Without that, the second job restarts from epoch 0.

set -euo pipefail

cd "$(dirname "$0")"

SCRIPTS=(
    adamw_base_2.4e-3.sh
    adamw_large_2.4e-3.sh
    muon_base_2.4e-3.sh
    muon_base_5.0e-3.sh
    muon_large_2.4e-3.sh
    muon_large_5.0e-3.sh
)

for script in "${SCRIPTS[@]}"; do
    id1=$(sbatch --parsable "$script")
    id2=$(sbatch --parsable --dependency=afterany:"$id1" "$script")
    printf '%-30s  %s -> %s\n' "$script" "$id1" "$id2"
done
