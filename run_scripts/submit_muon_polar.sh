#!/bin/bash
# Submit the two Muon + Polar Express pretrain ablations (ViT-Base ~560 ep,
# ViT-Large ~440 ep), each as a chain of CHAINS jobs linked by
# --dependency=afterany so the next job resumes when the previous one finishes
# or hits its 4-day walltime. Each job auto-resumes from checkpoint-latest.pth.
#
# CHAINS defaults to 3 (≈12 days wall cap per ablation): Polar Express is a bit
# heavier per step than vanilla Muon, and the prior 2-chain Muon runs only
# reached ~526/~427 epochs, so 3 chains gives headroom to actually hit the
# target. Over-provisioning is safe — once training reaches --epochs, a resumed
# job loads the final checkpoint and the epoch loop is a no-op, so it exits
# almost immediately.
#
# Usage:
#   bash run_scripts/submit_muon_polar.sh        # 3 chains each (default)
#   CHAINS=4 bash run_scripts/submit_muon_polar.sh
#   DRY_RUN=1 bash run_scripts/submit_muon_polar.sh   # print, don't submit

set -euo pipefail

cd "$(dirname "$0")"

CHAINS=${CHAINS:-3}
DRY_RUN=${DRY_RUN:-0}

SCRIPTS=(
    muon_polar_base_2.4e-3.sh
    muon_polar_large_2.4e-3.sh
)

submit() {  # submit one chain of CHAINS jobs for a single script
    local script=$1
    local dep="" id
    for ((i = 1; i <= CHAINS; i++)); do
        if [[ -n "$dep" ]]; then
            cmd=(sbatch --parsable --dependency=afterany:"$dep" "$script")
        else
            cmd=(sbatch --parsable "$script")
        fi
        if [[ "$DRY_RUN" == "1" ]]; then
            printf '  [dry-run] %s\n' "${cmd[*]}"
            id="<job$i>"
        else
            id=$("${cmd[@]}")
        fi
        printf '%-28s  chain %d/%d -> %s\n' "$script" "$i" "$CHAINS" "$id"
        dep=$id
    done
}

for script in "${SCRIPTS[@]}"; do
    submit "$script"
done
