#!/bin/bash
# Sample with every schedule of schedules.txt for every NFE and seed.
#
#   bash scripts/run_seeded.sh <denoiser> <save_dir>
#
# Environment overrides:
#   SOLVER      euler (default; figure 3), heun or dpm (appendix)
#   ORDER       order of the DPM solver (default 2)
#   NFES        solver steps T (default "4 8 16 32 64 128 256"; Heun: NFE = 2T-1)
#   SEEDS       default "0 1 2 3 4"
#   MODES       default "ODE SDE"
#   FILTER      only run schedules whose line matches this regex
#   BATCH       molecules per batch (default 258)
#   NUM_SAMPLES molecules per run (default 2560)
#   EXTRA       further Hydra overrides for scripts/sampling.py, e.g. paths.data_dir=...
set -uo pipefail
cd "$(dirname "$0")/.."

DENOISER="$(realpath "$1")"
SAVE_DIR="$(realpath -m "$2")"
SOLVER="${SOLVER:-euler}"
ORDER="${ORDER:-2}"
read -r -a nfes <<< "${NFES:-4 8 16 32 64 128 256}"
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4}"
read -r -a modes <<< "${MODES:-ODE SDE}"
FILTER="${FILTER:-.}"
BATCH="${BATCH:-258}"
NUM_SAMPLES="${NUM_SAMPLES:-2560}"
EXTRA="${EXTRA:-}"

grep -v '^\s*#' schedules.txt | grep -v '^\s*$' | grep -e "$FILTER" | while IFS='|' read -r name allowed family overrides; do
  name=$(echo "$name" | xargs); allowed=$(echo "$allowed" | xargs); family=$(echo "$family" | xargs)
  case "$family/$SOLVER" in
    snr/euler) sampler=sde/snr_euler ;;
    snr/heun) sampler=sde/snr_heun ;;
    snr/dpm) sampler="sde/dpm_solver order=$ORDER" ;;
    kve/euler) sampler=sde/euler ;;
    kve/heun) sampler=sde/heun ;;
    *) echo "skip $name: $SOLVER not available"; continue ;;
  esac
  for mode in "${modes[@]}"; do
    [[ ",$allowed," == *",$mode,"* ]] || continue
    # only Euler-Maruyama supports the SDE
    [ "$mode" = SDE ] && [ "$SOLVER" != euler ] && continue
    [ "$mode" = SDE ] && stochastic=True || stochastic=False
    for n in "${nfes[@]}"; do
      # NFES are solver steps; Heun uses 2T-1 function evaluations
      nfe=$n
      [ "$SOLVER" = heun ] && nfe=$(( 2 * n - 1 ))
      for seed in "${seeds[@]}"; do
        base="${name}_${mode}_${SOLVER}_nfe${nfe}_seed${seed}"
        [ -e "$SAVE_DIR/$base.db" ] && { echo "done: $base"; continue; }
        echo "== $base"
        python scripts/sampling.py sampler=$sampler $overrides stochastic="$stochastic" \
          T="$n" seed="$seed" data.batch_size="$BATCH" num_samples="$NUM_SAMPLES" \
          paths.denoiser_path="$DENOISER" paths.save_dir="$SAVE_DIR" \
          paths.file_base_name="$base" $EXTRA < /dev/null || exit 1
      done
    done
  done
done
