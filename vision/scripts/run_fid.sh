#!/bin/bash
# Generate images and compute the FID for every row of settings.txt and every seed.
#
#   bash scripts/run_fid.sh [settings.txt]
#
# Environment overrides:
#   SEEDS       seed indices (default "0 1 3 4 5"); index i uses the images
#               50000*i ... 50000*i+N_IMAGES-1. Index 2 (100000-149999) is
#               skipped because the Bayesian optimization tuned on it.
#   FILTER      only run rows matching this regex, e.g. "^CIFAR\|EDM\|heun"
#   EXPDIR      output root (default exp)
#   N_IMAGES    images per run (default 50000)
#   BATCH       batch size (default 128)
#   MODELS_DIR  local directory with the EDM pickles (default: download)
#   KEEP_IMAGES keep the generated images after computing the FID
set -uo pipefail
cd "$(dirname "$0")/.."

SPEC="${1:-settings.txt}"
EXPDIR="${EXPDIR:-exp}"
N_IMAGES="${N_IMAGES:-50000}"
BATCH="${BATCH:-128}"
FILTER="${FILTER:-.}"
read -r -a seeds <<< "${SEEDS:-0 1 3 4 5}"

CDN=https://nvlabs-fi-cdn.nvidia.com/edm
declare -A NETWORK=(
  [CIFAR]=edm-cifar10-32x32-uncond-vp.pkl [FFHQ]=edm-ffhq-64x64-uncond-vp.pkl
  [AFHQ]=edm-afhqv2-64x64-uncond-vp.pkl [imagenet]=edm-imagenet-64x64-cond-adm.pkl
)
declare -A FIDREF=(
  [CIFAR]=cifar10-32x32.npz [FFHQ]=ffhq-64x64.npz
  [AFHQ]=afhqv2-64x64.npz [imagenet]=imagenet-64x64.npz
)

grep -v '^\s*#' "$SPEC" | grep -v '^\s*$' | grep -e "$FILTER" | while IFS='|' read -r dataset label solver steps flags; do
  if [ -n "${MODELS_DIR:-}" ]; then
    network="$MODELS_DIR/${NETWORK[$dataset]}"
  else
    network="$CDN/pretrained/${NETWORK[$dataset]}"
  fi
  for seed in "${seeds[@]}"; do
    outdir="$EXPDIR/${dataset}_${label}_${solver}_steps${steps}_seed${seed}"
    [ -s "$outdir/fid.txt" ] && { echo "done: $outdir"; continue; }
    mkdir -p "$outdir"
    lo=$(( 50000 * seed )); hi=$(( lo + N_IMAGES - 1 ))
    echo "== $dataset $label $solver steps=$steps seed=$seed"
    python generate_tv_snr.py --outdir "$outdir" --network "$network" --solver "$solver" \
      --steps "$steps" --seeds "$lo-$hi" --batch "$BATCH" --subdirs $flags < /dev/null || exit 1
    python fid.py calc --images "$outdir" --ref "$CDN/fid-refs/${FIDREF[$dataset]}" \
      --num "$N_IMAGES" < /dev/null | tail -n 1 | tr -d '[:space:]' > "$outdir/fid.txt" || exit 1
    echo "FID=$(cat "$outdir/fid.txt") NFE=$(cat "$outdir/nfe.txt")"
    [ -z "${KEEP_IMAGES:-}" ] && find "$outdir" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
  done
done
