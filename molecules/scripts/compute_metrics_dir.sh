#!/bin/bash
# Full evaluation chain for a directory of generated-sample databases.
#
#   <dir>/*.db  ->  *.mol_dict  ->  *.res (cG-SchNet)  ->  *.vun.json
#
# cG-SchNet provides stability, uniqueness and novelty; RDKit provides validity.
# The definitions are documented in scripts/compute_vun.py.
#
# Usage:
#   scripts/compute_metrics_dir.sh <sample_dir> [split.npz]
#
#   split.npz  split used for novelty
#              (default data/split.npz)
#
# Environment overrides:
#   JOBS          files scored in parallel (default 1). Each run uses one CPU core
#                 and up to ~1 GB of memory.
#   CGSCHNET_DIR  cG-SchNet checkout   (default molecules/cG-SchNet)
#   CGSCHNET_FP_CACHE  cache for the training fingerprints
#                 (default ~/.cache/cgschnet_train_fps)
#   QM9_DB        QM9 database         (default data/qm9.db)
#
# Existing outputs are skipped, so the script is safe to re-run.
#
# A sample set without a single valid molecule makes cG-SchNet crash after it has
# printed its counts. That case is detected here and written as a .res with all
# counts 0. Any other failure writes no .res: the log is kept as
# <name>.res.failed, the file is retried on the next run, and the script exits 1.

set -u

base_path="${1:-}"
if [ -z "$base_path" ] || [ ! -d "$base_path" ]; then
  echo "Usage: $0 <sample_dir> [split.npz]" >&2
  exit 1
fi
base_path="$(cd "$base_path" && pwd)"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SPLIT_FILE="$(realpath "${2:-$REPO/data/split.npz}")"
QM9_DB="$(realpath "${QM9_DB:-$REPO/data/qm9.db}")"
CGSCHNET_DIR="${CGSCHNET_DIR:-$REPO/cG-SchNet}"
JOBS="${JOBS:-1}"
export REPO SPLIT_FILE QM9_DB CGSCHNET_DIR
# one core per run: keep numpy/torch from spawning threads on top of JOBS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

[ -f "$SPLIT_FILE" ] || { echo "Split file not found: $SPLIT_FILE" >&2; exit 1; }
[ -f "$QM9_DB" ] || { echo "QM9 database not found: $QM9_DB" >&2; exit 1; }

# Print the integer following a label in cG-SchNet's output ("" if absent).
count_after() {
  tr '\r' '\n' < "$2" | sed -n "s/^$1 \([0-9][0-9]*\).*/\1/p" | head -n 1
}

score_one() {
  local file="$1"
  local dir name res failed work out rc status pct
  local n_gen n_dup n_uv n_new n_tr n_va n_te
  dir=$(dirname "$file")
  name=$(basename "$file" .mol_dict)
  res="$dir/$name.res"
  failed="$dir/$name.res.failed"

  # cG-SchNet writes its output databases next to the input file, so every run
  # gets its own directory; parallel runs would otherwise clobber each other.
  work=$(mktemp -d)
  ln -s "$file" "$work/$name.mol_dict"
  out="$work/stdout"
  python3 "$REPO/scripts/run_filter_generated.py" "$work/$name.mol_dict" \
    --train_data_path "$QM9_DB" --split_file "$SPLIT_FILE" --threads 0 \
    > "$out" 2> "$work/stderr"
  rc=$?

  n_gen=$(count_after "Number of generated molecules:" "$out")
  n_dup=$(count_after "Number of duplicate molecules:" "$out")
  n_uv=$(count_after "Number of unique and valid molecules:" "$out")
  n_new=$(count_after "Number of new molecules:" "$out")
  n_tr=$(count_after "Number of molecules matching training data:" "$out")
  n_va=$(count_after "Number of molecules matching validation data:" "$out")
  n_te=$(count_after "Number of molecules matching test data:" "$out")

  if [ -z "$n_gen" ] || [ -z "$n_dup" ] || [ -z "$n_uv" ]; then
    status=failed    # crashed before printing its counts
  elif [ "$n_gen" -eq 0 ] || [ "$n_uv" -eq 0 ]; then
    status=no_valid  # expected crash in the statistics printout: all metrics 0
    n_dup=0; n_uv=0; n_new=0; n_tr=0; n_va=0; n_te=0
  elif [ "$rc" -ne 0 ] || [ -z "$n_new" ] || [ -z "$n_tr" ] || [ -z "$n_va" ] || [ -z "$n_te" ]; then
    status=failed
  else
    status=ok
  fi

  if [ "$status" = failed ]; then
    cat "$out" "$work/stderr" > "$failed"
    rm -rf "$work"
    echo "  FAILED (rc=$rc): $name  -> see $name.res.failed"
    return 1
  fi

  # stability rate = (unique valid + duplicates) / generated, as in the paper
  if [ "$n_gen" -eq 0 ]; then
    pct=0
  else
    pct=$(echo "scale=5; ($n_uv + $n_dup) / $n_gen * 100" | bc)
  fi

  {
    echo "Number of generated molecules: $n_gen"
    echo "Number of duplicate molecules: $n_dup"
    echo "Number of unique and valid molecules: $n_uv ($pct%)"
    echo "Number of new molecules: $n_new"
    echo "Number of molecules matching training data: $n_tr"
    echo "Number of molecules matching validation data: $n_va"
    echo "Number of molecules matching test data: $n_te"
    if [ "$status" = ok ]; then
      # atom, bond and ring statistics, as kept in the published .res files
      tr '\r' '\n' < "$out" | sed -n '/^Number of molecules matching test data:/,$p' | tail -n +2
    else
      echo "No valid molecules: all counts set to 0."
    fi
    echo "ID: $name"
  } > "$res.tmp" && mv "$res.tmp" "$res"

  rm -f "$failed"
  rm -rf "$work"
  echo "  $status: $name"
}
export -f count_after score_one

echo "=== 1/3  .db -> .mol_dict"
python3 "$REPO/scripts/convert_db_to_mol_dict.py" "$base_path"

echo "=== 2/3  stability, uniqueness, novelty (cG-SchNet, $JOBS in parallel)"
todo=()
while IFS= read -r -d '' f; do
  if [ -f "${f%.mol_dict}.res" ]; then
    echo "  skip (exists): $(basename "${f%.mol_dict}").res"
  else
    todo+=("$f")
  fi
done < <(find "$base_path" -maxdepth 1 -name '*.mol_dict' -print0 | sort -z)
echo "  ${#todo[@]} file(s) to score"
if [ "${#todo[@]}" -gt 0 ]; then
  # The first file runs alone so that it fills the training-fingerprint cache;
  # the parallel runs after it only load the cache.
  score_one "${todo[0]}"
  if [ "${#todo[@]}" -gt 1 ]; then
    printf '%s\0' "${todo[@]:1}" | xargs -0 -r -n 1 -P "$JOBS" bash -c 'score_one "$1"' _
  fi
fi

echo "=== 3/3  validity and summary (.vun.json, $JOBS in parallel)"
find "$base_path" -maxdepth 1 -name '*.mol_dict' -print0 | sort -z |
  while IFS= read -r -d '' f; do
    stem="${f%.mol_dict}"
    if [ -f "$stem.res" ] && [ ! -f "$stem.vun.json" ]; then
      printf '%s\0' "$f"
    fi
  done | xargs -0 -r -n 1 -P "$JOBS" python3 "$REPO/scripts/compute_vun.py"

n_failed=$(find "$base_path" -maxdepth 1 -name '*.res.failed' | wc -l)
if [ "$n_failed" -gt 0 ]; then
  echo "done, but $n_failed file(s) failed in cG-SchNet: see *.res.failed and re-run." >&2
  exit 1
fi
echo "done."
