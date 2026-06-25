#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — fast end-to-end validation of the whole pipeline.
#
# Runs the EXACT same steps as run_all.sh, but on tiny inputs (a few examples,
# one seed, tiny grids, fewer SHAP samples), so it finishes in minutes and
# exercises every code path / output file. Use it to catch path, config, or
# environment problems BEFORE committing to the multi-hour real run.
#
# The numbers it produces are NOT meaningful results — it only proves the
# pipeline runs and that make_figures.py renders.
#
# USAGE (from the repo root):
#     ./smoke_test.sh            # GPU 1
#     GPU=0 ./smoke_test.sh      # choose GPU
# Any knob can still be overridden, e.g.  DATASETS=sst2 ./smoke_test.sh
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Tiny overrides (each still lets you override the override from the env).
export RUN_LABEL="${RUN_LABEL:-smoke}"
export SAMPLE_SIZE="${SAMPLE_SIZE:-5}"
export ATTACK_SAMPLE_SIZE="${ATTACK_SAMPLE_SIZE:-5}"
export SEEDS="${SEEDS:-1}"
export NUM_COPIES="${NUM_COPIES:-10}"
export COPIES_GRID="${COPIES_GRID:-1,10}"
export SAMPLES_GRID="${SAMPLES_GRID:-5,25}"
export NUM_EXAMPLES="${NUM_EXAMPLES:-1}"
export SHAP_N_SAMPLES="${SHAP_N_SAMPLES:-5}"     # keep SHAP cheap for the smoke run
export ATTACK_METHODS="${ATTACK_METHODS:-DeepWordBug}"  # one attacker is enough to test the path
export DATASETS="${DATASETS:-sst2 agnews}"       # both, to catch dataset-specific bugs
export STOP_ON_FAIL="${STOP_ON_FAIL:-1}"         # fail fast: the point is to surface errors

echo "=================================================================="
echo " SMOKE TEST — tiny inputs, ~minutes. NOT real results."
echo " Validates the full pipeline before you launch run_all.sh."
echo "=================================================================="
exec "$HERE/run_all.sh"