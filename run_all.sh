#!/usr/bin/env bash
# =============================================================================
# run_all.sh — ShapSelfDenoise full figure pipeline (see RUN_GUIDE.md)
#
# Runs every data-generation step in order, then renders all figures. Each step
# is logged to logs/<run>/<step>.log; a per-step failure is recorded but does
# NOT abort the rest (so one bad attack run can't waste the whole night). A
# pass/fail summary is printed at the end.
#
# USAGE (run from the repo root, inside tmux):
#     tmux new -s shap                 # start a session
#     conda activate <env> / source venv/bin/activate
#     ./run_all.sh                     # full run on GPU 1
#     GPU=0 ./run_all.sh               # choose GPU
#     SAMPLE_SIZE=50 SEEDS=2 ./run_all.sh   # override any knob (see below)
#   detach with Ctrl-b then d; reattach later with:  tmux attach -t shap
#
# Every knob below can be overridden from the environment.
# =============================================================================
set -u
set -o pipefail

# Pick the Python interpreter — prefer the project venv so you don't have to
# remember to activate it in every tmux pane. Override with PYTHON=... .
if [ -n "${PYTHON:-}" ]; then
    :                                                   # honor explicit choice
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"                  # an activated venv
elif [ -x "./venv/bin/python" ]; then
    PYTHON="./venv/bin/python"                          # repo-local venv/
elif [ -x "./.venv/bin/python" ]; then
    PYTHON="./.venv/bin/python"                         # repo-local .venv/
else
    PYTHON="$(command -v python || command -v python3)" # last resort
fi
export CUDA_VISIBLE_DEVICES="${GPU:-1}"        # single GPU, always
# Reduce CUDA fragmentation across the many model loads/frees in one run.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PRECISION="${PRECISION:-half}"                  # half precision, always
DATASETS="${DATASETS:-sst2 agnews}"
ATTACK_METHODS="${ATTACK_METHODS:-DeepWordBug TextBugger}"
BATCHSIZE="${BATCHSIZE:-16}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"               # timing + sweeps + certify
ATTACK_SAMPLE_SIZE="${ATTACK_SAMPLE_SIZE:-200}" # SelfDenoise paper Table 1 used 200
SEEDS="${SEEDS:-3}"
NUM_COPIES="${NUM_COPIES:-100}"
MASK_RATE="${MASK_RATE:-0.05}"                  # ShapSelfDenoise
SELFDENOISE_MASK_RATE="${SELFDENOISE_MASK_RATE:-0.05}"  # SelfDenoise (paper: 5%)
SHAP_N_SAMPLES="${SHAP_N_SAMPLES:-25}"
COPIES_GRID="${COPIES_GRID:-1,5,10,25,50,100}"
SAMPLES_GRID="${SAMPLES_GRID:-5,10,25,50,100}"
NUM_EXAMPLES="${NUM_EXAMPLES:-3}"
SHAP_DATASET="${SHAP_DATASET:-sst2}"            # shap_samples sweep is SST-2 only
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"               # set 1 to abort on first failure
RUN_LABEL="${RUN_LABEL:-run}"

LOG_DIR="logs/${RUN_LABEL}_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"
RESULTS=()
SCRIPT_START=$(date +%s)

run_step() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    echo
    echo "============================================================"
    echo ">>> START $name   $(date '+%F %T')"
    echo ">>> $*"
    echo ">>> log: $logfile"
    echo "============================================================"
    local s; s=$(date +%s)
    "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    local d=$(( $(date +%s) - s ))
    if [ "$rc" -eq 0 ]; then
        echo ">>> OK   $name  (${d}s)"
        RESULTS+=("OK    ${name}  (${d}s)")
    else
        echo ">>> FAIL $name  (rc=$rc, ${d}s)"
        RESULTS+=("FAIL  ${name}  (rc=$rc, ${d}s)")
        if [ "$STOP_ON_FAIL" = "1" ]; then
            echo ">>> STOP_ON_FAIL=1 — aborting."
            print_summary
            exit "$rc"
        fi
    fi
}

print_summary() {
    local total=$(( $(date +%s) - SCRIPT_START ))
    echo
    echo "############################################################"
    echo "# Pipeline finished $(date '+%F %T')   (total ${total}s)"
    echo "############################################################"
    local nfail=0
    for r in "${RESULTS[@]:-}"; do
        echo "  $r"
        [[ "$r" == FAIL* ]] && nfail=$((nfail+1))
    done
    echo
    echo "Figures: figures/output/*.pdf (+ .png)    Logs: $LOG_DIR"
    [ "$nfail" -gt 0 ] && echo "WARNING: $nfail step(s) failed — see the logs above."
}

# ---- preflight --------------------------------------------------------------
if [ ! -f main.py ] || [ ! -f comparet_time.py ]; then
    echo "ERROR: run this from the repo root (main.py / comparet_time.py not found)."
    exit 1
fi
if [ -z "${PYTHON:-}" ] || ! "$PYTHON" -c "import numpy" >/dev/null 2>&1; then
    echo "ERROR: '${PYTHON:-<none>}' can't import numpy — wrong interpreter."
    echo "       Your dependencies live in the venv. Fix with either:"
    echo "         source venv/bin/activate     && ./run_all.sh"
    echo "         PYTHON=venv/bin/python ./run_all.sh"
    exit 1
fi
echo "ShapSelfDenoise pipeline — start $(date '+%F %T')"
echo "  python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
echo "  GPU=$CUDA_VISIBLE_DEVICES  precision=$PRECISION  datasets='$DATASETS'"
echo "  sample_size=$SAMPLE_SIZE  attack_sample_size=$ATTACK_SAMPLE_SIZE  seeds=$SEEDS  copies=$NUM_COPIES"
echo "  mask_rate(shap)=$MASK_RATE  mask_rate(selfdenoise)=$SELFDENOISE_MASK_RATE  copies_grid=$COPIES_GRID"
echo "  logs -> $LOG_DIR    (tip: 'tail -f $LOG_DIR/<step>.log' in another pane)"

# ---- Step 1: timing  ->  time_bar, time_distribution ------------------------
for ds in $DATASETS; do
    run_step "01_time_${ds}" \
        "$PYTHON" comparet_time.py --dataset "$ds" --precision "$PRECISION" \
            --batchsize "$BATCHSIZE" --sample-size "$SAMPLE_SIZE" --num-copies "$NUM_COPIES" \
            --mask-rate "$MASK_RATE" --selfdenoise-mask-rate "$SELFDENOISE_MASK_RATE" \
            --shap-n-samples "$SHAP_N_SAMPLES"
done

# ---- Step 2: ensemble sweep  ->  pareto, time_vs_ensemble, acc_vs_ensemble --
run_step "02_ensemble" \
    "$PYTHON" figures/collect_figure_data.py --task ensemble --dataset both \
        --precision "$PRECISION" --batchsize "$BATCHSIZE" --sample-size "$SAMPLE_SIZE" \
        --seeds "$SEEDS" --mask-rate "$MASK_RATE" \
        --selfdenoise-mask-rate "$SELFDENOISE_MASK_RATE" --copies-grid "$COPIES_GRID"

# ---- Step 3: selection ablation  ->  ablation -------------------------------
run_step "03_selection" \
    "$PYTHON" figures/collect_figure_data.py --task selection --dataset both \
        --precision "$PRECISION" --batchsize "$BATCHSIZE" --sample-size "$SAMPLE_SIZE" \
        --seeds "$SEEDS" --mask-rate "$MASK_RATE"

# ---- Step 4: SHAP-samples sweep (SST-2 only)  ->  shap_samples --------------
run_step "04_shap_samples" \
    "$PYTHON" figures/collect_figure_data.py --task shap_samples --dataset "$SHAP_DATASET" \
        --precision "$PRECISION" --batchsize "$BATCHSIZE" --sample-size "$SAMPLE_SIZE" \
        --seeds "$SEEDS" --mask-rate "$MASK_RATE" --samples-grid "$SAMPLES_GRID"

# ---- Step 5: heatmap examples  ->  heatmap ----------------------------------
run_step "05_examples" \
    "$PYTHON" figures/collect_figure_data.py --task examples --dataset both \
        --precision "$PRECISION" --batchsize "$BATCHSIZE" \
        --mask-rate "$MASK_RATE" --num-examples "$NUM_EXAMPLES"

# ---- Step 6: attack (bug-fixed)  ->  robustness -----------------------------
for ds in $DATASETS; do
    for m in $ATTACK_METHODS; do
        run_step "06_attack_${ds}_${m}" \
            "$PYTHON" main.py --mode attack --dataset "$ds" --method "$m" --defence shap \
                --precision "$PRECISION" --batchsize "$BATCHSIZE" \
                --sample-size "$ATTACK_SAMPLE_SIZE" --mask_word "<mask>"
    done
done

# ---- Step 7: certify (mask-rate sweep)  ->  maskrate ------------------------
for ds in $DATASETS; do
    run_step "07_certify_${ds}" \
        "$PYTHON" main.py --mode certify --dataset "$ds" --precision "$PRECISION" \
            --batchsize "$BATCHSIZE" --sample-size "$SAMPLE_SIZE" --mask_word "<mask>"
done

# ---- Step 8: render every figure that has data ------------------------------
run_step "08_figures" "$PYTHON" figures/make_figures.py

print_summary