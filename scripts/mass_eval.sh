#!/bin/bash
# ============================================================
# Mass Evaluation — evaluate all checkpoints in parallel
#
# Runs lm-evaluation-harness on ALL checkpoints that don't
# have eval results yet. Uses all 8 GPUs (1 eval per GPU).
# ============================================================
set -euo pipefail

PYTHON="/opt/pytorch/bin/python3"
WORK_DIR="/home/ec2-user/fsx/oxRL"
CKPT_DIR="/home/ec2-user/fsx/oxrl_checkpoints/neurips2026"
EVAL_DIR="/home/ec2-user/fsx/oxrl_results/eval"
TASKS="${1:-gsm8k}"  # default: gsm8k, can pass "gsm8k,math" etc.

cd "$WORK_DIR"
export HF_HOME=/home/ec2-user/fsx/.cache/huggingface

echo "============================================================"
echo "Mass Evaluation Pipeline - $(date)"
echo "Tasks: $TASKS"
echo "============================================================"

# Collect all unevaluated checkpoints
declare -a PENDING=()

for exp_dir in "$CKPT_DIR"/*/; do
    [ ! -d "$exp_dir" ] && continue
    exp_id=$(basename "$exp_dir")

    # Find latest checkpoint with config.json
    latest_ckpt=$(find "$exp_dir" -name "config.json" -path "*/iter*" 2>/dev/null | sort | tail -1)
    [ -z "$latest_ckpt" ] && continue
    ckpt_path=$(dirname "$latest_ckpt")

    # Check if already evaluated for ALL requested tasks
    eval_out="$EVAL_DIR/$exp_id"
    already_done=true
    for task in $(echo "$TASKS" | tr ',' ' '); do
        if ! find "$eval_out" -name "results_*.json" 2>/dev/null | head -1 | grep -q .; then
            already_done=false
            break
        fi
    done
    if $already_done; then
        echo "  SKIP: $exp_id (already evaluated)"
        continue
    fi

    PENDING+=("$exp_id|$ckpt_path")
done

TOTAL=${#PENDING[@]}
echo "Found $TOTAL checkpoints to evaluate"

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to evaluate!"
    exit 0
fi

# Launch evaluations in parallel (8 at a time, 1 GPU each)
RUNNING=0
GPU=0
declare -a PIDS=()
declare -a GPU_MAP=()

eval_one() {
    local gpu=$1
    local exp_id=$2
    local ckpt_path=$3
    local eval_out="$EVAL_DIR/$exp_id"
    local safe_path=$(echo "$ckpt_path" | tr '/' '_')

    mkdir -p "$eval_out/$safe_path"

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m oxrl.eval.run_eval \
        --checkpoint "$ckpt_path" \
        --tasks "$TASKS" \
        --output-dir "$eval_out/$safe_path" \
        --trust-remote-code \
        > "$eval_out/${exp_id}_eval.log" 2>&1

    return $?
}

DONE=0
IDX=0

while [ $IDX -lt $TOTAL ] || [ ${#PIDS[@]} -gt 0 ]; do
    # Launch jobs up to 8 GPUs
    while [ $IDX -lt $TOTAL ] && [ ${#PIDS[@]} -lt 8 ]; do
        entry="${PENDING[$IDX]}"
        exp_id="${entry%%|*}"
        ckpt_path="${entry##*|}"

        gpu=${#PIDS[@]}
        echo "  [START] GPU $gpu: $exp_id"

        eval_one $gpu "$exp_id" "$ckpt_path" &
        PIDS+=($!)
        GPU_MAP+=("$exp_id")
        IDX=$((IDX + 1))
    done

    # Wait for any job to finish
    if [ ${#PIDS[@]} -gt 0 ]; then
        wait -n 2>/dev/null || true
        # Check which PIDs finished
        NEW_PIDS=()
        NEW_MAP=()
        for i in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                NEW_PIDS+=("${PIDS[$i]}")
                NEW_MAP+=("${GPU_MAP[$i]}")
            else
                DONE=$((DONE + 1))
                echo "  [DONE $DONE/$TOTAL] ${GPU_MAP[$i]}"
            fi
        done
        PIDS=("${NEW_PIDS[@]+"${NEW_PIDS[@]}"}")
        GPU_MAP=("${NEW_MAP[@]+"${NEW_MAP[@]}"}")
    fi
done

echo ""
echo "============================================================"
echo "Mass evaluation complete: $DONE/$TOTAL checkpoints evaluated"
echo "Results in: $EVAL_DIR"
echo "============================================================"
