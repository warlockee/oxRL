#!/bin/bash
# ================================================================
# Reviewer 5 - Evaluate Gemma 3 Checkpoints on GSM8K
# ================================================================
set -euo pipefail

PYTHON="/opt/pytorch/bin/python3"
WORK_DIR="/home/ec2-user/fsx/oxRL"
CKPT_BASE="/home/ec2-user/efs/oxrl_archive/reviewer5_checkpoints"
EVAL_DIR="/home/ec2-user/fsx/oxrl_results/eval_reviewer5"
LOG_DIR="/home/ec2-user/fsx/oxrl_logs/neurips2026/reviewer5/eval"

mkdir -p "$EVAL_DIR" "$LOG_DIR"
cd "$WORK_DIR"

export HF_HOME=/home/ec2-user/fsx/.cache/huggingface
export HF_TOKEN=${HF_TOKEN}

echo "================================================================"
echo "  Gemma 3 Evaluation - $(date -u)"
echo "================================================================"

eval_checkpoint() {
    local exp_id="$1"
    local gpu="$2"
    local n_epochs="$3"

    # Find checkpoint path
    local ckpt="$CKPT_BASE/$exp_id/$exp_id/iter$(printf '%06d' $n_epochs)"
    if [ ! -d "$ckpt" ]; then
        ckpt="$CKPT_BASE/$exp_id/iter$(printf '%06d' $n_epochs)"
    fi
    if [ ! -d "$ckpt" ]; then
        echo "  [SKIP] No checkpoint: $exp_id (tried iter$(printf '%06d' $n_epochs))"
        return 1
    fi

    local eval_out="$EVAL_DIR/$exp_id"
    if [ -f "$eval_out/results.json" ]; then
        echo "  [SKIP] Already evaluated: $exp_id"
        return 0
    fi

    mkdir -p "$eval_out"
    echo "  [EVAL] $exp_id on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m oxrl.eval.run_eval \
        --checkpoint "$ckpt" \
        --tasks gsm8k \
        --output-dir "$eval_out" \
        --trust-remote-code \
        > "$LOG_DIR/eval_${exp_id}.log" 2>&1
}

# GPUs to use for eval (free ones)
GPUS=("$@")
if [ ${#GPUS[@]} -eq 0 ]; then
    GPUS=(6 7)
fi

echo "Using GPUs: ${GPUS[*]}"
GPU_IDX=0

# 1B checkpoints (3 epochs = iter000003)
PIDS=()
for alg in sft dpo simpo; do
    for seed in 42 123 456; do
        exp_id="${alg}_gemma3_1b_gsm8k_s${seed}"
        gpu=${GPUS[$GPU_IDX]}
        eval_checkpoint "$exp_id" "$gpu" 3 &
        PIDS+=($!)
        GPU_IDX=$(( (GPU_IDX + 1) % ${#GPUS[@]} ))

        if [ ${#PIDS[@]} -ge ${#GPUS[@]} ]; then
            for pid in "${PIDS[@]}"; do wait "$pid" || true; done
            PIDS=()
        fi
    done
done

# Wait for remaining 1B evals
for pid in "${PIDS[@]}"; do wait "$pid" || true; done
PIDS=()
echo "=== 1B evaluation complete ==="

# 4B checkpoints - check EFS for these
echo "=== Starting 4B evaluation ==="
for alg in sft dpo simpo; do
    for seed in 42 123 456; do
        exp_id="${alg}_gemma3_4b_gsm8k_s${seed}"
        gpu=${GPUS[$GPU_IDX]}
        eval_checkpoint "$exp_id" "$gpu" 3 &
        PIDS+=($!)
        GPU_IDX=$(( (GPU_IDX + 1) % ${#GPUS[@]} ))

        if [ ${#PIDS[@]} -ge ${#GPUS[@]} ]; then
            for pid in "${PIDS[@]}"; do wait "$pid" || true; done
            PIDS=()
        fi
    done
done

for pid in "${PIDS[@]}"; do wait "$pid" || true; done
echo "=== 4B evaluation complete ==="

echo ""
echo "================================================================"
echo "  All evaluations complete - $(date -u)"
echo "================================================================"

# Print summary
echo ""
echo "=== Results Summary ==="
for exp_id in $(ls "$EVAL_DIR" 2>/dev/null | sort); do
    result=$(cat "$EVAL_DIR/$exp_id/results.json" 2>/dev/null | /opt/pytorch/bin/python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d.get(\"gsm8k_accuracy\", d.get(\"accuracy\", \"?\"))}')" 2>/dev/null)
    echo "  $exp_id: $result"
done
