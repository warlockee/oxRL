#!/bin/bash
# ================================================================
# Reviewer 5 - Gemma 3 Training Launch
# ================================================================
# Launches SFT/DPO/SimPO training for Gemma 3 1B, 4B, 12B
# Run this AFTER data generation is complete
# ================================================================
set -euo pipefail

PYTHON="/opt/pytorch/bin/python3"
WORK_DIR="/home/ec2-user/fsx/oxRL"
DATA_DIR="/home/ec2-user/fsx/oxrl_data/neurips2026"
CONFIG_DIR="/home/ec2-user/fsx/oxrl_configs/neurips2026/reviewer5"
LOG_DIR="/home/ec2-user/fsx/oxrl_logs/neurips2026/reviewer5"
CKPT_DIR="/home/ec2-user/fsx/oxrl_checkpoints/neurips2026/reviewer5"

cd "$WORK_DIR"
export HF_HOME=/home/ec2-user/fsx/.cache/huggingface
export HF_TOKEN=${HF_TOKEN}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export DS_SKIP_CUDA_CHECK=1
export TOKENIZERS_PARALLELISM=false

echo "================================================================"
echo "  Gemma 3 Training - $(date -u)"
echo "================================================================"

# First extract SFT data from preference data
echo "=== Extracting SFT data ==="
for slug in gemma-3-1b-it gemma-3-4b-it gemma-3-12b-it; do
    for split in train test; do
        pref="$DATA_DIR/gsm8k_${slug}_prefs_${split}.parquet"
        sft="$DATA_DIR/gsm8k_${slug}_sft_${split}.parquet"
        if [ -f "$pref" ] && [ ! -f "$sft" ]; then
            echo "  Extracting SFT from $pref..."
            $PYTHON -m oxrl.data.extract_sft --input "$pref" --output "$sft"
        elif [ -f "$sft" ]; then
            echo "  [SKIP] $sft already exists"
        else
            echo "  [WAIT] $pref not ready yet"
        fi
    done
done

launch_sl_job() {
    local exp_id="$1"
    local gpu="$2"
    local cfg="$CONFIG_DIR/$exp_id/config.yaml"

    if [ ! -f "$cfg" ]; then
        echo "  [SKIP] Config not found: $cfg"
        return 1
    fi

    if [ -f "$CKPT_DIR/$exp_id/training_complete" ]; then
        echo "  [SKIP] Already completed: $exp_id"
        return 0
    fi

    local port=$((35000 + gpu * 100 + RANDOM % 50))

    echo "  [START] $exp_id on GPU $gpu (port $port) at $(date -u +%H:%M:%S)"

    CUDA_VISIBLE_DEVICES=$gpu \
    MASTER_ADDR=localhost \
    MASTER_PORT=$port \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
    $PYTHON -m deepspeed.launcher.runner \
        --include="localhost:$gpu" \
        --master_port "$port" \
        --module oxrl.main_sl \
        --config-file "$cfg" \
        --experiment_id "$exp_id" \
        > "$LOG_DIR/${exp_id}.log" 2>&1 &

    echo $! > "$LOG_DIR/${exp_id}.pid"
    return 0
}

wait_wave() {
    local desc="$1"; shift
    local pids=("$@")
    local failed=0
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            echo "  [OK] PID $pid done"
        else
            echo "  [FAIL] PID $pid (exit $?)"
            ((failed++))
        fi
    done
    echo "  [$desc] ${#pids[@]} done, $failed failed at $(date -u)"
}

# ================================================================
# Wave 1: Gemma 3 1B (9 runs, ~30min each, all 8 GPUs)
# ================================================================
echo ""
echo "=== Wave 1: Gemma 3 1B (9 runs) ==="
W1=()
GPU=0
for alg in sft dpo simpo; do
    for seed in 42 123 456; do
        launch_sl_job "${alg}_gemma3_1b_gsm8k_s${seed}" $GPU && W1+=($!)
        GPU=$(( (GPU + 1) % 8 ))
    done
done
wait_wave "Wave1-1B" "${W1[@]}" || true

# ================================================================
# Wave 2: Gemma 3 4B (9 runs, ~2h each, all 8 GPUs)
# ================================================================
echo ""
echo "=== Wave 2: Gemma 3 4B (9 runs) ==="
W2=()
GPU=0
for alg in sft dpo simpo; do
    for seed in 42 123 456; do
        launch_sl_job "${alg}_gemma3_4b_gsm8k_s${seed}" $GPU && W2+=($!)
        GPU=$(( (GPU + 1) % 8 ))
    done
done
wait_wave "Wave2-4B" "${W2[@]}" || true

# ================================================================
# Wave 3: Gemma 3 12B (9 runs, ~10h each, 4 at a time)
# ================================================================
echo ""
echo "=== Wave 3a: Gemma 3 12B (4 runs) ==="
W3A=()
launch_sl_job "sft_gemma3_12b_gsm8k_s42" 0 && W3A+=($!)
launch_sl_job "sft_gemma3_12b_gsm8k_s123" 1 && W3A+=($!)
launch_sl_job "sft_gemma3_12b_gsm8k_s456" 2 && W3A+=($!)
launch_sl_job "dpo_gemma3_12b_gsm8k_s42" 3 && W3A+=($!)
wait_wave "Wave3a-12B" "${W3A[@]}" || true

echo ""
echo "=== Wave 3b: Gemma 3 12B (4 runs) ==="
W3B=()
launch_sl_job "dpo_gemma3_12b_gsm8k_s123" 0 && W3B+=($!)
launch_sl_job "dpo_gemma3_12b_gsm8k_s456" 1 && W3B+=($!)
launch_sl_job "simpo_gemma3_12b_gsm8k_s42" 2 && W3B+=($!)
launch_sl_job "simpo_gemma3_12b_gsm8k_s123" 3 && W3B+=($!)
wait_wave "Wave3b-12B" "${W3B[@]}" || true

echo ""
echo "=== Wave 3c: Gemma 3 12B (1 run) ==="
W3C=()
launch_sl_job "simpo_gemma3_12b_gsm8k_s456" 0 && W3C+=($!)
wait_wave "Wave3c-12B" "${W3C[@]}" || true

echo ""
echo "================================================================"
echo "  ALL GEMMA 3 TRAINING COMPLETE - $(date -u)"
echo "================================================================"
