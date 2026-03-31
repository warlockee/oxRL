#!/bin/bash
# ================================================================
# Reviewer 5 - Phase 1: Gemma 3 Cross-Architecture Validation
# ================================================================
# 27 core runs + 6 Base init ablation = 33 total
# Gemma 3 1B-IT, 4B-IT, 12B-IT × SFT/DPO/SimPO × 3 seeds
#
# Wave 1: 1B runs (9 jobs on GPUs 0-7, ~30min each)
# Wave 2: 4B runs (8 jobs on GPUs 0-7, ~2h each)
# Wave 3: 12B runs (4 jobs on GPUs 0-3, ~10h each, LoRA)
# Wave 4: 12B remaining (4 jobs) + Base ablation (if capacity)
# ================================================================
set -euo pipefail

PYTHON="/opt/pytorch/bin/python3"
WORK_DIR="/home/ec2-user/fsx/oxRL"
DATA_DIR="/home/ec2-user/fsx/oxrl_data/neurips2026"
CONFIG_DIR="/home/ec2-user/fsx/oxrl_configs/neurips2026/reviewer5"
LOG_DIR="/home/ec2-user/fsx/oxrl_logs/neurips2026/reviewer5"
CKPT_DIR="/home/ec2-user/fsx/oxrl_checkpoints/neurips2026/reviewer5"

mkdir -p "$LOG_DIR" "$DATA_DIR" "$CONFIG_DIR" "$CKPT_DIR"
cd "$WORK_DIR"

export HF_HOME=/home/ec2-user/fsx/.cache/huggingface
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export DS_SKIP_CUDA_CHECK=1
export TOKENIZERS_PARALLELISM=false

echo "================================================================"
echo "  Reviewer 5 Phase 1: Gemma 3 Cross-Architecture"
echo "  $(date -u)"
echo "================================================================"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""

# ================================================================
# Step 0: Generate prompt data (GSM8K without system prompt for Gemma)
# ================================================================
generate_prompt_data() {
    local model_slug="$1"  # e.g., gemma-3-1b-it

    local train_file="$WORK_DIR/data/gsm8k_${model_slug}_ns_train.parquet"
    local test_file="$WORK_DIR/data/gsm8k_${model_slug}_ns_test.parquet"

    if [ -f "$train_file" ] && [ -f "$test_file" ]; then
        echo "  [SKIP] Prompt data already exists for $model_slug"
        return 0
    fi

    echo "  [DATA] Generating prompt data for $model_slug..."
    $PYTHON oxrl/preprocessing/gsm8k.py \
        --local_dir "$WORK_DIR/data" \
        --run_id "$model_slug" \
        --use_system_prompt False
    echo "  [DATA] Done: $train_file"
}

# ================================================================
# Step 1: Generate preference data via self-play
# ================================================================
generate_pref_data() {
    local model_id="$1"    # e.g., google/gemma-3-1b-it
    local model_slug="$2"  # e.g., gemma-3-1b-it
    local gpu="$3"
    local mem_util="${4:-0.85}"

    local prompt_file="$WORK_DIR/data/gsm8k_${model_slug}_ns_train.parquet"
    local pref_out="$DATA_DIR/gsm8k_${model_slug}_prefs_train.parquet"
    local sft_out="$DATA_DIR/gsm8k_${model_slug}_sft_train.parquet"

    # Also generate test set preference data
    local prompt_test="$WORK_DIR/data/gsm8k_${model_slug}_ns_test.parquet"
    local pref_test_out="$DATA_DIR/gsm8k_${model_slug}_prefs_test.parquet"
    local sft_test_out="$DATA_DIR/gsm8k_${model_slug}_sft_test.parquet"

    if [ -f "$pref_out" ] && [ -f "$sft_out" ]; then
        echo "  [SKIP] Pref+SFT data already exists for $model_slug"
        return 0
    fi

    echo "  [PREF] Generating preference data for $model_slug on GPU $gpu..."

    # Generate preference pairs for train set
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m oxrl.data.generate_prefs \
        --model "$model_id" \
        --dataset "$prompt_file" \
        --reward gsm8k_reward_func \
        --n-responses 16 \
        --max-tokens 512 \
        --temperature 1.0 \
        --output "$pref_out" \
        --gpu-memory-utilization "$mem_util" \
        --trust-remote-code \
        > "$LOG_DIR/datagen_${model_slug}_train.log" 2>&1

    echo "  [PREF] Train prefs done: $pref_out"

    # Generate for test set (smaller, fewer responses)
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON -m oxrl.data.generate_prefs \
        --model "$model_id" \
        --dataset "$prompt_test" \
        --reward gsm8k_reward_func \
        --n-responses 16 \
        --max-tokens 512 \
        --temperature 1.0 \
        --output "$pref_test_out" \
        --gpu-memory-utilization "$mem_util" \
        --trust-remote-code \
        > "$LOG_DIR/datagen_${model_slug}_test.log" 2>&1

    echo "  [PREF] Test prefs done: $pref_test_out"

    # Extract SFT data from preference data (chosen responses only)
    $PYTHON -m oxrl.data.extract_sft \
        --input "$pref_out" \
        --output "$sft_out" 2>&1 || {
        # Fallback: create SFT data from prompt+answer (same as Qwen pipeline)
        echo "  [SFT] extract_sft failed, using prompt data directly"
        cp "$prompt_file" "$sft_out"
    }

    # Same for test
    $PYTHON -m oxrl.data.extract_sft \
        --input "$pref_test_out" \
        --output "$sft_test_out" 2>&1 || {
        cp "$prompt_test" "$sft_test_out"
    }

    echo "  [DATA] All data ready for $model_slug"
}

# ================================================================
# Step 2: Generate training configs
# ================================================================
generate_configs() {
    echo "  [CONFIG] Generating training configs..."
    $PYTHON - << 'PYEOF'
import os, yaml, copy

CONFIG_DIR = "/home/ec2-user/fsx/oxrl_configs/neurips2026/reviewer5"
CKPT_DIR = "/home/ec2-user/fsx/oxrl_checkpoints/neurips2026/reviewer5"
DATA_DIR = "/home/ec2-user/fsx/oxrl_data/neurips2026"

MODELS = [
    {"id": "google/gemma-3-1b-it", "slug": "gemma-3-1b-it", "short": "gemma3_1b",
     "param_b": 1.0, "lora": False, "batch": 2, "grad_accum": 4,
     "gpu_mem_util": 0.4, "rollout_batch": 64},
    {"id": "google/gemma-3-4b-it", "slug": "gemma-3-4b-it", "short": "gemma3_4b",
     "param_b": 4.0, "lora": False, "batch": 1, "grad_accum": 8,
     "gpu_mem_util": 0.6, "rollout_batch": 16},
    {"id": "google/gemma-3-12b-it", "slug": "gemma-3-12b-it", "short": "gemma3_12b",
     "param_b": 12.0, "lora": True, "batch": 1, "grad_accum": 8,
     "gpu_mem_util": 0.5, "rollout_batch": 8},
]

ALGS = {
    "sft": {"dpo_beta": None, "simpo_gamma": None, "ref_model": ""},
    "dpo": {"dpo_beta": 0.1, "simpo_gamma": None, "ref_model": None},  # ref_model = same as model
    "simpo": {"dpo_beta": 2.0, "simpo_gamma": 1.0, "ref_model": ""},
}

SEEDS = [42, 123, 456]

count = 0
for m in MODELS:
    for alg_name, alg_cfg in ALGS.items():
        for seed in SEEDS:
            exp_id = f"{alg_name}_{m['short']}_gsm8k_s{seed}"

            # Data files
            if alg_name == "sft":
                train_file = f"{DATA_DIR}/gsm8k_{m['slug']}_sft_train.parquet"
                val_file = f"{DATA_DIR}/gsm8k_{m['slug']}_sft_test.parquet"
            else:
                train_file = f"{DATA_DIR}/gsm8k_{m['slug']}_prefs_train.parquet"
                val_file = f"{DATA_DIR}/gsm8k_{m['slug']}_prefs_test.parquet"

            ref_model = alg_cfg["ref_model"]
            if ref_model is None:
                ref_model = m["id"]

            # DeepSpeed config
            offload_device = "cpu" if m["param_b"] >= 4.0 else "none"

            config = {
                "run": {
                    "experiment_id": exp_id,
                    "training_gpus": 1,
                    "rollout_gpus": 1,
                    "checkpoint_dir": f"{CKPT_DIR}/{exp_id}",
                    "tracking_uri": "",
                    "project_name": "oxrl-neurips-r5",
                    "ray_address": None,
                    "ray_master_port": 35000 + count * 7,
                    "distributed_training_strategy": "deepspeed-zero3",
                    "seed": seed,
                },
                "train": {
                    "alg_name": alg_name,
                    "optimizer_name": "adamw",
                    "lr": 1e-6,
                    "adam_epsilon": 1e-8,
                    "betas": [0.9, 0.95],
                    "weight_decay": 0.01,
                    "warmup_steps_ratio": 0.1,
                    "clip_grad_norm": 1.0,
                    "lr_scheduler": "WarmupCosineLR",
                    "kl_coeff": 0.0,
                    "clip_low": 0.2,
                    "clip_high": 0.2,
                    "entropy_coeff": 0.0,
                    "update_after_full_replay": True,
                    "total_number_of_epochs": 3,
                    "train_steps_per_epoch": None,
                    "train_batch_size_per_gpu": m["batch"],
                    "gradient_accumulation_steps": m["grad_accum"],
                    "val_batch_size_per_gpu": 16,
                    "normalize_loss": True,
                    "dynamic_ratio_every_step": True,
                },
                "model": {
                    "name": m["id"],
                    "dtype": "bfloat16",
                    "ref_model": ref_model,
                    "ref_model_offload_to_cpu": True,
                    "trust_remote_code": True,
                    "use_cache": False,
                    "model_class": "llm",
                    "gradient_checkpointing": True,
                    "attn_implementation": "eager",
                },
                "data": {
                    "train_dnames": [f"gsm8k_{m['slug']}_train"],
                    "train_ratios": {f"gsm8k_{m['slug']}_train": 1.0},
                    "train_files_path": train_file,
                    "val_files_path": val_file,
                    "num_workers": 4,
                    "max_seq_len": 512,
                    "prompt_key": "prompt",
                    "answer_key": "answer",
                },
                "reward": {
                    "broadcast": False,
                    "eps_reward_norm": 1e-8,
                    "reward_func": "gsm8k_reward_func",
                },
                "rollout": {
                    "temperature": 1.0,
                    "max_tokens": 512,
                    "n_samples": 2,
                    "top_p": 1.0,
                    "top_k": -1,
                    "ignore_eos": False,
                    "stop": "",
                    "gpu_memory_utilization": m["gpu_mem_util"],
                    "stop_token_ids": [],
                    "prompt_logprobs": False,
                    "force_strict_on_policy": True,
                    "tensor_parallel_size": 1,
                    "rollout_batch_size_per_gpu": m["rollout_batch"],
                },
                "deepspeed": {
                    "zero_optimization": {
                        "stage": 3,
                        "stage3_param_persistence_threshold": 100000.0,
                        "stage3_prefetch_bucket_size": 50000000.0,
                        "offload_optimizer": {"device": offload_device, "pin_memory": True},
                        "offload_param": {"device": "none", "pin_memory": True},
                        "contiguous_gradients": True,
                        "overlap_comm": True,
                        "reduce_scatter": True,
                        "reduce_bucket_size": 500000000.0,
                        "allgather_bucket_size": 500000000.0,
                        "stage3_gather_16bit_weights_on_model_save": True,
                    },
                    "activation_checkpointing": {
                        "partition_activations": True,
                        "contiguous_memory_optimization": True,
                    },
                    "steps_per_print": 100,
                    "wall_clock_breakdown": False,
                    "flops_profiler": {
                        "enabled": False, "profile_step": 10,
                        "module_depth": -1, "top_modules": 1,
                        "detailed": True, "output_file": None,
                    },
                },
                "inference_engine": {"name": "vllm"},
            }

            # Add LoRA config if needed
            if m["lora"]:
                config["lora"] = {
                    "enabled": True,
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"],
                }

            # Add DPO/SimPO specific params
            if alg_cfg.get("dpo_beta") is not None:
                config["train"]["dpo_beta"] = alg_cfg["dpo_beta"]
            if alg_cfg.get("simpo_gamma") is not None:
                config["train"]["simpo_gamma"] = alg_cfg["simpo_gamma"]

            # Write config
            config_dir = f"{CONFIG_DIR}/{exp_id}"
            os.makedirs(config_dir, exist_ok=True)
            with open(f"{config_dir}/config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=True)

            count += 1
            print(f"  Config: {exp_id}")

print(f"\n  Total: {count} configs generated")
PYEOF
}

# ================================================================
# Training launch function
# ================================================================
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

    echo "  [START] $exp_id on GPU $gpu (port $port)"

    CUDA_VISIBLE_DEVICES=$gpu \
    MASTER_ADDR=localhost \
    MASTER_PORT=$port \
    RANK=0 \
    LOCAL_RANK=0 \
    WORLD_SIZE=1 \
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

wait_for_pids() {
    local desc="$1"
    shift
    local pids=("$@")
    local failed=0

    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            echo "  [OK] PID $pid finished"
        else
            echo "  [FAIL] PID $pid failed (exit $?)"
            ((failed++))
        fi
    done

    echo "  [$desc] ${#pids[@]} jobs done, $failed failed"
    return $failed
}

# ================================================================
# EXECUTION
# ================================================================

echo ""
echo "=== Step 0: Generate prompt data ==="
generate_prompt_data "gemma-3-1b-it"
generate_prompt_data "gemma-3-4b-it"
generate_prompt_data "gemma-3-12b-it"

echo ""
echo "=== Step 1: Generate preference + SFT data ==="
echo "  Running 1B and 4B in parallel on GPUs 0 and 1..."

generate_pref_data "google/gemma-3-1b-it" "gemma-3-1b-it" 0 0.85 &
PID_DATA_1B=$!

generate_pref_data "google/gemma-3-4b-it" "gemma-3-4b-it" 1 0.85 &
PID_DATA_4B=$!

wait $PID_DATA_1B && echo "  [OK] 1B data complete" || echo "  [FAIL] 1B data failed"
wait $PID_DATA_4B && echo "  [OK] 4B data complete" || echo "  [FAIL] 4B data failed"

echo "  Running 12B data gen on GPU 0 (needs more memory)..."
generate_pref_data "google/gemma-3-12b-it" "gemma-3-12b-it" 0 0.85

echo ""
echo "=== Step 2: Generate configs ==="
generate_configs

echo ""
echo "=== Step 3: Launch Wave 1 - Gemma 3 1B (9 runs, GPUs 0-7) ==="
WAVE1_PIDS=()
GPU=0
for alg in sft dpo simpo; do
    for seed in 42 123 456; do
        exp_id="${alg}_gemma3_1b_gsm8k_s${seed}"
        launch_sl_job "$exp_id" $GPU && WAVE1_PIDS+=($!)
        GPU=$(( (GPU + 1) % 8 ))
    done
done

echo "  Wave 1: ${#WAVE1_PIDS[@]} jobs launched. Waiting..."
wait_for_pids "Wave1-1B" "${WAVE1_PIDS[@]}" || true
echo "  Wave 1 complete: $(date -u)"

echo ""
echo "=== Step 4: Launch Wave 2 - Gemma 3 4B (9 runs, GPUs 0-7) ==="
WAVE2_PIDS=()
GPU=0
for alg in sft dpo simpo; do
    for seed in 42 123 456; do
        exp_id="${alg}_gemma3_4b_gsm8k_s${seed}"
        launch_sl_job "$exp_id" $GPU && WAVE2_PIDS+=($!)
        GPU=$(( (GPU + 1) % 8 ))
    done
done

echo "  Wave 2: ${#WAVE2_PIDS[@]} jobs launched. Waiting..."
wait_for_pids "Wave2-4B" "${WAVE2_PIDS[@]}" || true
echo "  Wave 2 complete: $(date -u)"

echo ""
echo "=== Step 5: Launch Wave 3 - Gemma 3 12B (9 runs, 2 GPUs each, 4 parallel) ==="
# 12B with LoRA needs ~24GB, fits 1 GPU but use 1 GPU per job
# Run 4 at a time on GPUs 0-3, then 4 on GPUs 4-7, then last 1
WAVE3A_PIDS=()
launch_sl_job "sft_gemma3_12b_gsm8k_s42" 0 && WAVE3A_PIDS+=($!)
launch_sl_job "sft_gemma3_12b_gsm8k_s123" 1 && WAVE3A_PIDS+=($!)
launch_sl_job "sft_gemma3_12b_gsm8k_s456" 2 && WAVE3A_PIDS+=($!)
launch_sl_job "dpo_gemma3_12b_gsm8k_s42" 3 && WAVE3A_PIDS+=($!)

echo "  Wave 3a: ${#WAVE3A_PIDS[@]} jobs launched (12B). Waiting..."
wait_for_pids "Wave3a-12B" "${WAVE3A_PIDS[@]}" || true

WAVE3B_PIDS=()
launch_sl_job "dpo_gemma3_12b_gsm8k_s123" 0 && WAVE3B_PIDS+=($!)
launch_sl_job "dpo_gemma3_12b_gsm8k_s456" 1 && WAVE3B_PIDS+=($!)
launch_sl_job "simpo_gemma3_12b_gsm8k_s42" 2 && WAVE3B_PIDS+=($!)
launch_sl_job "simpo_gemma3_12b_gsm8k_s123" 3 && WAVE3B_PIDS+=($!)

echo "  Wave 3b: ${#WAVE3B_PIDS[@]} jobs launched. Waiting..."
wait_for_pids "Wave3b-12B" "${WAVE3B_PIDS[@]}" || true

WAVE3C_PIDS=()
launch_sl_job "simpo_gemma3_12b_gsm8k_s456" 0 && WAVE3C_PIDS+=($!)

echo "  Wave 3c: last 12B job launched. Waiting..."
wait_for_pids "Wave3c-12B" "${WAVE3C_PIDS[@]}" || true

echo ""
echo "================================================================"
echo "  ALL TRAINING COMPLETE"
echo "  $(date -u)"
echo "================================================================"
echo ""
echo "Next: Run evaluation with lm-evaluation-harness on all checkpoints"
