#!/bin/bash
# Auto-launch pipeline when 7B training finishes
# Polls every 60s for GPU availability, then kicks off the full pipeline.
set -euo pipefail

LOG="/home/ec2-user/fsx/oxrl_logs/auto_launch.log"
PIPELINE="/home/ec2-user/fsx/oxRL/scripts/neurips_pipeline.sh"

echo "[$(date)] Auto-launcher started. Waiting for 7B jobs to finish..." | tee -a "$LOG"

while true; do
    # Count running deepspeed/training processes
    running=$(ps aux | grep -E "oxrl\.main_sl|oxrl\.main_rl" | grep -v grep | wc -l)
    
    if [ "$running" -eq 0 ]; then
        echo "[$(date)] All training jobs finished! Launching pipeline..." | tee -a "$LOG"
        break
    fi
    
    # Log progress of one representative job
    progress=$(grep -oP 'Epoch \d+/\d+:\s+\d+%' /home/ec2-user/fsx/oxrl_logs/neurips2026/core_math_gsm8k_dpo_qwen7b_gsm8k_s42.log 2>/dev/null | tail -1 || echo "unknown")
    echo "[$(date)] $running training processes still running. DPO 7B: $progress" | tee -a "$LOG"
    sleep 60
done

# Small delay to let GPU memory fully release
sleep 30

# First evaluate the 7B checkpoints that just finished
echo "[$(date)] Evaluating 7B checkpoints first..." | tee -a "$LOG"
cd /home/ec2-user/fsx/oxRL
bash scripts/mass_eval.sh gsm8k 2>&1 | tee -a "$LOG"

# Aggregate results after 7B eval
echo "[$(date)] Aggregating results (including 7B)..." | tee -a "$LOG"
/opt/pytorch/bin/python3 scripts/aggregate_results.py 2>&1 | tail -20 | tee -a "$LOG"
/opt/pytorch/bin/python3 scripts/generate_figures.py 2>&1 | tail -10 | tee -a "$LOG"

# Then launch the full retraining pipeline
echo "[$(date)] Starting full retraining pipeline..." | tee -a "$LOG"
bash "$PIPELINE" 2>&1 | tee -a /home/ec2-user/fsx/oxrl_logs/pipeline.log
echo "[$(date)] Pipeline complete!" | tee -a "$LOG"
