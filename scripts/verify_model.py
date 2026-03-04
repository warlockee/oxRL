#!/usr/bin/env python3
"""
Lightweight model verification script for oxRL model onboarding.

Verifies that a model can be:
1. Loaded via transformers AutoModelForCausalLM
2. Tokenized with chat_template
3. Forward pass produces valid logits
4. Backward pass computes gradients
5. SFT training step completes without error

Usage:
    python scripts/verify_model.py --model "HuggingFaceTB/SmolLM2-360M-Instruct"
    deepspeed --include localhost:4 scripts/verify_model.py --model "HuggingFaceTB/SmolLM2-360M-Instruct"
"""

import argparse
import json
import os
import sys
import time
import gc
import torch
import deepspeed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oxrl.utils.setup import load_tokenizer, load_model_and_ref, ensure_sliding_window_cache, ensure_loss_kwargs, ensure_cache_usable_length, ensure_auto_docstring_union_type


def verify_model(model_id: str, gpu_id: int = 0) -> dict:
    """Run a full verification of model compatibility with oxRL.

    Returns a dict with verification results.
    """
    ensure_sliding_window_cache()
    ensure_loss_kwargs()
    ensure_cache_usable_length()
    ensure_auto_docstring_union_type()

    results = {
        "model_id": model_id,
        "steps": {},
        "success": False,
        "error": None,
    }

    start = time.time()

    # Step 1: Load tokenizer
    print(f"[VERIFY] Step 1: Loading tokenizer for {model_id}...")
    try:
        tokenizer = load_tokenizer(model_id, trust_remote_code=True, rank=0)
        results["steps"]["tokenizer"] = "OK"
        print(f"[VERIFY]   Vocab size: {tokenizer.vocab_size}")
        print(f"[VERIFY]   Pad token: {tokenizer.pad_token}")
    except Exception as e:
        results["steps"]["tokenizer"] = f"FAIL: {e}"
        results["error"] = str(e)
        return results

    # Step 2: Verify chat template
    print(f"[VERIFY] Step 2: Checking chat_template...")
    try:
        test_messages = [{"role": "user", "content": "Hello, how are you?"}]
        formatted = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        results["steps"]["chat_template"] = "OK"
        print(f"[VERIFY]   Chat template works, formatted length: {len(formatted)}")
    except Exception as e:
        results["steps"]["chat_template"] = f"FAIL: {e}"
        results["error"] = str(e)
        return results

    # Step 3: Load model (always to CPU first)
    print(f"[VERIFY] Step 3: Loading model...")
    try:
        model, _ = load_model_and_ref(
            model_path=model_id,
            model_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_impl="eager",
            device_map="cpu",
        )
        param_count = sum(p.numel() for p in model.parameters())
        results["steps"]["model_load"] = "OK"
        results["param_count"] = param_count
        results["param_count_b"] = round(param_count / 1e9, 2)
        print(f"[VERIFY]   Model loaded: {param_count/1e9:.2f}B parameters")
        print(f"[VERIFY]   Model class: {type(model).__name__}")
    except Exception as e:
        results["steps"]["model_load"] = f"FAIL: {e}"
        results["error"] = str(e)
        return results

    # Determine if model is too large for naive GPU placement
    param_b = results.get("param_count_b", 0)
    large_model = param_b > 3.0
    very_large_model = param_b > 20.0
    test_input = "Hello world, this is a test."

    if not large_model:
        # Step 4: Forward pass (small models -- fit on single GPU)
        print(f"[VERIFY] Step 4: Running forward pass...")
        try:
            model.train()
            inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True, max_length=64)

            device = torch.device(f"cuda:{gpu_id}")
            model = model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)

            loss = outputs.loss
            results["steps"]["forward"] = "OK"
            results["initial_loss"] = loss.item()
            print(f"[VERIFY]   Forward pass OK, loss: {loss.item():.4f}")
        except Exception as e:
            results["steps"]["forward"] = f"FAIL: {e}"
            results["error"] = str(e)
            return results

        # Step 5: Backward pass
        print(f"[VERIFY] Step 5: Running backward pass...")
        try:
            loss.backward()
            grad_count = sum(1 for p in model.parameters() if p.grad is not None)
            total_params = sum(1 for p in model.parameters())
            results["steps"]["backward"] = "OK"
            print(f"[VERIFY]   Backward pass OK, {grad_count}/{total_params} params have gradients")
        except Exception as e:
            results["steps"]["backward"] = f"FAIL: {e}"
            results["error"] = str(e)
            return results

        # Reset for DeepSpeed
        model = model.cpu()
        model.zero_grad()
        torch.cuda.empty_cache()
    else:
        # For large models, skip naive GPU placement -- go straight to DeepSpeed
        print(f"[VERIFY] Steps 4-5: Skipping naive forward/backward (model too large: {param_b}B params)")
        print(f"[VERIFY]   Will verify forward+backward via DeepSpeed ZeRO-3 + CPU offloading")
        results["steps"]["forward"] = "SKIPPED (large model, verified via DeepSpeed)"
        results["steps"]["backward"] = "SKIPPED (large model, verified via DeepSpeed)"

    # Ensure model is fully on CPU before DeepSpeed init
    model = model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # Step 6: DeepSpeed initialization (simulated SFT step)
    print(f"[VERIFY] Step 6: Testing DeepSpeed initialization...")
    try:
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()

        if large_model:
            offload_device = "cpu"
            zero_stage = 3
        else:
            offload_device = "none"
            zero_stage = 2

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": {"device": offload_device, "pin_memory": True},
                "offload_param": {"device": offload_device, "pin_memory": True},
                "contiguous_gradients": True,
                "overlap_comm": False if very_large_model else True,
                "stage3_param_persistence_threshold": 1e4 if very_large_model else 1e5,
                "stage3_max_live_parameters": 1e7 if very_large_model else 1e9,
                "stage3_max_reuse_distance": 1e7 if very_large_model else 1e9,
                "stage3_prefetch_bucket_size": 5e6 if very_large_model else 5e7,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "bf16": {"enabled": True},
            "flops_profiler": {"enabled": False},
        }

        # For CPU offloading, let DeepSpeed manage the optimizer
        if offload_device == "cpu":
            ds_config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": 1e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                },
            }
            ds_config["zero_force_ds_cpu_optimizer"] = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        if offload_device == "cpu":
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=trainable_params,
                config=ds_config,
            )
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                model_parameters=trainable_params,
                config=ds_config,
            )

        # Run one training step through DeepSpeed
        inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inputs = {k: v.to(model_engine.device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        outputs = model_engine(**inputs)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()

        results["steps"]["deepspeed"] = "OK"
        results["deepspeed_loss"] = loss.item()
        results["initial_loss"] = loss.item()
        print(f"[VERIFY]   DeepSpeed training step OK, loss: {loss.item():.4f}")
    except Exception as e:
        results["steps"]["deepspeed"] = f"FAIL: {e}"
        results["error"] = str(e)
        return results

    elapsed = time.time() - start
    results["success"] = True
    results["elapsed_seconds"] = round(elapsed, 1)

    print(f"\n[VERIFY] === VERIFICATION PASSED for {model_id} in {elapsed:.1f}s ===")
    print(f"[VERIFY]   Parameters: {results['param_count_b']}B")
    print(f"[VERIFY]   Initial loss: {results['initial_loss']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify model compatibility with oxRL")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed local rank")
    args = parser.parse_args()

    # Set GPU
    if args.local_rank >= 0:
        gpu_id = args.local_rank
    else:
        gpu_id = args.gpu

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))

    results = verify_model(args.model, gpu_id=0)  # 0 because CUDA_VISIBLE_DEVICES remaps

    # Save results
    slug = args.model.split("/")[-1].lower()
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "registry", slug)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "verify_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[VERIFY] Results saved to: {results_path}")

    if not results["success"]:
        print(f"[VERIFY] FAILED: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
