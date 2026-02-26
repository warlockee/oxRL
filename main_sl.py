import os
import random
import numpy as np
import argparse
import deepspeed
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as torch_dist
from tqdm import tqdm
import gc
import time

# imports local methods, classes, etc.
import oxrl.configs.load as cfg# all config arguments
# Import local datasets module directly from file to avoid conflict with HuggingFace 'datasets' package
import importlib.util as _ilu
_spec_pr = _ilu.spec_from_file_location("_prompt_response", os.path.join(os.path.dirname(__file__), "oxrl", "datasets", "prompt_response.py"))
_mod_pr = _ilu.module_from_spec(_spec_pr)
_spec_pr.loader.exec_module(_mod_pr)
PromptResponseDataset = _mod_pr.PromptResponseDataset

_spec_pp = _ilu.spec_from_file_location("_prompt_preference", os.path.join(os.path.dirname(__file__), "oxrl", "datasets", "prompt_preference.py"))
_mod_pp = _ilu.module_from_spec(_spec_pp)
_spec_pp.loader.exec_module(_mod_pp)
PromptPreferenceDataset = _mod_pp.PromptPreferenceDataset

from oxrl.utils.utils import safe_string_to_torch_dtype, get_experiment_dir_name
from oxrl.utils.logging import setup_logging, setup_mlflow, log_metrics, end_run
from oxrl.utils.setup import set_random_seeds, get_distributed_info, load_tokenizer, load_model_and_ref
from oxrl.algs.sft import SFT
from oxrl.algs.dpo import DPO
from oxrl.algs.orpo import ORPO
from oxrl.algs.kto import KTO

SL_ALGORITHMS = {"sft": SFT, "dpo": DPO, "orpo": ORPO, "kto": KTO}

def load_models_and_tokenizer(model_name, model_dtype, ref_model_name, trust_remote_code, attn_impl, rank):
    '''
        Load models and tokenizer.
        It also loads the ref model if provided.
    '''
    # convert string to torch dtype if it is not already
    model_dtype = safe_string_to_torch_dtype(model_dtype)

    model, ref_model = load_model_and_ref(
        model_path=model_name,
        model_dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        attn_impl=attn_impl,
        ref_model_path=ref_model_name
    )

    # 2. Tokenizer initialization
    tokenizer = load_tokenizer(model_name, trust_remote_code=trust_remote_code, rank=rank)

    return model, ref_model, tokenizer

def training_engine_setup(deepspeed_config, model, ref_model=None):
    '''
        This function is responsible for setting up distributed training engine.
        For now, it only supports deepspeed.
    '''
    # Convert pydantic model to python Dict for DeepSpeed
    ds_config_dict = deepspeed_config.model_dump()

    # check to avoid re-initializing distributed backend
    if not torch.distributed.is_initialized():
        # 1. Initialize distributed training engine
        deepspeed.init_distributed()

    # 2. Initialize model engine
    # Filter for trainable parameters (crucial for LoRA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    model_engine, optimizer, _, _ = deepspeed.initialize(
                                                        model=model,
                                                        model_parameters=trainable_params,
                                                        config=ds_config_dict
                                                        )
    ref_model_engine = None
    if ref_model is not None:
        # ref_model is supported here in case if we want to add
        # additional metrics, divergence, etc. Note, ref_model will not be optimized.
        try:
            ref_model.to(model_engine.device)
            ref_model.eval()
            ref_model_engine = ref_model

        except Exception:
            # fallback: initialize with DeepSpeed
            # Use a minimal config for ref model
            ref_ds_config = {
                "train_micro_batch_size_per_gpu": ds_config_dict.get("train_micro_batch_size_per_gpu"),
                "bf16": ds_config_dict.get("bf16"),
                "fp16": ds_config_dict.get("fp16"),
                "zero_optimization": ds_config_dict.get("zero_optimization"),
            }
            ref_model_engine, _, _, _ = deepspeed.initialize(
                                                    model=ref_model,
                                                    config=ref_ds_config
                                                    )

    return model_engine, ref_model_engine, optimizer

def data_loader_setup(params, tokenizer, rank, world_size, batch_size, split):
    '''
       Setup DataLoader for distributed training.
    '''
    # 1. Initialize our custom datasets
    data_path = params.data.train_files_path if split == 'train' else params.data.val_files_path
    
    alg_name = params.train.alg_name.lower()
    if alg_name in ["dpo", "orpo", "kto"]:
        dataset = PromptPreferenceDataset(prompt_key=params.data.prompt_key,
                                          chosen_key=params.data.chosen_key,
                                          rejected_key=params.data.rejected_key,
                                          max_seq_len=params.data.max_seq_len,
                                          tokenizer=tokenizer,
                                          data_path=data_path)
    else:
        dataset = PromptResponseDataset(prompt_key=params.data.prompt_key,
                                        answer_key=params.data.answer_key,
                                        max_seq_len=params.data.max_seq_len,
                                        tokenizer=tokenizer,
                                        data_path=data_path)

    shuffle = True if split == 'train' else False
    drop_last = True if split == 'train' else False

    # 2. Initialize distributed sampler
    sampler = DistributedSampler(dataset,
                                 shuffle=shuffle,
                                 num_replicas=world_size,
                                 rank=rank)

    # 3. Initialize data loader
    def worker_init_fn(worker_id):
        worker_seed = params.run.seed + worker_id + (rank * 100000)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=params.data.num_workers,
                            pin_memory=True,
                            drop_last=drop_last,
                            worker_init_fn=worker_init_fn
                            )

    return dataloader, sampler

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./config/dummy.yaml", help="config file")
    parser.add_argument("--experiment_id", type=str, default="run_1", help="experiment id")
    parser.add_argument("--log-level", type=str, default="INFO", help="logging level")
    args = parser.parse_args()

    ########
    # 1. Setup Environment
    ########
    rank, world_size, local_rank = get_distributed_info()
    logger = setup_logging(rank=rank, log_level=args.log_level)

    ########
    # 2. Load config and other misc. setup
    ########
    config = cfg.load_and_verify(method="sl",
                                 input_yaml=args.config_file,
                                 experiment_id=args.experiment_id,
                                 world_size=world_size,
                                 )
    set_random_seeds(seed=config.run.seed)

    # Setup MLflow (only on rank 0)
    mlflow_run = setup_mlflow(config=config, tracking_uri=config.run.tracking_uri, rank=rank)
    logger.info(f"Config loaded. experiment_id: {config.run.experiment_id}")

    ########
    # 4. load model or previous checkpoints
    ########
    model, ref_model, tokenizer = load_models_and_tokenizer(model_name=config.model.name,
                                                            model_dtype=config.model.dtype,
                                                            ref_model_name=config.model.ref_model,
                                                            trust_remote_code=config.model.trust_remote_code,
                                                            attn_impl=config.model.attn_implementation,
                                                            rank=rank)

    ########
    # 5. Setup trainiing and inference engines
    ########
    model_engine, ref_model_engine, optimizer = training_engine_setup(deepspeed_config=config.deepspeed,
                                                                      model=model,
                                                                      ref_model=ref_model)

    if config.model.gradient_checkpointing:
        logger.info("Gradient checkpointing enabled")
        model_engine.gradient_checkpointing_enable()

    ########
    # 6. Build env or data loader
    ########
    train_dataloader, train_sampler = data_loader_setup(params=config,
                                                        tokenizer=tokenizer,
                                                        batch_size=config.train.train_batch_size_per_gpu,
                                                        split='train',
                                                        world_size=world_size,
                                                        rank=rank)

    val_dataloader, _ = data_loader_setup(params=config,
                                          tokenizer=tokenizer,
                                          batch_size=config.train.val_batch_size_per_gpu,
                                          split='val',
                                          world_size=world_size,
                                          rank=rank)

    ########
    # 7. Intitate the learning algorithm
    ########
    alg_name = config.train.alg_name.lower()
    if alg_name not in SL_ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {alg_name}. Available: {list(SL_ALGORITHMS)}")
    alg_cls = SL_ALGORITHMS[alg_name]

    if alg_name == "sft":
        alg = alg_cls(model_engine=model_engine,
                       optimizer=optimizer,
                       use_cache=config.model.use_cache,
                       normalize_loss=config.train.normalize_loss)
    elif alg_name == "dpo":
        alg = alg_cls(model_engine=model_engine,
                       ref_model_engine=ref_model_engine,
                       optimizer=optimizer,
                       beta=config.train.beta,
                       use_cache=config.model.use_cache)
    elif alg_name == "orpo":
        alg = alg_cls(model_engine=model_engine,
                       optimizer=optimizer,
                       beta=config.train.beta,
                       use_cache=config.model.use_cache)
    elif alg_name == "kto":
        alg = alg_cls(model_engine=model_engine,
                       ref_model_engine=ref_model_engine,
                       optimizer=optimizer,
                       beta=config.train.beta,
                       use_cache=config.model.use_cache)

    ########
    # 8. Training and evaluation loop
    ########
    if rank == 0:
        print("Starting training...")

    total_number_of_train_samples = len(train_dataloader.dataset)
    micro_batches_per_epoch = config.train.micro_batches_per_epoch
    optimizer_steps_per_epoch = micro_batches_per_epoch // config.train.gradient_accumulation_steps

    # Warn if micro_batches_per_epoch is not divisible by gradient_accumulation_steps
    if micro_batches_per_epoch % config.train.gradient_accumulation_steps != 0:
        remainder = micro_batches_per_epoch % config.train.gradient_accumulation_steps
        # raising error to enforce correctness
        raise ValueError(
            f"micro_batches_per_epoch ({micro_batches_per_epoch}) MUST be divisible by "
            f"gradient_accumulation_steps ({config.train.gradient_accumulation_steps}) to ensure "
            "all gradients are applied within the epoch boundaries. "
            f"Adjust configuration. Remainder: {remainder}"
        )

    logger.info("=" * 50)
    logger.info(f"Starting training: {config.train.total_number_of_epochs} epochs")
    logger.info(
        f"Train set: {len(train_dataloader.dataset)} samples | "
        f"micro_batches/epoch={micro_batches_per_epoch} | "
        f"optimizer_steps/epoch={optimizer_steps_per_epoch} | "
        f"grad_accum={config.train.gradient_accumulation_steps}"
    )
    logger.info(
        f"batch_size_per_gpu={config.train.train_batch_size_per_gpu} | "
        f"global_batch_size={config.train.train_batch_size_per_gpu * config.train.gradient_accumulation_steps * world_size}"
    )
    logger.info("=" * 50)
    global_step = 0
    # Sync before starting
    # Ensure all nodes have loaded the model and data before anyone starts iterating
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    for epoch in range(config.train.total_number_of_epochs):
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)
        # Ensure gradients are zeroed at the start of epoch to prevent any bleeding from previous epoch
        # if accumulation steps were not perfectly aligned (though we enforce alignment above).
        model_engine.optimizer.zero_grad()

        ########
        # 8.1 Training loop
        ########
        if rank == 0:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.train.total_number_of_epochs}", disable=(rank != 0))
        else:
            progress_bar = train_dataloader

        # micro_batches_per_epoch = number of micro-batch iterations per epoch.
        # This allows processing a subset of the data per epoch (useful for large datasets).
        # Optimizer steps per epoch = micro_batches_per_epoch // gradient_accumulation_steps

        for step, micro_batch in enumerate(progress_bar):
            # Limit to micro_batches_per_epoch iterations
            if step >= micro_batches_per_epoch:
                break

            # Move batch to gpu (deepspeed engine device)
            micro_batch = {k: v.to(model_engine.device) for k, v in micro_batch.items()}

            # Run one train step for micro-batch.
            metric = alg.train_step(micro_batch)

            # Only increment global_step when ds actually updates weights
            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1

            # logging
            if rank == 0:
                progress_bar.set_postfix(loss=metric['loss'])
                if mlflow_run and model_engine.is_gradient_accumulation_boundary():
                    log_metrics({
                        "train/loss": metric['loss'],
                    }, step=global_step)

        # Sync before validation to ensure consistent state
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{config.train.total_number_of_epochs} completed in {epoch_time:.2f} seconds")

        # Clear graph and to reclaim fragmented memory from training ONCE per epoch
        torch.cuda.empty_cache()
        gc.collect()

        ########
        # 8.2 Validation loop
        ########
        # to be safe and caculate loss average across batches and across GPUs correctly, we use
        # the following instead computes per-rank average and then all-reduces averages
        local_sum   = torch.tensor(0.0, device=model_engine.device)
        local_count = torch.tensor(0.0, device=model_engine.device)

        for data in val_dataloader:
            val_batch = {k: v.to(model_engine.device) for k, v in data.items()}
            val_metric = alg.eval_step(val_batch)
            local_sum += float(val_metric['loss'])
            local_count += 1

        # Aggregate across all ranks. it's safe to do this even if not distributed as it skips.
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_count, op=torch.distributed.ReduceOp.SUM)

        global_avg_loss = (local_sum / torch.clamp(local_count, min=1.0)).item()
        if rank == 0:
            print(f"Epoch {epoch+1}, Validation Loss: {global_avg_loss}")
            if mlflow_run:
                log_metrics({
                    "val/loss": global_avg_loss,
                }, step=global_step)

        ########
        # 8.3 Save checkpoint
        ########
        # Sync before saving to ensure no one is still writing
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        tag = f"iter{epoch+1:06d}"
        model_path = get_experiment_dir_name(output_dir=config.run.checkpoint_dir, tag=tag, experiment_id=config.run.experiment_id)
        logger.info(f"[Epoch {epoch+1}] Saving checkpoint to {model_path}")

        # DeepSpeed handles the collective saving internally so we don't need to worry about different ranks.
        model_engine.save_checkpoint(model_path)

        # Wait for saving to complete on all ranks
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    logger.info("Training completed successfully!")

    # End MLflow run cleanly
    if rank == 0 and mlflow_run:
        end_run()
