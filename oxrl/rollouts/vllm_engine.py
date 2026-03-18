import os
import io
import base64
import tempfile
import torch
import gc
import ray
from vllm import LLM
from typing import Optional, List, Callable, Any, Dict

from oxrl.utils.setup import ensure_sliding_window_cache
from oxrl.rollouts.sampling import make_sampling_params
from oxrl.rollouts.logprob_utils import extract_logprobs
from oxrl.rollouts.reward_scoring import score_response, normalize_rewards

ensure_sliding_window_cache()


def _decode_multimodal(prompts: List[Dict]) -> List[Dict]:
    """Strip metadata and decode base64 multimodal data for vLLM."""
    from PIL import Image
    import soundfile as sf

    vllm_prompts = []
    for p in prompts:
        new_p = {k: v for k, v in p.items() if k not in ["metadata", "prompt_structured"]}
        if "prompt_structured" in p:
            new_p["prompt"] = p["prompt_structured"]

        if "multi_modal_data" in new_p:
            if "image" in new_p["multi_modal_data"]:
                img_str = new_p["multi_modal_data"]["image"]
                img_data = base64.b64decode(img_str)
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                new_p["multi_modal_data"] = {"image": image}
            elif "audio" in new_p["multi_modal_data"]:
                audio_str = new_p["multi_modal_data"]["audio"]
                audio_data_bytes = base64.b64decode(audio_str)
                audio_data, sample_rate = sf.read(io.BytesIO(audio_data_bytes))
                new_p["multi_modal_data"] = {"audio": (audio_data, sample_rate)}
        vllm_prompts.append(new_p)
    return vllm_prompts


@ray.remote
class VLLMRolloutEngine:
    def __init__(self,
                 seed:int,
                 model_path: str,
                 trust_remote_code: bool,
                 temperature: float,
                 max_tokens: int,
                 n_samples: int,
                 top_p: float,
                 top_k: int,
                 ignore_eos: bool,
                 stop: Optional[List[str]],
                 stop_token_ids: Optional[List[int]],
                 prompt_logprobs: bool,
                 force_strict_on_policy: bool,
                 reward_func: Callable,
                 tensor_parallel_size: int,
                 eos_id: int,
                 reward_broadcast: bool,
                 eps_reward_norm: float,
                 gpu_memory_utilization: float,
                 engine_id: int = 0,
                 ):


        # reward function
        self.reward_func = reward_func
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.eos_id = eos_id

        # sampling config
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.n_samples = int(n_samples)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.seed = seed
        self.ignore_eos = bool(ignore_eos)
        self.stop = stop if stop else None
        self.stop_token_ids = stop_token_ids if stop_token_ids else None
        self.prompt_logprobs = prompt_logprobs
        self.force_strict_on_policy = bool(force_strict_on_policy)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.engine_id = int(engine_id)

        # vllm engine config
        self.model_path = model_path
        self._init_model_path = model_path  # preserve for tokenizer lookup
        self.loaded_version = -1
        self.trust_remote_code = trust_remote_code
        self.vllm_engine = None
        self.refresh_model(model_path, 0)
        self.sampling_params = make_sampling_params(
            seed=self.seed,
            n_samples=self.n_samples,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.ignore_eos,
            prompt_logprobs=self.prompt_logprobs,
            force_strict_on_policy=self.force_strict_on_policy,
        )

        # reward normalization
        self.eps_reward_norm = float(eps_reward_norm)
        # If True, broadcast a single scalar reward across all tokens in the sequence.
        self.reward_broadcast = bool(reward_broadcast)

    def log(self, msg: str) -> None:
        '''
            Log message only if this is the first engine to avoid clutter.
        '''
        if self.engine_id == 0:
            print(f"[VLLMEngine][Rank {self.engine_id}] {msg}")

    def _copy_tokenizer_to_tmpdir(self, tmpdir: str) -> None:
        """Copy tokenizer files into tmpdir so vLLM can load the model.

        Tries three sources in order:
        1. The original model_path on the init (could be HF cache dir or model ID)
        2. The HuggingFace hub cache for the model
        3. AutoTokenizer.from_pretrained() download
        """
        import shutil
        import glob

        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json",
            "vocab.json", "merges.txt", "special_tokens_map.json",
            "generation_config.json", "added_tokens.json",
        ]
        os.makedirs(tmpdir, exist_ok=True)

        # If any tokenizer file already exists in tmpdir, skip
        if os.path.exists(os.path.join(tmpdir, "tokenizer_config.json")):
            return

        # Source 1: the original init model_path (HF cache snapshot dir)
        # self._init_model_path stores the original model path from __init__
        init_path = getattr(self, "_init_model_path", self.model_path)

        # Try to resolve HF model ID to cache dir
        src_dir = None
        if os.path.isdir(init_path):
            src_dir = init_path
        else:
            # It's an HF model ID like "Qwen/Qwen2.5-0.5B-Instruct"
            # Try to find the cached snapshot
            try:
                from huggingface_hub import snapshot_download
                src_dir = snapshot_download(init_path, local_files_only=True)
            except Exception:
                pass

        if src_dir and os.path.isdir(src_dir):
            copied = 0
            for fname in tokenizer_files:
                src = os.path.join(src_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(tmpdir, fname))
                    copied += 1
            if copied > 0:
                self.log(f"Copied {copied} tokenizer files from {src_dir}")
                return

        # Source 2: AutoTokenizer download
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                init_path, trust_remote_code=self.trust_remote_code
            )
            tokenizer.save_pretrained(tmpdir)
            self.log(f"Saved tokenizer from AutoTokenizer({init_path})")
        except Exception as e:
            self.log(f"WARNING: Could not copy tokenizer files: {e}")


    def refresh_model(self, model_path: str, version: int) -> bool:
        '''
           Refresh model only if version changed.
        '''
        if self.vllm_engine is not None and \
           self.loaded_version == version and \
           model_path == self.model_path:
            self.log(f"Model already at version {version}, skipping refresh")
            return False

        self.log(f"Refreshing model to version {version} from {model_path}")

        # only for local paths not HF model identifier (e.g., google/gemma-3-1b-it)
        if os.path.exists(model_path):
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config.json not found in {model_path}")

        self.model_path = model_path
        self.load_model()
        self.loaded_version = version
        self.log(f"Model refreshed to version {version}")
        return True

    def load_model(self) -> None:
        '''
           Load vLLM engine with cleanup and error handling steps.
        '''
        if self.vllm_engine is not None:
            # delete the old engine and free up memory
            try:
                del self.vllm_engine
            except Exception as e:
                print(f"Error deleting vllm_engine: {e}")

            self.vllm_engine = None
            # memory cleanup
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            # more cleanup
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass

        # Load new model
        # Disable V1 multiprocessing to keep everything in the same process.
        # This avoids subprocess GPU isolation issues when running under Ray.
        # (VLLM_USE_V1=0 is no longer recognized in vLLM >= 0.15; V1 is the only engine.)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        try:
            self.vllm_engine = LLM(model=self.model_path,
                                   trust_remote_code=self.trust_remote_code,
                                   tensor_parallel_size=self.tensor_parallel_size,
                                   gpu_memory_utilization=self.gpu_memory_utilization,
                                   enforce_eager=True,
                                  )
            self.log(f"Successfully loaded vLLM model from {self.model_path}")

        except Exception as e:
            print(f"Failed to load vLLM model from {self.model_path}: {e}")
            self.vllm_engine = None
            raise

    def refresh_model_from_state_dict(self, state_dict_ref, config_dict, version: int) -> bool:
        """Refresh model weights from a Ray object store reference.

        Writes the state dict to a local tmpdir as model.safetensors + config.json,
        then rebuilds the vLLM engine from that directory.

        Note: vLLM V1 (>=0.17) does not support in-place weight reload via
        collective_rpc('update_config') / collective_rpc('reload_weights').
        We always do a full engine rebuild, which is safe and reliable.

        Args:
            state_dict_ref: Ray ObjectRef pointing to the state dict.
            config_dict: Model config as a dict (for config.json).
            version: Policy version number.

        Returns:
            True if model was refreshed, False if skipped (already at version).
        """
        if self.loaded_version == version:
            self.log(f"Model already at version {version}, skipping refresh")
            return False

        self.log(f"Refreshing model to version {version} from object store...")

        # 1. Retrieve state dict from Ray object store
        # Note: Ray auto-resolves ObjectRef args, so state_dict_ref
        # is already an OrderedDict when it arrives here.
        state_dict = state_dict_ref

        # Pop metadata keys (they shouldn't be written as weights)
        state_dict.pop("__model_config_dict__", None)
        state_dict.pop("__value_head_state_dict__", None)

        # 2. Write to a persistent local tmpdir (reuse across refreshes)
        from oxrl.tools.checkpoint import save_state_dict_to_safetensors, save_config_json

        tmpdir = os.path.join(tempfile.gettempdir(), f"oxrl_vllm_e{self.engine_id}")
        save_state_dict_to_safetensors(tmpdir, state_dict)
        del state_dict  # free memory

        # vLLM requires config.json + tokenizer files to load from a local dir.
        # Save config from the gathered state dict metadata.
        if config_dict is not None:
            save_config_json(tmpdir, config_dict)

        # Copy tokenizer files from the original model into the tmpdir.
        # vLLM V1 needs the tokenizer to initialize the engine (for chat
        # template, EOS detection, etc). Without these files, LLM() raises
        # "expected str, bytes or os.PathLike object, not NoneType".
        self._copy_tokenizer_to_tmpdir(tmpdir)

        # 3. Full engine rebuild from tmpdir (always reliable)
        self.model_path = tmpdir
        self.load_model()
        self.loaded_version = version
        self.log(f"Model refreshed to version {version} (rebuild from {tmpdir})")
        return True

    def generate(self,
                 prompts: List[Dict[str, List[int]]],
                 current_iter: int,
                 policy_version: int) -> List[Dict[str, Any]]:
        '''
        prompts: [{'prompt_token_ids': [2,..]}, {'prompt_token_ids': [...]}, ...]
        Returns a list of rollout samples. length ~ B * n_samples.

        token-aligned and prediction-aligned logprobs/mask/done are returned.
        Prediction-aligned means: logit position t predicts token at t+1 (SFT-style shift).
        '''
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise TypeError(f"prompts must be a non-empty list, got {type(prompts)}")

        if self.force_strict_on_policy and int(policy_version) != int(self.loaded_version):
            raise ValueError(
                f"Off-policy rollout: policy_version={int(policy_version)} "
                f"but loaded_version={int(self.loaded_version)}. ")

        assert self.vllm_engine is not None, f"{self.model_path} not loaded."

        # Strip metadata and decode multimodal data for vLLM
        metadata_list = [p.get("metadata") for p in prompts]
        vllm_prompts = _decode_multimodal(prompts)

        self.log(f"Generating completions for {len(vllm_prompts)} prompts with {self.n_samples} samples each")
        generated_outputs = self.vllm_engine.generate(vllm_prompts,
                                                     sampling_params=self.sampling_params,
                                                     use_tqdm=False)
        self.log(f"Generation complete for {len(vllm_prompts)} prompts")

        # generated_outputs has prompt_ids and other outputs
        # this works even if n_samples >= 1
        rollout_samples = []
        for prompt_idx, data in enumerate(generated_outputs):
            group_samples = []
            group_stats   = {'rewards': [], 'lengths': []}
            prompt_mm_data = prompts[prompt_idx].get("multi_modal_data", None)
            prompt_ids = list(data.prompt_token_ids or [])
            prompt_len = len(prompt_ids)
            if prompt_len == 0:
                raise ValueError(f"No prompt token ids found in generated output: {data}")

            # process generated responses
            for response in data.outputs:
                response_ids = list(response.token_ids)
                response_len = len(response_ids)
                finish_reason = getattr(response, "finish_reason", None)
                stop_reason   = getattr(response, "stop_reason", None)

                # all have length [T] and token_aligned as described above
                seq_len = prompt_len + response_len
                input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.int64, device='cpu')

                token_masks      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                token_dones      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                token_old_logprobs = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')

                # prediction-level
                pred_masks      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                pred_dones      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                pred_old_logprobs = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')

                rewards   = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')

                # Build per-response metadata for reward function
                resp_metadata = metadata_list[prompt_idx].copy() if metadata_list[prompt_idx] else {}
                resp_metadata["response_text"] = getattr(response, "text", "")
                resp_metadata.setdefault("prompt_text", "")

                # Score the response regardless of length (empty responses get negative reward)
                rewards_resp, is_per_token = score_response(self.reward_func, prompt_ids, response_ids, finish_reason, metadata=resp_metadata)
                rewards[prompt_len:] = rewards_resp

                # is_per_token is False, then rewards_resp will only have value for the last element
                group_stats['rewards'].append(rewards_resp.sum().item())
                group_stats['lengths'].append(len(response_ids))

                if response_len > 0:
                    if response.logprobs is None:
                        raise ValueError("response.logprobs is None. Check if SamplingParams(logprobs=1) is set.")

                    # token-aligned
                    token_masks[prompt_len:] = 1
                    response_logprobs = extract_logprobs(response_ids, response.logprobs)
                    token_old_logprobs[prompt_len:] = response_logprobs

                    # pred-aligned
                    # response token j is at token index prompt_len + j in input_ids
                    # and is predicted by logits index prompt_len + j - 1
                    pred_start = prompt_len - 1
                    pred_end   = seq_len - 1
                    pred_masks[pred_start:pred_end] = 1
                    pred_old_logprobs[pred_start:pred_end] = response_logprobs

                    # Terminal handling:
                    #  stop: ended due to EOS or a stop condition → done=1
                    #  length: truncated → done=0, need to bootstrap
                    if finish_reason == "stop":
                        token_dones[seq_len - 1] = 1
                        # pred-aligned terminal is at the logit index that predicts last token
                        pred_dones[seq_len - 2] = 1

                    # if stop_reason is None, it means it ended on eos
                    # see https://docs.vllm.ai/en/stable/api/vllm/outputs/#vllm.outputs.CompletionOutput
                    eos_in_tokens = (response_ids[-1] == self.eos_id)
                    ended_on_eos  = (finish_reason == "stop" and stop_reason is None and eos_in_tokens)

                else:
                    ended_on_eos = False

                group_samples.append({
                    "iter": int(current_iter),
                    "policy_version": int(policy_version),
                    "loaded_version": int(self.loaded_version),

                    # token-aligned
                    "input_ids": input_ids, #[T]
                    "rewards": rewards, #[T]
                    "zscores": rewards.clone(), #[T] replaced in normalize_rewards if n_samples > 1
                    "token_masks": token_masks, #[T] 1 on response/valid tokens
                    "token_dones": token_dones, #[T] 1 on last token if terminal
                    "token_old_logprobs": token_old_logprobs, #[T] 0 on prompt

                    # pred-aligned
                    "pred_masks": pred_masks, #[T]
                    "pred_dones": pred_dones, #[T]
                    "pred_old_logprobs": pred_old_logprobs, #[T]

                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                    "ended_on_eos": ended_on_eos,

                    "response_ids": response_ids, # list[int]
                    "prompt_ids": prompt_ids, # list[int]
                    "response_text": getattr(response, "text", ""),
                    "response_len": response_len,

                    # multimodal data (base64 encoded, or None)
                    "multi_modal_data": prompt_mm_data,
                })
            normalize_rewards(
                samples=group_samples,
                stats=group_stats,
                prompt_len=prompt_len,
                is_per_token=is_per_token,
                eps_reward_norm=self.eps_reward_norm,
                reward_broadcast=self.reward_broadcast)
            rollout_samples.extend(group_samples)

        return rollout_samples

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import ray
    ray.init(local_mode=True)
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

    def default_reward_func(prompt_ids, response_ids, finish_reason, metadata=None):
        is_per_token = False
        r = torch.zeros((len(response_ids),), dtype=torch.float32)

        if len(response_ids) == 0:
            return r, is_per_token

        r[-1] = 1.0 if str(finish_reason) == "stop" else 0.0

        return r, is_per_token

    vllm = VLLMRolloutEngine.remote(model_path='google/gemma-3-1b-it',
                                    trust_remote_code=True,
                                    temperature=1,
                                    max_tokens=1024,
                                    n_samples=5,
                                    top_p=1,
                                    top_k=-1,
                                    seed=50,
                                    ignore_eos=False,
                                    stop=None,
                                    stop_token_ids=None,
                                    prompt_logprobs=None,
                                    force_strict_on_policy=True,
                                    reward_func=default_reward_func,
                                    tensor_parallel_size=1,
                                    eos_id=tokenizer.eos_token_id,
                                    reward_broadcast=True,
                                    eps_reward_norm=1e-8,
                                    gpu_memory_utilization=0.5)

    dummy_data = ["Hello, how are you?",
                  "Summer is the best season!",
                  "I love playing chess.",
                  ]
    samples_ids = []
    for i in dummy_data:
        prompt_ids = tokenizer.apply_chat_template(
                                        conversation= [{"role": "user", "content": i}],
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors=None,
                                        )
        samples_ids.append({"prompt_token_ids": prompt_ids})
    output = vllm.generate.remote(samples_ids, 1, 0)
    print(output)
