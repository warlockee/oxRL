import argparse
import os
import datasets


def create_prompt(instruction, use_system_prompt):
    '''
       This creates general message with or without system prompt.
    '''
    if use_system_prompt:
        system_prompt = 'You are a helpful, harmless, and honest assistant. Provide a thorough and well-structured response.'
        message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction}
                  ]

    else:
        message = [
                    {"role": "user", "content": instruction}
                  ]

    return message


def extract_instruction(example):
    '''
       Extract the instruction/query from the example, handling different
       schema versions of UltraFeedback (raw vs processed).
    '''
    # Raw UltraFeedback has "instruction" field
    if "instruction" in example and example["instruction"]:
        return example["instruction"]
    # Some processed versions use "query"
    if "query" in example and example["query"]:
        return example["query"]
    # Some versions use "prompt"
    if "prompt" in example and example["prompt"]:
        return example["prompt"]
    # Fallback
    return ""


def extract_completion(example):
    '''
       Extract a completion/answer from the example if available.
       For UltraFeedback used with format reward, the ground truth is not
       strictly needed — the reward function checks structural quality.
       We store the best completion if available, otherwise empty string.
    '''
    # Raw UltraFeedback has "completions" as a list of dicts with "response" and "overall_score"
    if "completions" in example and example["completions"]:
        completions = example["completions"]
        if isinstance(completions, list) and len(completions) > 0:
            # Pick the highest-scored completion
            best = None
            best_score = -1
            for c in completions:
                score = c.get("overall_score", 0) or 0
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 0
                if score > best_score:
                    best_score = score
                    best = c
            if best and "response" in best:
                return best["response"]
    # Processed versions may have "chosen" field (list of messages or string)
    if "chosen" in example and example["chosen"]:
        chosen = example["chosen"]
        if isinstance(chosen, str):
            return chosen
        if isinstance(chosen, list):
            # Extract assistant response from message list
            for msg in chosen:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
    # Some versions have "response" directly
    if "response" in example and example["response"]:
        return example["response"]
    return ""


def make_map_fn(split, params):
    '''
       This function reads data and returns a dictionary.
    '''
    def process_fn(example, idx):
        instruction = extract_instruction(example)
        completion  = extract_completion(example)
        data        = {
            "prompt": create_prompt(instruction, params.use_system_prompt),
            "answer": completion,
            "reward_model": {"ground_truth": completion},
            "split": split,
            "index": idx,
        }
        return data

    return process_fn


def create_file_name(params, split):
    '''
       This function creates file name based on the params.
    '''
    fpart = 'wsp' if params.use_system_prompt else 'ns'
    file_name = f"ultrafeedback_{params.run_id}_{fpart}_{split}.parquet"
    return file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="openbmb/UltraFeedback")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="123245")
    parser.add_argument("--use_system_prompt", default=False, type=lambda x: str(x).lower() in ("true", "1", "yes"))
    parser.add_argument("--num_proc", default=4)
    args = parser.parse_args()

    ########
    # load dataset from huggingface
    ########
    dataset = datasets.load_dataset(args.data_source, trust_remote_code=True)

    if "train" in dataset and "test" in dataset:
        train_dataset = dataset["train"]
        test_dataset  = dataset["test"]
    elif "train" in dataset:
        # UltraFeedback typically has only a train split, so we create a test split
        split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset  = split_dataset["test"]
    else:
        # Handle case where there is no "train" key — use first available split
        first_key = list(dataset.keys())[0]
        split_dataset = dataset[first_key].train_test_split(test_size=0.05, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset  = split_dataset["test"]

    ########
    # map dataset
    ########
    train_dataset = train_dataset.map(function=make_map_fn("train", params=args), with_indices=True, num_proc=args.num_proc)
    test_dataset = test_dataset.map(function=make_map_fn("test", params=args), with_indices=True, num_proc=args.num_proc)

    ########
    # filter out prompts that are too long (>1500 chars ≈ ~500 tokens, well within 2048 max_seq_len)
    ########
    MAX_PROMPT_CHARS = 1500

    def prompt_not_too_long(example):
        prompt = example.get("prompt", [])
        total = sum(len(m.get("content", "")) for m in prompt if isinstance(m, dict))
        return total <= MAX_PROMPT_CHARS

    before_train = len(train_dataset)
    before_test = len(test_dataset)
    train_dataset = train_dataset.filter(prompt_not_too_long, num_proc=args.num_proc)
    test_dataset = test_dataset.filter(prompt_not_too_long, num_proc=args.num_proc)
    print(f"Filtered long prompts: train {before_train} -> {len(train_dataset)}, test {before_test} -> {len(test_dataset)}")

    ########
    # save dataset
    ########
    os.makedirs(args.local_dir, exist_ok=True)
    train_file_name = os.path.join(args.local_dir, create_file_name(args, "train"))
    test_file_name = os.path.join(args.local_dir, create_file_name(args, "test"))
    train_dataset.to_parquet(train_file_name)
    test_dataset.to_parquet(test_file_name)

    print("\n")
    print(f"Train file: {train_file_name} with {len(train_dataset)} examples.")
    print(f"Test file: {test_file_name} with {len(test_dataset)} examples.")
    print("Done.")
