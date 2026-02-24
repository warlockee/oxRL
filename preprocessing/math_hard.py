import argparse
import os
import re
import datasets


def create_prompt(problem, use_system_prompt):
    '''
       This creates general message with or without system prompt.
    '''
    if use_system_prompt:
        system_prompt = 'Solve the following math problem step by step. Put your final answer within \\boxed{}.'
        message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem}
                  ]

    else:
        message = [
                    {"role": "user", "content": problem}
                  ]

    return message


def extract_solution(solution_str):
    '''
       This extracts the content within \\boxed{} from the solution string.
       Handles nested braces within \\boxed{}.
    '''
    # Find \boxed{ and then match balanced braces
    idx = solution_str.rfind("\\boxed{")
    if idx == -1:
        # Fallback: return the full solution string stripped
        return solution_str.strip()

    # Walk forward from the opening brace to find the matching close
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(solution_str) and depth > 0:
        if solution_str[i] == '{':
            depth += 1
        elif solution_str[i] == '}':
            depth -= 1
        i += 1

    result = solution_str[start:i - 1]
    return result.strip()


def make_map_fn(split, params):
    '''
       This function reads data and returns a dictionary.
    '''
    def process_fn(example, idx):
        problem    = example.pop("problem")
        answer_raw = example.pop("solution")
        solution   = extract_solution(answer_raw)
        data       = {
            "prompt": create_prompt(problem, params.use_system_prompt),
            "answer": solution,
            "reward_model": {"ground_truth": solution},
            "split": split,
            "index": idx,
        }
        return data

    return process_fn


def create_file_name(params, split):
    '''
       This function creates file name based on the params.
    '''
    fpart = 'wsp' if params.use_system_prompt == True else 'ns'
    file_name = f"math_hard_{params.run_id}_{fpart}_{split}.parquet"
    return file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="lighteval/MATH")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="123245")
    parser.add_argument("--use_system_prompt", default=False, type=lambda x: str(x).lower() in ("true", "1", "yes"))
    parser.add_argument("--num_proc", default=4)
    args = parser.parse_args()

    ########
    # load dataset from huggingface
    ########
    try:
        dataset = datasets.load_dataset(args.data_source)
    except Exception as e:
        print(f"Failed to load {args.data_source}: {e}")
        print("Falling back to EleutherAI/hendrycks_math (all configs)...")
        args.data_source = "EleutherAI/hendrycks_math"
        configs = ['algebra', 'counting_and_probability', 'geometry',
                   'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
        train_parts, test_parts = [], []
        for cfg in configs:
            ds = datasets.load_dataset(args.data_source, cfg)
            train_parts.append(ds['train'])
            test_parts.append(ds['test'])
        from datasets import concatenate_datasets, DatasetDict
        dataset = DatasetDict({
            'train': concatenate_datasets(train_parts),
            'test': concatenate_datasets(test_parts),
        })

    if "test" in dataset:
        train_dataset = dataset["train"]
        test_dataset  = dataset["test"]
    else:
        # If no test split, create one from train
        split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset  = split_dataset["test"]

    ########
    # map dataset
    ########
    train_dataset = train_dataset.map(function=make_map_fn("train", params=args), with_indices=True, num_proc=args.num_proc)
    test_dataset = test_dataset.map(function=make_map_fn("test", params=args), with_indices=True, num_proc=args.num_proc)

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
