import argparse
import os
import datasets


def create_prompt(task_description, use_system_prompt):
    '''
       This creates general message with or without system prompt.
    '''
    if use_system_prompt:
        system_prompt = 'Write a Python function to solve the following problem. Provide only the function implementation.'
        message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_description}
                  ]

    else:
        message = [
                    {"role": "user", "content": task_description}
                  ]

    return message


def extract_test_cases(test_list):
    '''
       This joins the list of assert statements into a single string.
    '''
    if isinstance(test_list, list):
        return "\n".join(test_list)
    return str(test_list)


def make_map_fn(split, params):
    '''
       This function reads data and returns a dictionary.
    '''
    def process_fn(example, idx):
        task_description = example.pop("text")
        test_list        = example.pop("test_list")
        test_cases_str   = extract_test_cases(test_list)
        data             = {
            "prompt": create_prompt(task_description, params.use_system_prompt),
            "answer": test_cases_str,
            "reward_model": {"ground_truth": test_cases_str, "test_cases": test_cases_str},
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
    file_name = f"mbpp_{params.run_id}_{fpart}_{split}.parquet"
    return file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="google-research-datasets/mbpp")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="123245")
    parser.add_argument("--use_system_prompt", default=False, type=lambda x: str(x).lower() in ("true", "1", "yes"))
    parser.add_argument("--num_proc", default=4)
    args = parser.parse_args()

    ########
    # load dataset from huggingface
    ########
    dataset = datasets.load_dataset(args.data_source, trust_remote_code=True)

    # MBPP has "train", "validation", "test" splits. We use "train" for training
    # and "test" for evaluation. If splits are missing, handle gracefully.
    if "train" in dataset and "test" in dataset:
        train_dataset = dataset["train"]
        test_dataset  = dataset["test"]
    elif "train" in dataset:
        # If no test split, create one from train
        split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset  = split_dataset["test"]
    else:
        # Some versions have only "full" — combine and split
        full = datasets.concatenate_datasets(
            [dataset[s] for s in dataset.keys()]
        )
        split_dataset = full.train_test_split(test_size=0.1, seed=42)
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
