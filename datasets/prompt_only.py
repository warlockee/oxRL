import os
from torch.utils.data import Dataset
from datasets import load_dataset

class PromptOnlyDataset(Dataset):
    '''
        Returns tokenized prompt ids as a list[int] which can be a variable length.
    '''
    def __init__(self,
                prompt_key: str,
                tokenizer,
                max_seq_len: int,
                data_path: str,
                return_text: bool=False,
                answer_key: str="",
                model_name: str="",
                ):
        assert prompt_key != "", "prompt_key cannot be empty"
        assert max_seq_len > 0, "max_seq_len must be > 0"
        assert tokenizer is not None, "tokenizer cannot be None"
        assert isinstance(data_path, str), "data_path must be a string"
        assert os.path.exists(os.path.expanduser(data_path)), f"{data_path} does not exist"
        assert tokenizer.pad_token_id is not None, "tokenizer must have a pad token"
        assert tokenizer.eos_token_id is not None, "tokenizer must have an eos token"

        self.prompt_key  = prompt_key
        self.max_seq_len = int(max_seq_len)
        self.tokenizer   = tokenizer
        self.data_path   = data_path
        self.return_text = return_text
        self.answer_key  = answer_key
        
        self.processor = None
        if model_name:
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                pass
        
        self._load_data()

    def _load_data(self):
        '''
           Loads the data from a parquet file.
        '''
        try:
            # Loads lazily (disk/cache). split='train' is a HF datasets arg:
            #  without it we get a DatasetDict; with it we get a Dataset.
            self.data = load_dataset("parquet", data_files=self.data_path, split="train")

        except Exception as e:
            raise Exception(f"Failed to load data from {self.data_path}: {str(e)}")

        self.len_data = len(self.data)

    def __getitem__(self, idx):
        '''
           data is a dict with the following format:
           {
            {
                "prompt": [{"role": "system", "content": "..."},
                           {"role": "user", "content": "..."}
                           ],
            },...
           }
           Note system prompt is optional.
        '''
        sample = self.data[idx]
        if self.prompt_key not in sample:
            raise KeyError(f"Missing key '{self.prompt_key}' in sample {sample}: keys={list(sample.keys())}")

        message = sample[self.prompt_key]
        # message cannot be empty
        if not message or (isinstance(message, list) and len(message) == 0):
            raise ValueError(f"Sample {idx}:{sample}: Prompt cannot be empty")

        # Tokenize prompt for vLLM rollout
        # Some tokenizers (e.g. SmolLM2) return near-empty results with
        # tokenize=True but work correctly with tokenize=False + encode.
        # We try tokenize=True first; if that yields <= 2 tokens we fall back.
        
        # Use processor for multimodal if available
        is_multimodal = ("image_base64" in sample or "audio_base64" in sample)
        if self.processor and is_multimodal:
            import base64
            import io
            from PIL import Image
            import soundfile as sf
            
            # Deep copy or construct new message to avoid modifying cache
            processed_message = []
            for m in message:
                new_content = []
                if isinstance(m.get("content"), list):
                    for part in m["content"]:
                        if part.get("type") == "image" and "image_base64" in sample:
                            img_data = base64.b64decode(sample["image_base64"])
                            image = Image.open(io.BytesIO(img_data)).convert("RGB")
                            new_content.append({"type": "image", "image": image})
                        elif part.get("type") == "audio" and "audio_base64" in sample:
                            audio_data_bytes = base64.b64decode(sample["audio_base64"])
                            audio_data, sample_rate = sf.read(io.BytesIO(audio_data_bytes))
                            new_content.append({"type": "audio", "audio": (audio_data, sample_rate)})
                        else:
                            new_content.append(part)
                else:
                    new_content = m.get("content")
                processed_message.append({"role": m["role"], "content": new_content})

            prompt_ids = self.processor.apply_chat_template(
                processed_message,
                add_generation_prompt=True,
                tokenize=True,
            )
        else:
            prompt_ids = self.tokenizer.apply_chat_template(
                                            conversation=message,
                                            add_generation_prompt=True,
                                            tokenize=True,
                                            return_tensors=None,
                                            )
        
        if not is_multimodal and (not isinstance(prompt_ids, list) or len(prompt_ids) <= 2):
            # Fallback: render as text then encode
            prompt_text = self.tokenizer.apply_chat_template(
                                            conversation=message,
                                            add_generation_prompt=True,
                                            tokenize=False,
                                            return_tensors=None,
                                            )
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if not isinstance(prompt_ids, list) or len(prompt_ids) == 0:
            raise ValueError(f"Sample {idx}:{sample}: tokenization produced empty prompt_ids")

        # Validate prompt length -- truncate if too long instead of failing
        if len(prompt_ids) >= self.max_seq_len:
            prompt_ids = prompt_ids[:self.max_seq_len - 1]

        result = {
            "prompt_token_ids": prompt_ids,
            "prompt_structured": message # Original list of dicts
        }
        
        if "image_base64" in sample:
            result["multi_modal_data"] = {"image": sample["image_base64"]}
        elif "audio_base64" in sample:
            result["multi_modal_data"] = {"audio": sample["audio_base64"]}

        # Pass answer through as metadata so reward functions can access it.
        # answer_key controls which parquet column to read; it always arrives
        # at the reward function as metadata["answer"].
        if self.answer_key and self.answer_key in sample:
            result["metadata"] = {"answer": sample[self.answer_key]}

        if self.return_text:
            prompt_text = self.tokenizer.apply_chat_template(
                                            conversation=message,
                                            add_generation_prompt=True,
                                            tokenize=False,
                                            return_tensors=None,
                                            )
            result["text"] = prompt_text

        return result

    def __len__(self):
        return self.len_data

    def collate_fn(self, batch):
        '''
            Since pytorch's default collate tries to stack sequences, we need to override it.
            Otherwise, we will get all sequences must have equal size error.
            This function keeps variable-length items as python objects
        '''
        if self.return_text == False:
            # batch: List[List[int]] -> No, wait, __getitem__ returns a dict!
            return batch

        prompt_texts = []
        prompt_token_ids   = []
        # If return_text is True we lose metadata and multi_modal_data, but let's assume return_text is False during training.
        for x in batch:
            prompt_token_ids.append(x["prompt_token_ids"])
            prompt_texts.append(x["text"])

        return prompt_texts, prompt_token_ids

if __name__ == "__main__":
    '''
        This is a simple test to make sure the dataset works.
    '''
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import pandas as pd

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    # add pad token if it doesn't exist, not useful here but good practice
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    random_prompts = [
        {'prompt': [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello, how are you?"}]},
        {'prompt': [{"role": "user", "content": "What is the meaning of life?"}]},
        {'prompt': [{"role": "user", "content": "What is the meaning of the universe?"}]},
        {'prompt': [{"role": "user", "content": "This is is a just rather long prompt that is going to be tokenized. This is a test to make sure the dataset works."}]},

    ]
    df = pd.DataFrame(random_prompts)
    df.to_parquet("./promptonly.parquet", index=False)

    dataset = PromptOnlyDataset(
                prompt_key="prompt",
                tokenizer=tokenizer,
                max_seq_len=1024,
                data_path="./promptonly.parquet",
                return_text=False,
                )
    dataloader = DataLoader(dataset,
                            batch_size=3,
                            collate_fn=dataset.collate_fn,
                            shuffle=False)
    for d in dataloader:
        print(d)
        print("\n")