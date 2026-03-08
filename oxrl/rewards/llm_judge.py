"""LLM-as-Judge reward function.

Uses an OpenAI-compatible chat completions API to score responses.
Callable and picklable for use with Ray actors.
"""
import os
import re
import time
from typing import Any, Dict, List, Optional

import torch

from oxrl.rewards.backend import RewardBackend


class LLMJudgeReward(RewardBackend):
    """Scores responses using an external LLM judge via OpenAI-compatible API.

    The judge prompt is formatted with {prompt} and {response} placeholders.
    The first number in the judge's response is extracted as the score.
    """

    def __init__(self, config):
        self.api_base = config.api_base
        self.model = config.model
        self.prompt_template = config.prompt_template
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay
        self.timeout = config.timeout
        self.fallback_score = config.fallback_score
        self.normalize_to_01 = config.normalize_to_01
        self.api_key = config.api_key or os.environ.get(config.api_key_env, "")
        # httpx client is lazily initialized and excluded from pickling
        self._client = None

    def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def cleanup(self):
        """Close the httpx client if open."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __call__(self, prompt_ids, response_ids, finish_reason, metadata=None):
        resp_len = len(response_ids)
        rewards = torch.zeros((resp_len,), dtype=torch.float32)
        if resp_len == 0:
            return rewards, False

        prompt_text = (metadata or {}).get("prompt_text", "")
        response_text = (metadata or {}).get("response_text", "")

        score = self._query_judge(prompt_text, response_text)
        rewards[-1] = score
        return rewards, False

    def _query_judge(self, prompt_text: str, response_text: str) -> float:
        """Send prompt+response to judge LLM, extract numeric score."""
        judge_prompt = self.prompt_template.format(
            prompt=prompt_text, response=response_text
        )

        for attempt in range(self.max_retries):
            try:
                client = self._get_client()
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                resp = client.post(
                    f"{self.api_base}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": judge_prompt}],
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return self._extract_score(text)

            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return self.fallback_score

    def _extract_score(self, text: str) -> float:
        """Extract the first number from judge response text."""
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match is None:
            return self.fallback_score
        score = float(match.group(1))
        score = min(score, 10.0)
        if self.normalize_to_01:
            score = score / 10.0
        return score
