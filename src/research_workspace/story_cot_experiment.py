import argparse
import json
import os
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numeric answer from GSM8K solution text."""
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None
    # fallback: last number in text
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def parse_model_number(text: str) -> Optional[float]:
    """Extract numeric answer from model output."""
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def parse_aqua_choice(text: str) -> Optional[str]:
    """Extract option letter A-E from model output."""
    m = re.search(r"\b([A-E])\b", text.upper())
    if m:
        return m.group(1)
    return None


def load_gsm8k(n: int = 20, split: str = "test", seed: int = 42) -> List[Dict[str, Any]]:
    data = load_from_disk("datasets/gsm8k")[split]
    idxs = list(range(len(data)))
    random.seed(seed)
    random.shuffle(idxs)
    selected = idxs[:n]
    samples = []
    for idx in selected:
        item = dict(data[idx])
        item["sample_id"] = f"gsm8k-{split}-{idx}"
        samples.append(item)
    return samples


def load_aqua(n: int = 20, split: str = "validation", seed: int = 42) -> List[Dict[str, Any]]:
    data = load_from_disk("datasets/aqua_rat")[split]
    idxs = list(range(len(data)))
    random.seed(seed)
    random.shuffle(idxs)
    selected = idxs[:n]
    samples = []
    for idx in selected:
        item = dict(data[idx])
        item["sample_id"] = f"aqua-{split}-{idx}"
        samples.append(item)
    return samples


def format_prompt(sample: Dict[str, Any], dataset: str, style: str) -> List[Dict[str, str]]:
    """Build chat messages for different prompting styles."""
    system_base = "You are a careful math tutor. Show reasoning then give a final answer."
    question = sample["question"]
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_base}]

    if dataset == "aqua":
        options = sample["options"]
        opts_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        question = f"{question}\nOptions:\n{opts_text}\nRespond with the letter of the correct option."

    if style == "direct":
        user = f"Solve the problem and give only the final answer.\n\nProblem: {question}"
    elif style == "cot":
        user = (
            "Solve the problem. Think step by step and end with 'Final Answer: <answer>'.\n\n"
            f"Problem: {question}"
        )
    elif style == "story":
        user = (
            "Solve the problem by telling a concise story of how the situation unfolds. "
            "Use narrative reasoning (characters, events, quantities) and conclude with "
            "'Final Answer: <answer>'. Keep the story focused on the math."
            f"\n\nProblem: {question}"
        )
    else:
        raise ValueError(f"Unknown style {style}")

    messages.append({"role": "user", "content": user})
    return messages


@dataclass
class ModelConfig:
    model: str
    temperature: float
    max_tokens: int = 256
    top_p: float = 1.0


@dataclass
class SampleResult:
    sample_id: str
    dataset: str
    style: str
    question: str
    gold: Any
    prediction: Any
    raw_output: str
    correct: bool
    usage_prompt_tokens: int
    usage_completion_tokens: int
    extra: Dict[str, Any]


class StoryCoTRunner:
    def __init__(self, client: Optional[OpenAI], model_config: ModelConfig, provider: str = "openai", hf_pipeline=None):
        self.client = client
        self.cfg = model_config
        self.provider = provider
        self.hf_pipeline = hf_pipeline

    def call_model(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> Dict[str, Any]:
        """Call OpenAI chat model and return text plus usage."""
        temp = self.cfg.temperature if temperature is None else temperature
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=temp,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_tokens,
            )
            content = resp.choices[0].message.content or ""
            usage = resp.usage or None
            usage_info = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            return {"text": content, "usage": usage_info}

        # HuggingFace pipeline fallback (chat formatted as plain text prompt).
        prompt = self.messages_to_prompt(messages)
        outputs = self.hf_pipeline(
            prompt,
            max_new_tokens=self.cfg.max_tokens,
            do_sample=temp > 0,
            temperature=temp if temp > 0 else 1.0,
            top_p=self.cfg.top_p,
            return_full_text=False,
        )
        content = outputs[0]["generated_text"]
        usage_info = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        return {"text": content, "usage": usage_info}

    @staticmethod
    def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a plain-text prompt for HF generation."""
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def predict_single(self, sample: Dict[str, Any], dataset: str, style: str) -> SampleResult:
        messages = format_prompt(sample, dataset, style)
        output = self.call_model(messages)
        raw_text = output["text"]
        usage = output["usage"]
        gold, pred = self.parse_answers(sample, raw_text, dataset)
        correct = (gold is not None and pred is not None and self.compare(gold, pred, dataset))
        return SampleResult(
            sample_id=sample.get("sample_id", ""),
            dataset=dataset,
            style=style,
            question=sample["question"],
            gold=gold,
            prediction=pred,
            raw_output=raw_text,
            correct=correct,
            usage_prompt_tokens=usage.get("prompt_tokens") or 0,
            usage_completion_tokens=usage.get("completion_tokens") or 0,
            extra={},
        )

    def predict_self_consistency(
        self, sample: Dict[str, Any], dataset: str, style: str, k: int = 3, temperature: float = 0.7
    ) -> SampleResult:
        messages = format_prompt(sample, dataset, style)
        candidates: List[Any] = []
        outputs: List[str] = []
        usages: List[Dict[str, int]] = []
        for _ in range(k):
            output = self.call_model(messages, temperature=temperature)
            outputs.append(output["text"])
            usages.append(output["usage"])
            _, pred = self.parse_answers(sample, output["text"], dataset)
            candidates.append(pred)
        gold, _ = self.parse_answers(sample, outputs[-1], dataset)  # gold same regardless of output
        vote = self.majority_vote(candidates)
        correct = gold is not None and vote is not None and self.compare(gold, vote, dataset)
        usage_prompt = sum(u.get("prompt_tokens") or 0 for u in usages)
        usage_completion = sum(u.get("completion_tokens") or 0 for u in usages)
        return SampleResult(
            sample_id=sample.get("sample_id", ""),
            dataset=dataset,
            style=f"{style}_self_consistency_k{k}",
            question=sample["question"],
            gold=gold,
            prediction=vote,
            raw_output="\\n\\n".join(outputs),
            correct=correct,
            usage_prompt_tokens=usage_prompt,
            usage_completion_tokens=usage_completion,
            extra={"candidates": candidates},
        )

    def parse_answers(self, sample: Dict[str, Any], output: str, dataset: str) -> tuple[Any, Any]:
        if dataset == "gsm8k":
            gold = parse_gsm8k_answer(sample["answer"])
            pred = parse_model_number(output)
        elif dataset == "aqua":
            gold = sample["correct"].strip().upper()
            pred = parse_aqua_choice(output)
        else:
            raise ValueError(f"Unknown dataset {dataset}")
        return gold, pred

    @staticmethod
    def compare(gold: Any, pred: Any, dataset: str) -> bool:
        if gold is None or pred is None:
            return False
        if dataset == "gsm8k":
            return abs(float(gold) - float(pred)) < 1e-3
        return str(gold).upper() == str(pred).upper()

    @staticmethod
    def majority_vote(candidates: List[Any]) -> Optional[Any]:
        filtered = [c for c in candidates if c is not None]
        if not filtered:
            return None
        counter = Counter(filtered)
        most_common = counter.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            return most_common[0][0]
        # tie-breaker: return first non-null
        return filtered[0]


def run_experiment(
    model: str = "gpt-4.1",
    gsm8k_n: int = 20,
    aqua_n: int = 20,
    seed: int = 42,
    save_dir: str = "results",
    provider: str = "openai",
    hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> Dict[str, Any]:
    """Run experiments across datasets and prompting styles."""
    set_seed(seed)
    ensure_dir(save_dir)
    provider_in_use = provider
    client = None
    hf_pipe = None

    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; falling back to HuggingFace model.")
        provider_in_use = "hf"

    if provider_in_use == "openai":
        client = OpenAI()
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        model_hf = AutoModelForCausalLM.from_pretrained(
            hf_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        hf_pipe = pipeline(
            "text-generation",
            model=model_hf,
            tokenizer=tokenizer,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    runner = StoryCoTRunner(client, ModelConfig(model=model, temperature=0.0), provider=provider_in_use, hf_pipeline=hf_pipe)

    datasets_plan = [
        ("gsm8k", load_gsm8k, gsm8k_n),
        ("aqua", load_aqua, aqua_n),
    ]
    styles = ["direct", "cot", "story"]
    all_results: List[SampleResult] = []

    for dataset_name, loader, n in datasets_plan:
        samples = loader(n=n, seed=seed)
        print(f"Running {dataset_name} with {len(samples)} samples")
        for style in styles:
            for sample in tqdm(samples, desc=f"{dataset_name}-{style}"):
                res = runner.predict_single(sample, dataset=dataset_name, style=style)
                all_results.append(res)
        # self-consistency only for story
        for sample in tqdm(samples, desc=f"{dataset_name}-story-sc"):
            res = runner.predict_self_consistency(sample, dataset=dataset_name, style="story", k=3, temperature=0.7)
            all_results.append(res)

    # Save raw outputs
    raw_path = ensure_dir(save_dir) / "raw_outputs.jsonl"
    with raw_path.open("w") as f:
        for res in all_results:
            line = asdict(res)
            f.write(json.dumps(line) + "\n")

    metrics = summarize_results(all_results)
    metrics_path = ensure_dir(save_dir) / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


def summarize_results(results: List[SampleResult]) -> Dict[str, Any]:
    """Aggregate accuracy and token usage."""
    summary: Dict[str, Dict[str, Any]] = {}
    by_group: Dict[tuple[str, str], List[SampleResult]] = {}
    for res in results:
        key = (res.dataset, res.style)
        by_group.setdefault(key, []).append(res)

    for (dataset, style), items in by_group.items():
        acc = sum(1 for r in items if r.correct) / len(items)
        prompt_tokens = [r.usage_prompt_tokens for r in items]
        completion_tokens = [r.usage_completion_tokens for r in items]
        summary[f"{dataset}::{style}"] = {
            "n": len(items),
            "accuracy": acc,
            "prompt_tokens_mean": statistics.mean(prompt_tokens),
            "completion_tokens_mean": statistics.mean(completion_tokens),
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Story CoT experiments on GSM8K and AQuA.")
    parser.add_argument("--model", default="gpt-4.1", help="Model name for API calls.")
    parser.add_argument("--gsm8k_n", type=int, default=20, help="Number of GSM8K samples.")
    parser.add_argument("--aqua_n", type=int, default=20, help="Number of AQuA samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_dir", default="results", help="Directory to store outputs.")
    parser.add_argument("--provider", default="openai", choices=["openai", "hf"], help="Backend provider.")
    parser.add_argument("--hf_model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model id if provider=hf or fallback.")
    args = parser.parse_args()
    run_experiment(
        model=args.model,
        gsm8k_n=args.gsm8k_n,
        aqua_n=args.aqua_n,
        seed=args.seed,
        save_dir=args.save_dir,
        provider=args.provider,
        hf_model=args.hf_model,
    )
