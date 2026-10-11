"""Config loading, data preparation, model I/O and small helpers shared by every stage."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import yaml

from .answers import clean_rationale, gold_answer

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """YAML config with optional dotted overrides: --set sft.lr=1e-4 eval.num_examples=200."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        value = yaml.safe_load(raw)
        if isinstance(value, str):  # PyYAML reads "2e-4" as a string (YAML 1.1); fix that
            try:
                value = float(value)
            except ValueError:
                pass
        node[leaf] = value
    return cfg


def cli(description: str) -> tuple[dict[str, Any], argparse.Namespace]:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--model", default=None, help="Model path or hub id (eval only)")
    ap.add_argument("--name", default=None, help="Run name used for output folders (eval only)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.get("seed", 42))
    return cfg, args


def out_path(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["output_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def stage_model_path(cfg: dict, stage: str) -> str:
    """Where each stage's merged model lives. 'base' is the hub id."""
    if stage == "base":
        return cfg["base_model"]
    return str(Path(cfg["output_dir"]) / "models" / stage)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def write_json(obj: Any, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(rows: list[dict], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def format_prompt(cfg: dict, question: str) -> str:
    return cfg["prompt_template"].format(question=question.strip())


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def decontaminate(train_rows: list[dict], test_rows: list[dict], n: int) -> tuple[list[dict], list[dict]]:
    """Drop train questions that share any n-gram with a test question.

    GSM8K is known to be clean, but checking costs seconds, and a result built on
    leaked test items is worthless. Returns (kept, removed).
    """
    test_grams: set[tuple[str, ...]] = set()
    for r in test_rows:
        test_grams |= _ngrams(r["question"], n)
    kept, removed = [], []
    for r in train_rows:
        (removed if _ngrams(r["question"], n) & test_grams else kept).append(r)
    return kept, removed


def dedup(rows: list[dict]) -> list[dict]:
    """Exact-duplicate removal on whitespace/case-normalised question text."""
    seen, out = set(), []
    for r in rows:
        key = " ".join(r["question"].lower().split())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


STAGE_SPLITS = ("sft", "pairs", "grpo")


def split_by_stage(rows: list[dict], sizes: dict[str, int], seed: int) -> dict[str, list[dict]]:
    """Disjoint train slices: fixed-size 'pairs' and 'grpo', and 'sft' gets the rest.

    Each stage trains on questions no earlier stage has seen: DPO pairs are
    sampled on questions the SFT model was not trained on (so its success rate
    there is honest), and GRPO's questions are new to both SFT and DPO.
    """
    n_pairs, n_grpo = sizes["pairs"], sizes["grpo"]
    if n_pairs + n_grpo >= len(rows):
        raise ValueError(f"stage_sizes {sizes} leave no SFT data out of {len(rows)} train questions")
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    return {"pairs": shuffled[:n_pairs], "grpo": shuffled[n_pairs:n_pairs + n_grpo],
            "sft": shuffled[n_pairs + n_grpo:]}


def load_gsm8k(cfg: dict) -> dict[str, list[dict]]:
    """GSM8K as plain dicts with cleaned rationale and numeric gold answer.

    Returns 'test', the cleaned 'train', and its disjoint per-stage slices
    'sft', 'pairs' and 'grpo'. Also writes a data report (duplicates,
    contamination, split sizes, length stats) so the data step is inspectable.
    """
    from datasets import load_dataset

    dcfg = cfg["data"]
    ds = load_dataset(dcfg["dataset"], dcfg.get("subset", "main"))
    split = {}
    for name in ("train", "test"):
        split[name] = [
            {"id": f"{name}-{i}", "question": r["question"], "rationale": clean_rationale(r["answer"]),
             "gold": gold_answer(r["answer"])}
            for i, r in enumerate(ds[name])
        ]
    raw_train = len(split["train"])
    split["train"] = dedup(split["train"])
    n_dupes = raw_train - len(split["train"])
    split["train"], removed = decontaminate(split["train"], split["test"], dcfg.get("decontam_ngram", 13))
    n_clean = len(split["train"])
    if dcfg.get("max_train"):
        split["train"] = split["train"][: dcfg["max_train"]]
    split.update(split_by_stage(split["train"], dcfg["stage_sizes"], cfg["seed"]))

    # Cheap to recompute, so rewritten every call: it always matches the current config.
    lens = sorted(len(r["rationale"].split()) for r in split["train"])
    write_json({
        "train_raw": raw_train, "exact_duplicates_removed": n_dupes,
        "decontam_ngram": dcfg.get("decontam_ngram", 13), "contaminated_removed": len(removed),
        "contaminated_examples": [r["question"][:160] for r in removed[:10]],
        "train_clean": n_clean, "train_used": len(split["train"]), "test": len(split["test"]),
        **{f"{s}_questions": len(split[s]) for s in STAGE_SPLITS},
        "rationale_words_median": lens[len(lens) // 2], "rationale_words_p95": lens[int(len(lens) * .95)],
    }, out_path(cfg, "data", "data_report.json"))
    return split


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def device_and_dtype(for_training: bool = False):
    import torch
    if torch.cuda.is_available():
        # bf16 where supported (A100/L4/H100). T4/P100 fall back: fp32 master
        # weights + fp16 autocast for training, plain fp16 for inference.
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", (torch.float32 if for_training else torch.float16)
    return "cpu", torch.float32


def mixed_precision_flags() -> dict[str, bool]:
    import torch
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": True}


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path: str, for_training: bool = False):
    from transformers import AutoModelForCausalLM
    device, dtype = device_and_dtype(for_training)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    return model.to(device)


def lora_config(scfg: dict):
    from peft import LoraConfig
    return LoraConfig(
        r=scfg["lora_r"], lora_alpha=scfg["lora_alpha"], lora_dropout=scfg["lora_dropout"],
        target_modules=scfg.get("target_modules", "all-linear"), task_type="CAUSAL_LM",
    )


def merge_and_save(trainer, tokenizer, dest: str) -> None:
    """Merge the LoRA adapter into the weights so the next stage starts from a plain model."""
    model = trainer.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    Path(dest).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tokenizer.save_pretrained(dest)


def save_log_history(trainer, cfg: dict, stage: str) -> None:
    write_json(trainer.state.log_history, out_path(cfg, "logs", f"{stage}_log_history.json"))
