"""Environment check, run once per new Kaggle session before any training.

Prints library versions and the GPU, then runs the real code path on a tiny
scale: load GSM8K through the pipeline's data loader, load the base model in
the inference dtype, and answer one test question greedily. Catches broken
installs, missing GPU and dtype problems in about a minute instead of an hour
into a stage.

    python scripts/check_env.py [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
import importlib
import time

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--max-new-tokens", type=int, default=200)
args = ap.parse_args()

print("== versions")
for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "lm_eval"):
    try:
        print(f"  {name:13s}{importlib.import_module(name).__version__}")
    except ImportError:
        print(f"  {name:13s}NOT INSTALLED")

import torch  # noqa: E402

print("== hardware")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU {props.name}, {total / 2**30:.1f} GB ({free / 2**30:.1f} GB free), "
          f"{torch.cuda.device_count()} visible, bf16 supported: {torch.cuda.is_bf16_supported()}")
else:
    print("  NO GPU: turn on the GPU accelerator in the notebook settings")

from posttrain.answers import extract_lenient, extract_strict, is_correct  # noqa: E402
from posttrain.common import (device_and_dtype, format_prompt, load_config, load_gsm8k,  # noqa: E402
                              load_model, load_tokenizer)

print("== data")
cfg = load_config(args.config)
data = load_gsm8k(cfg)
print("  " + ", ".join(f"{k} {len(v)}" for k, v in data.items()))

print("== base model, one greedy answer")
device, dtype = device_and_dtype()
t0 = time.time()
tok = load_tokenizer(cfg["base_model"])
model = load_model(cfg["base_model"]).eval()
print(f"  loaded {cfg['base_model']} on {device} as {dtype} in {time.time() - t0:.0f}s, "
      f"eos={tok.eos_token!r} pad={tok.pad_token!r}")
row = data["test"][0]
enc = tok(format_prompt(cfg, row["question"]), return_tensors="pt").to(model.device)
t0 = time.time()
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
new = out[0, enc["input_ids"].shape[1]:]
text = tok.decode(new, skip_special_tokens=True)
dt = time.time() - t0
print(f"  {len(new)} tokens in {dt:.1f}s ({len(new) / dt:.0f} tok/s, batch 1)")
print("  ---- completion ----\n" + text + "\n  --------------------")
print(f"  gold {row['gold']:g} | strict {extract_strict(text)} | lenient {extract_lenient(text)} | "
      f"lenient correct: {is_correct(extract_lenient(text), row['gold'])}")
print("ENV OK")
