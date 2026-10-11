# Post-training a 0.5B LLM for math reasoning: SFT, on-policy DPO, then GRPO with verifiable rewards

A small, reproducible version of the modern LLM post-training pipeline, run on a single free
Kaggle T4. It takes `Qwen2.5-0.5B` (the base model, not Instruct) through three stages on GSM8K
grade-school math and measures what each stage actually buys:

1. **SFT** with LoRA on cleaned reference solutions
2. **DPO** on preference pairs the SFT model generated itself, graded by a program
3. **GRPO** (RL with verifiable rewards): the reward is "is the final number right"

Each stage trains on its own disjoint slice of the GSM8K train split, so the DPO pairs are
sampled on questions the SFT model has not seen, and GRPO trains on questions neither earlier
stage has seen. Every stage is evaluated the same way, and `Qwen2.5-0.5B-Instruct` is evaluated
as an external reference point.

> **Status: in progress.** The project is being built and verified one step at a time; each
> step below is pushed only after it has been checked. No results yet.

## Progress

| # | Step | Status |
|---|---|---|
| 0 | Repo setup, pinned dependencies | done |
| 1 | Answer parsing, verifier, rewards + unit tests | done |
| 2 | Data loading, dedup, decontamination, disjoint stage splits | done |
| 3 | Kaggle environment check | |
| 4 | Smoke run of every stage on a tiny model | |
| 5 | Baselines: base and Instruct | |
| 6 | SFT | |
| 7 | On-policy preference pairs | |
| 8 | DPO | |
| 9 | GRPO (+ ablation: GRPO from SFT) | |
| 10 | Forgetting check, report, model release | |

## Install

```bash
pip install -e ".[eval,dev]"
```

Versions are pinned in `pyproject.toml`; `torch` is left to the platform's build.

Author: Habeeb Olawale Hammed ([GitHub](https://github.com/Haryanfe777) ·
[LinkedIn](https://www.linkedin.com/in/habeeb-olawale-hammed-a5156b153/))
