import pytest

from posttrain.common import decontaminate, dedup, load_config, split_by_stage


def test_dedup_and_decontaminate():
    train = [{"question": "How many apples does Tom have if he buys three more today"},
             {"question": "how many apples  does Tom have if he buys three more today"},
             {"question": "A train leaves the station at noon"}]
    test = [{"question": "How many apples does Tom have if he buys three more today and sells two"}]
    assert len(dedup(train)) == 2
    kept, removed = decontaminate(dedup(train), test, n=8)
    assert len(removed) == 1 and kept[0]["question"].startswith("A train")


def test_split_by_stage_is_disjoint_complete_and_deterministic():
    rows = [{"id": f"train-{i}"} for i in range(100)]
    split = split_by_stage(rows, {"pairs": 20, "grpo": 15}, seed=42)
    ids = {s: {r["id"] for r in split[s]} for s in ("sft", "pairs", "grpo")}
    assert [len(ids[s]) for s in ("sft", "pairs", "grpo")] == [65, 20, 15]
    assert not ids["sft"] & ids["pairs"] and not ids["sft"] & ids["grpo"] and not ids["pairs"] & ids["grpo"]
    assert ids["sft"] | ids["pairs"] | ids["grpo"] == {r["id"] for r in rows}
    assert split == split_by_stage(rows, {"pairs": 20, "grpo": 15}, seed=42)
    assert split != split_by_stage(rows, {"pairs": 20, "grpo": 15}, seed=7)
    assert [r["id"] for r in rows][:3] == ["train-0", "train-1", "train-2"]  # input not shuffled in place


def test_split_by_stage_rejects_sizes_that_leave_no_sft_data():
    with pytest.raises(ValueError):
        split_by_stage([{"id": str(i)} for i in range(10)], {"pairs": 5, "grpo": 5}, seed=0)


def test_config_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("sft: {lr: 0.1}\nseed: 1\n")
    cfg = load_config(str(p), ["sft.lr=2e-4", "eval.num_examples=10"])
    assert cfg["sft"]["lr"] == 2e-4 and cfg["eval"]["num_examples"] == 10
