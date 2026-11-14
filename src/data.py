"""data.py: PPNL single-goal data for Qwen2.5-3B QLoRA.

Input  = nl_description            Target = agent_as_a_point (inherits T5 pipeline)
Sample = "{nl}\nPlan:" + " {target}" + EOS ; loss only on the part after "Plan:".

Usage on Kaggle:  python data.py --check
"""
import argparse
import glob
import json
import os
import statistics

TEN_DATASET = os.environ.get("TEN_DATASET", "ppnl-single-goal-original")
MAX_SEQ_LEN = 128
MODEL_NAME = "Qwen/Qwen2.5-3B"
PROMPT_SUFFIX = "\nPlan:"
UNREACHABLE = "Goal not reachable"

SPLITS = {
    "train": "iid/train_set.json",
    "dev": "iid/dev_set.json",
    "test_seen": "iid/test_seen.json",
    "test_unseen": "iid/test_unseen.json",
    "ood_5x5": "ood/test_unseen_5x5.json",
    "ood_7x7": "ood/test_unseen_7x7.json",
    "ood_moreobsts": "ood/test_unseen_moreobsts.json",
}


def find_root():
    """Locate the single_goal_original folder (Kaggle may add an extra directory level)."""
    env = os.environ.get("PPNL_ROOT")
    cands = [env] if env else []
    cands += glob.glob(f"/kaggle/input/{TEN_DATASET}/**/single_goal_original", recursive=True)
    cands += glob.glob("/kaggle/input/*/**/single_goal_original", recursive=True)
    cands += glob.glob("./**/single_goal_original", recursive=True)
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "iid")):
            return c
    raise FileNotFoundError("single_goal_original not found. Run `!ls /kaggle/input/` and set PPNL_ROOT.")


def load_split(name, root=None):
    root = root or find_root()
    with open(os.path.join(root, SPLITS[name]), encoding="utf-8") as f:
        return json.load(f)


def is_reachable(example):
    return UNREACHABLE not in example["agent_as_a_point"]


def build_prompt(example):
    return example["nl_description"].strip() + PROMPT_SUFFIX


def build_target(example):
    return " " + example["agent_as_a_point"].strip()


def encode_example(example, tok, max_len=MAX_SEQ_LEN):
    """Tokenize prompt and target separately so the prompt/target boundary matches inference."""
    p = tok(build_prompt(example), add_special_tokens=False)["input_ids"]
    t = tok(build_target(example), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = p + t
    labels = [-100] * len(p) + t
    truncated = len(ids) > max_len
    return {"input_ids": ids[:max_len], "labels": labels[:max_len], "truncated": truncated}


def build_dataset(split, tok, max_len=MAX_SEQ_LEN, root=None, limit=None):
    data = load_split(split, root)
    if limit:
        data = data[:limit]
    rows = [encode_example(x, tok, max_len) for x in data]
    return rows


class Collator:
    """Right-pad for training. Labels padded with -100."""

    def __init__(self, tok):
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __call__(self, batch):
        import torch
        m = max(len(b["input_ids"]) for b in batch)
        ids, att, lab = [], [], []
        for b in batch:
            n = m - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * n)
            att.append([1] * len(b["input_ids"]) + [0] * n)
            lab.append(b["labels"] + [-100] * n)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(att),
            "labels": torch.tensor(lab),
        }


def get_tokenizer(name=MODEL_NAME):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def check():
    root = find_root()
    tok = get_tokenizer()
    print("root:", root, "| eos:", tok.eos_token, tok.eos_token_id, "| pad:", tok.pad_token_id)
    for name in SPLITS:
        d = load_split(name, root)
        n_un = sum(not is_reachable(x) for x in d)
        rows = [encode_example(x, tok) for x in d]
        L = [len(r["input_ids"]) for r in rows]
        print(f"{name:14s} n={len(d):5d} unreachable={n_un:3d} "
              f"len p50={pct(L, .5)} p95={pct(L, .95)} max={max(L)} truncated={sum(r['truncated'] for r in rows)}")
    r = build_dataset("train", tok, root=root, limit=3)
    for i, x in enumerate(r):
        prompt_part = tok.decode([t for t, l in zip(x["input_ids"], x["labels"]) if l == -100])
        target_part = tok.decode([l for l in x["labels"] if l != -100])
        print(f"--- sample {i}\nPROMPT(no loss): {prompt_part!r}\nTARGET(loss)  : {target_part!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        check()
