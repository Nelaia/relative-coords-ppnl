"""push_hf.py: push the best adapter of one run into its own subfolder of the shared HF repo.

Dry run (no network, shows what would be uploaded):  python push_hf.py --dry_run
Push                                               :  python push_hf.py
Token: Kaggle Secret HF_TOKEN (write access) or env HF_TOKEN. Never hard-coded.

Only the best adapter + tokenizer + results.json + train_info.json go to <repo>/<RUN_NAME>/.
Other subfolders are never touched; the root README.md table gets/updates one row for this run.
Everything is sent in ONE commit (adapter files and README together).
"""
import argparse
import json
import os
import shutil

import data as D

REPO = "Nelaia/qwen2.5-3b-ppnl-lora"
RUN_NAME = "a_abs_single-goal-original"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./working"
START, END = "<!-- RUNS_TABLE_START -->", "<!-- RUNS_TABLE_END -->"
HEADER = ("| Run | Dataset | Hyperparameters | test_seen (iid) | test_unseen (iid) "
          "| OOD 5x5 | OOD 7x7 | OOD more obstacles |\n|---|---|---|---|---|---|---|---|")

README_NEW = f"""---
base_model: {D.MODEL_NAME}
library_name: peft
tags: [lora, qlora, ppnl, path-planning]
---
# Qwen2.5-3B QLoRA adapters for PPNL

One adapter per subfolder (one subfolder per dataset/run). Load one run:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "{REPO}", subfolder="<run name>")
```

Cells show **Exact match / Success rate** in %. Greedy decoding, "Goal not reachable" examples skipped.

{START}
{HEADER}
{END}
"""


def fmt(m):
    return f"{100 * m['Exact match']:.1f} / {100 * m['Success rate']:.1f}"


def make_row(run, dataset, info, res):
    hp = (f"ep{info['epochs']:g}, lr {info['lr']:.0e}".replace("e-0", "e-") + f", bs {info['bs']}, "
          f"r{info['lora']['r']}/a{info['lora']['alpha']}, len {info['max_seq_len']}")
    cells = [f"`{run}`", dataset, hp] + [fmt(res[s]) for s in
                                         ("test_seen", "test_unseen", "ood_5x5", "ood_7x7", "ood_moreobsts")]
    return "| " + " | ".join(cells) + " |"


def update_readme(old, run, row):
    """Insert or replace this run's row (keyed by run name) inside the marked table; sorted by run."""
    if old is None:
        old = README_NEW
    if START not in old or END not in old:
        old = old.rstrip() + f"\n\n{START}\n{HEADER}\n{END}\n"
    head, rest = old.split(START, 1)
    body, tail = rest.split(END, 1)
    rows = {}
    for line in body.strip().splitlines():
        if line.startswith("|") and not line.startswith("| Run") and not line.startswith("|---"):
            rows[line.split("|")[1].strip().strip("`")] = line
    rows[run] = row
    table = HEADER + "\n" + "\n".join(rows[k] for k in sorted(rows))
    return f"{head}{START}\n{table}\n{END}{tail}"


def get_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        tok = os.environ.get("HF_TOKEN")
        if not tok:
            raise RuntimeError("HF_TOKEN not found: add it in Kaggle Secrets (write access) or set env HF_TOKEN.")
        return tok


def stage(base, run, force):
    best = os.path.join(base, "best_adapter")
    res_path = os.path.join(base, "eval", "results.json")
    info_path = os.path.join(base, "train_info.json")
    for p in (best, res_path, info_path, os.path.join(best, "adapter_model.safetensors")):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing: {p} (run train.py and eval.py for this run first)")
    res_all = json.load(open(res_path))
    missing = [s for s in D.SPLITS if s not in res_all["results"]]
    if (res_all.get("limit") or missing) and not force:
        raise RuntimeError(f"results.json is a smoke run or incomplete (limit={res_all.get('limit')}, "
                           f"missing={missing}). Run the full eval.py, or pass --force.")
    cfg = json.load(open(os.path.join(best, "adapter_config.json")))
    if cfg.get("base_model_name_or_path") != D.MODEL_NAME and not force:
        raise RuntimeError(f"adapter base model is {cfg.get('base_model_name_or_path')}, expected {D.MODEL_NAME}")
    info = json.load(open(info_path))
    st = os.path.join(base, "_upload")
    shutil.rmtree(st, ignore_errors=True)
    shutil.copytree(best, st, ignore=shutil.ignore_patterns("README.md", "checkpoint-*"))  # drop peft's placeholder card
    shutil.copy(res_path, os.path.join(st, "results.json"))
    shutil.copy(info_path, os.path.join(st, "train_info.json"))
    return st, info, res_all["results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default=RUN_NAME)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dataset", default="PPNL single_goal_original")
    ap.add_argument("--private", action="store_true", help="create the repo as private (default: public)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    base = os.path.join(WORK, a.run_name)
    st, info, res = stage(base, a.run_name, a.force)
    row = make_row(a.run_name, a.dataset, info, res)
    files = sorted(os.listdir(st))
    print(f"upload to {a.repo}/{a.run_name}/ : {files}")
    print("size (MB):", round(sum(os.path.getsize(os.path.join(st, f)) for f in files) / 1e6, 1))

    if a.dry_run:
        print("\n[dry run] README table would be:\n" + update_readme(None, a.run_name, row).split(START)[1])
        return

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    token = get_token()
    api = HfApi(token=token)
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    old = None
    if api.file_exists(a.repo, "README.md", repo_type="model"):
        old = open(hf_hub_download(a.repo, "README.md", repo_type="model", token=token), encoding="utf-8").read()
    ops = [CommitOperationAdd(path_in_repo=f"{a.run_name}/{f}", path_or_fileobj=os.path.join(st, f)) for f in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md",
                                  path_or_fileobj=update_readme(old, a.run_name, row).encode("utf-8")))
    api.create_commit(a.repo, ops, repo_type="model", commit_message=f"Add/update run {a.run_name}")
    print(f"done: https://huggingface.co/{a.repo}/tree/main/{a.run_name}")


if __name__ == "__main__":
    main()
