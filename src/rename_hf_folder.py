"""rename_hf_folder.py - rename one run folder inside the HF model repo, in a single commit.

Hugging Face has no folder rename, so every file is copied to the new path and the old folder is deleted.
JSON files (results.json, train_info.json) are rewritten so any mention of the old run name is updated, and the
run row in the root README.md table is renamed too.

Dry run first:
  python rename_hf_folder.py --old rnl_rel-goal_single-goal-original --new rnl_rel-robot_single-goal-original --dry_run
Then run it without --dry_run. Needs a write token (env HF_TOKEN or `huggingface-cli login`).
"""
import argparse
import os

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi, hf_hub_download

REPO = "Nelaia/qwen2.5-3b-ppnl-lora"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(a.repo, repo_type="model") if f.startswith(a.old + "/")]
    if not files:
        raise SystemExit(f"no files under {a.old}/ in {a.repo}")
    if any(f.startswith(a.new + "/") for f in api.list_repo_files(a.repo, repo_type="model")):
        raise SystemExit(f"{a.new}/ already exists in {a.repo}; refusing to overwrite")

    ops = []
    for f in files:
        dst = a.new + f[len(a.old):]
        if f.endswith(".json"):  # small text files: rewrite the run name inside
            text = open(hf_hub_download(a.repo, f, repo_type="model", token=token), encoding="utf-8").read()
            ops.append(CommitOperationAdd(dst, text.replace(a.old, a.new).encode("utf-8")))
        else:
            ops.append(CommitOperationCopy(src_path_in_repo=f, path_in_repo=dst))
    if api.file_exists(a.repo, "README.md", repo_type="model"):
        readme = open(hf_hub_download(a.repo, "README.md", repo_type="model", token=token), encoding="utf-8").read()
        if a.old in readme:
            ops.append(CommitOperationAdd("README.md", readme.replace(a.old, a.new).encode("utf-8")))
    ops += [CommitOperationDelete(f) for f in files]

    for op in ops:
        print(type(op).__name__, getattr(op, "src_path_in_repo", ""), "->" if hasattr(op, "src_path_in_repo") else "",
              op.path_in_repo)
    if a.dry_run:
        print("\n[dry run] nothing changed")
        return
    api.create_commit(a.repo, ops, repo_type="model", commit_message=f"Rename {a.old} -> {a.new}")
    print(f"done: https://huggingface.co/{a.repo}/tree/main/{a.new}")


if __name__ == "__main__":
    main()
