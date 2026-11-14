"""generate_predictions.py - run every adapter on every split and archive the per-sample predictions.

For each of the 5 conditions it (1) builds the converted dataset with make_condition.py if needed,
(2) loads the matching LoRA adapter from the Hugging Face repo, (3) greedy-decodes all splits, and
(4) writes outputs/pred_<cond>_<split>.csv. All 30 CSV files are zipped into outputs/all_predictions.zip.

CSV columns: id (position among reachable examples), orig_idx (index in the original PPNL json file),
input (the prompt actually given to the model), answer (reference plan), pred (model output).

Example:
  python generate_predictions.py --ppnl_root /path/to/single_goal_original
"""
import argparse
import glob
import os
import subprocess
import sys
import zipfile

import pandas as pd
import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
from train import generate_plans  # noqa: E402

# condition -> (adapter subfolder in the HF repo, frame/format marker expected in the converted prompts)
RUNS = {
    "a": ("a_abs_single-goal-original", None),
    "rnum": ("rnum_rel-robot_single-goal-original", "relative to the robot"),
    "rnl": ("rnl_rel-robot_single-goal-original", "The goal is"),
    "gnum": ("gnum_rel-goal_single-goal-original", "relative to the goal"),
    "gnl": ("gnl_rel-goal_single-goal-original", "The robot is"),
}
SPLIT_NAMES = ["dev", "test_seen", "test_unseen", "ood_5x5", "ood_7x7", "ood_moreobsts"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ppnl_root", required=True,
                    help="Path to the original PPNL single_goal_original folder (contains iid/ and ood/).")
    ap.add_argument("--repo_id", default="Nelaia/qwen2.5-3b-ppnl-lora",
                    help="Hugging Face repo hosting the LoRA adapters (one subfolder per run).")
    ap.add_argument("--output_dir", default="./outputs")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""),
                    help="Only needed while the repo is private (defaults to env HF_TOKEN).")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.hf_token:
        login(token=args.hf_token)
    os.makedirs(args.output_dir, exist_ok=True)
    make_script = os.path.join(HERE, "make_condition.py")

    tok = D.get_tokenizer()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)

    for cond, (subfolder, marker) in RUNS.items():
        data_root = args.ppnl_root if cond == "a" else os.path.join(args.output_dir, f"data_{cond}")
        if cond != "a" and not os.path.exists(data_root):
            print(f"\nGenerating dataset for {cond} -> {data_root}")
            subprocess.run([sys.executable, make_script, "--cond", cond, "--src", args.ppnl_root,
                            "--dst", data_root], check=True)

        print(f"\n==================== CONDITION: {cond} ({subfolder}) ====================")
        base = AutoModelForCausalLM.from_pretrained(D.MODEL_NAME, quantization_config=bnb, device_map={"": 0},
                                                    torch_dtype=torch.float16, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, args.repo_id, subfolder=subfolder).eval()

        for name in SPLIT_NAMES:
            save_path = os.path.join(args.output_dir, f"pred_{cond}_{name}.csv")
            pairs = [(i, x) for i, x in enumerate(D.load_split(name, data_root)) if D.is_reachable(x)]
            prompts = [D.build_prompt(x) for _, x in pairs]
            refs = [x["agent_as_a_point"] for _, x in pairs]
            if marker:  # guard against pairing an adapter with the wrong input frame/format
                assert marker in prompts[0], f"{cond}: prompt does not contain {marker!r}: {prompts[0]!r}"

            preds = generate_plans(model, tok, prompts, bs=args.batch_size, max_new_tokens=args.max_new_tokens)
            df = pd.DataFrame({"id": range(len(pairs)), "orig_idx": [i for i, _ in pairs],
                               "input": prompts, "answer": refs, "pred": preds})
            df.to_csv(save_path, index=False)

            em = sum("".join(p.split()) == "".join(r.split()) for p, r in zip(preds, refs)) / len(refs) * 100
            print(f"  -> {name:<14} ({len(df):>4} samples): EM = {em:6.2f}% | {save_path}")

        del model, base
        torch.cuda.empty_cache()

    csv_files = sorted(glob.glob(os.path.join(args.output_dir, "pred_*.csv")))
    print(f"\nFound {len(csv_files)} CSV files (expected 30)")
    zip_path = os.path.join(args.output_dir, "all_predictions.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in csv_files:
            zf.write(f, arcname=os.path.basename(f))
    print(f"DONE: {zip_path}")


if __name__ == "__main__":
    main()
