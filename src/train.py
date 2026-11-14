"""train.py: QLoRA 4-bit fine-tune of Qwen2.5-3B on PPNL single-goal (any of the 5 input conditions).

The condition is selected by the data folder (PPNL_ROOT) and the per-condition settings below:
  A            --max_seq_len 128  --bs 32
  R-num/G-num  --max_seq_len 256  --bs 32
  R-nl/G-nl    --max_seq_len 365  --bs 16 --grad_accum 2      (effective batch 32 in every case)

Smoke test (20 steps):  python train.py --max_steps 20
Full run               :  python train.py --epochs 3
Resume                 :  just re-run the same command (auto-resumes from the last checkpoint in
                          /kaggle/working/<RUN_NAME>/ckpt). To resume from a previous notebook
                          version's output attached as input: --resume_from <path/to/checkpoint-N>
"""
import argparse
import inspect
import json
import os
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint

import data as D

RUN_NAME = "a_abs_single-goal-original"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./working"


# ---------- generation + exact match (also imported by eval.py) ----------
@torch.no_grad()
def generate_plans(model, tok, prompts, bs=64, max_new_tokens=48):
    """Greedy decoding, left padding. Returns one plan string per prompt."""
    was_training = model.training
    model.eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    outs = []
    for i in range(0, len(prompts), bs):
        enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts[i:i + bs]]
        m = max(len(e) for e in enc)
        ids = torch.tensor([[pad] * (m - len(e)) + e for e in enc], device=model.device)
        att = torch.tensor([[0] * (m - len(e)) + [1] * len(e) for e in enc], device=model.device)
        g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens,
                           do_sample=False, num_beams=1, eos_token_id=tok.eos_token_id,
                           pad_token_id=pad, use_cache=True)
        for row in g[:, m:].tolist():
            if tok.eos_token_id in row:
                row = row[:row.index(tok.eos_token_id)]
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
    if was_training:
        model.train()
    return outs


def exact_match_score(model, tok, examples):
    preds = generate_plans(model, tok, [D.build_prompt(x) for x in examples])
    hit = sum("".join(p.split()) == "".join(x["agent_as_a_point"].split())
              for p, x in zip(preds, examples))
    return hit / max(1, len(examples))


# ---------- trainer pieces ----------
class ListDS(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = [{"input_ids": r["input_ids"], "labels": r["labels"]} for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class EMTrainer(Trainer):
    """Adds eval_exact_match (greedy generation on a dev subset) so best-checkpoint selection
    uses exact match like the T5 pipeline."""

    def __init__(self, *a, eval_examples=None, tok=None, **k):
        super().__init__(*a, **k)
        self.eval_examples, self.tok = eval_examples, tok

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        em = exact_match_score(self.model, self.tok, self.eval_examples)
        metrics[f"{metric_key_prefix}_exact_match"] = em
        self.log({f"{metric_key_prefix}_exact_match": em})
        return metrics


class TimeGuard(TrainerCallback):
    """Save and stop cleanly before the Kaggle session limit."""

    def __init__(self, max_hours):
        self.t0, self.limit = time.time(), max_hours * 3600

    def on_step_end(self, args, state, control, **kw):
        if time.time() - self.t0 > self.limit:
            control.should_save = True
            control.should_training_stop = True
            print(f"[TimeGuard] {self.limit / 3600:.1f}h reached: saving and stopping. Re-run to resume.")
        return control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default=RUN_NAME)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max_steps", type=int, default=-1, help=">0 = smoke test (separate output dir)")
    ap.add_argument("--bs", type=int, default=32, help="per-device batch size")
    ap.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--max_seq_len", type=int, default=D.MAX_SEQ_LEN, help="token budget per example (no truncation expected)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--eval_n", type=int, default=500, help="dev examples used for eval during training")
    ap.add_argument("--max_hours", type=float, default=8.0)
    ap.add_argument("--resume_from", default=None)
    a = ap.parse_args()
    set_seed(1)

    smoke = a.max_steps > 0
    base = os.path.join(WORK, a.run_name + ("_smoke" if smoke else ""))
    ckpt_dir, best_dir = os.path.join(base, "ckpt"), os.path.join(base, "best_adapter")
    os.makedirs(ckpt_dir, exist_ok=True)
    if smoke:
        a.save_steps, a.eval_n = min(a.save_steps, a.max_steps), 64

    tok = D.get_tokenizer()
    root = D.find_root()
    train_rows = D.build_dataset("train", tok, max_len=a.max_seq_len, root=root)
    dev = [x for x in D.load_split("dev", root) if D.is_reachable(x)][:a.eval_n]
    dev_rows = [D.encode_example(x, tok, a.max_seq_len) for x in dev]
    print(f"train={len(train_rows)} dev_eval={len(dev_rows)} truncated_train={sum(r['truncated'] for r in train_rows)}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(D.MODEL_NAME, quantization_config=bnb,
                                                 device_map={"": 0}, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():  # fp16 AMP needs trainable params in fp32
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    sig = inspect.signature(TrainingArguments).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    args = TrainingArguments(
        output_dir=ckpt_dir, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, optim="paged_adamw_8bit",
        fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25, save_strategy="steps", save_steps=a.save_steps, eval_steps=a.save_steps,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="exact_match",
        greater_is_better=True, report_to="none", remove_unused_columns=False, seed=1,
        dataloader_num_workers=2, **{eval_key: "steps"})

    trainer = EMTrainer(model=model, args=args, train_dataset=ListDS(train_rows),
                        eval_dataset=ListDS(dev_rows), data_collator=D.Collator(tok),
                        callbacks=[TimeGuard(a.max_hours)], eval_examples=dev, tok=tok)

    resume = a.resume_from or get_last_checkpoint(ckpt_dir)
    print("resume from:", resume)
    trainer.train(resume_from_checkpoint=resume)

    trainer.model.save_pretrained(best_dir)
    tok.save_pretrained(best_dir)
    info = {"run_name": a.run_name, "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_dev_exact_match_subset": trainer.state.best_metric, "epochs": a.epochs,
            "lr": a.lr, "bs": a.bs, "grad_accum": a.grad_accum, "effective_bs": a.bs * a.grad_accum,
            "max_seq_len": a.max_seq_len, "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
            "dev_subset_n": len(dev)}
    with open(os.path.join(base, "train_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("saved best adapter to", best_dir, "| best dev EM (subset):", trainer.state.best_metric)


if __name__ == "__main__":
    main()
