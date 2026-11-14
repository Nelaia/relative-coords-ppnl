"""collect_results.py - gather every run's results from the shared HF repo and build comparison tables.

For each subfolder <RUN_NAME>/ of the repo it reads results.json (eval.py output) and train_info.json
(train.py output). Outputs go to ./comparison/ :
  all_results.csv   one row per (run, split): Total + 6 metrics + 95% Wilson CI for the proportion metrics
  hyperparams.csv   one row per run: epochs, lr, batch, grad_accum, effective batch, max_seq_len, LoRA, best ckpt
  comparison.md     hyperparameters, one table per metric (runs x splits, best value in bold),
                    and delta vs condition A in percentage points for Exact match and Success rate
  comparison.tex    (with --latex) the same metric tables as booktabs tabulars

Run (a token is only needed while the repo is private: Kaggle Secret HF_TOKEN, env HF_TOKEN, or `huggingface-cli login`):
  pip install -q huggingface_hub
  python collect_results.py
  python collect_results.py --repo Nelaia/qwen2.5-3b-ppnl-lora --latex
Offline (folder that mirrors the repo layout <RUN>/results.json):
  python collect_results.py --local /path/to/folder
"""
import argparse
import csv
import json
import math
import os

REPO = "Nelaia/qwen2.5-3b-ppnl-lora"
METRICS = ["Exact match", "Success rate", "Distance", "Feasible", "Optimal", "Optimal (steps)"]
PROP = {"Exact match", "Success rate", "Feasible", "Optimal", "Optimal (steps)"}  # proportions in [0, 1]
SPLIT_ORDER = ["train", "dev", "test_seen", "test_unseen", "ood_5x5", "ood_7x7", "ood_moreobsts"]
SPLIT_NAME = {"train": "train (sanity)", "dev": "dev", "test_seen": "test_seen (iid)",
              "test_unseen": "test_unseen (iid)", "ood_5x5": "OOD 5x5", "ood_7x7": "OOD 7x7",
              "ood_moreobsts": "OOD more obstacles", "iid avg": "iid avg", "ood avg": "OOD avg"}
IID, OOD = ["test_seen", "test_unseen"], ["ood_5x5", "ood_7x7", "ood_moreobsts"]
CONDITIONS = [("a_", "A"), ("rnum", "R-num"), ("rnl", "R-nl"), ("gnum", "G-num"), ("gnl", "G-nl")]


def condition(run):
    r = run.lower()
    for prefix, label in CONDITIONS:
        if r.startswith(prefix):
            return label
    return run


def cond_rank(run):
    labels = [l for _, l in CONDITIONS]
    c = condition(run)
    return (labels.index(c) if c in labels else len(labels), run)


# ---------- loading ----------
def get_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None  # fall back to a cached `huggingface-cli login`


def fetch_hf(repo):
    from huggingface_hub import HfApi, hf_hub_download
    token = get_token()
    files = set(HfApi(token=token).list_repo_files(repo, repo_type="model"))
    runs = sorted({f.split("/")[0] for f in files if f.count("/") == 1 and f.endswith("/results.json")})
    out = {}
    for r in runs:
        res = json.load(open(hf_hub_download(repo, f"{r}/results.json", token=token), encoding="utf-8"))
        ti = f"{r}/train_info.json"
        info = json.load(open(hf_hub_download(repo, ti, token=token), encoding="utf-8")) if ti in files else {}
        out[r] = (res, info)
    return out


def fetch_local(root):
    out = {}
    for r in sorted(os.listdir(root)):
        rp, ip = (os.path.join(root, r, n) for n in ("results.json", "train_info.json"))
        if os.path.exists(rp):
            out[r] = (json.load(open(rp, encoding="utf-8")),
                      json.load(open(ip, encoding="utf-8")) if os.path.exists(ip) else {})
    return out


# ---------- numbers ----------
def wilson(p, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    d, c = 1 + z * z / n, p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def val(res, split, metric):
    v = res["results"].get(split, {}).get(metric)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def col_value(res, metric, col):
    if col in ("iid avg", "ood avg"):
        vs = [val(res, s, metric) for s in (IID if col == "iid avg" else OOD)]
        return None if any(v is None for v in vs) else sum(vs) / len(vs)  # unweighted mean over splits
    return val(res, col, metric)


def fmt(metric, v):
    if v is None:
        return "-"
    return f"{v:.2f}" if metric == "Distance" else f"{100 * v:.1f}"


def best_set(metric, values):
    present = [v for v in values if v is not None]
    if not present:
        return set()
    b = min(present) if metric == "Distance" else max(present)
    return {i for i, v in enumerate(values) if v is not None and abs(v - b) < 1e-12}


def columns(data, include_train):
    found = {s for res, _ in data.values() for s in res["results"]}
    cols = [s for s in SPLIT_ORDER if s in found and (include_train or s != "train")]
    return cols + ["iid avg", "ood avg"]


# ---------- tables ----------
def metric_table(runs, data, metric, cols):
    rows, per_col = [], {c: [col_value(data[r][0], metric, c) for r in runs] for c in cols}
    bests = {c: best_set(metric, per_col[c]) for c in cols}
    for i, r in enumerate(runs):
        rows.append([condition(r)] + [(fmt(metric, per_col[c][i]), i in bests[c]) for c in cols])
    return rows


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row[0]] + [f"**{t}**" if b and t != "-" else t for t, b in row[1:]]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def tex(header, rows):
    esc = lambda s: s.replace("_", r"\_").replace("%", r"\%")
    out = [r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    for row in rows:
        cells = [esc(row[0])] + [rf"\textbf{{{t}}}" if b and t != "-" else t for t, b in row[1:]]
        out.append(" & ".join(cells) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}"])


def delta_table(runs, data, metric, cols):
    base = next((r for r in runs if condition(r) == "A"), None)
    if base is None:
        return None
    rows = []
    for r in runs:
        if r == base:
            continue
        cells = []
        for c in cols:
            a, b = col_value(data[base][0], metric, c), col_value(data[r][0], metric, c)
            cells.append(("-" if a is None or b is None else f"{100 * (b - a):+.1f}", False))
        rows.append([condition(r)] + cells)
    return rows


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--local", default=None, help="read <RUN>/results.json from this folder instead of HF")
    ap.add_argument("--out", default="comparison")
    ap.add_argument("--include_train", action="store_true", help="also show the train split in the tables")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    data = fetch_local(a.local) if a.local else fetch_hf(a.repo)
    if not data:
        raise SystemExit("no <RUN>/results.json found")
    runs = sorted(data, key=cond_rank)
    for r in runs:
        if data[r][0].get("limit"):
            print(f"WARNING: {r} results.json is a smoke run (limit={data[r][0]['limit']})")
    os.makedirs(a.out, exist_ok=True)
    cols = columns(data, a.include_train)
    print("runs:", ", ".join(f"{condition(r)}={r}" for r in runs))

    # all_results.csv (always includes every split found, train included)
    all_splits = [s for s in SPLIT_ORDER if any(s in data[r][0]["results"] for r in runs)]
    head = ["run", "condition", "split", "total"] + METRICS
    head += [f"{m} ci95_lo" for m in METRICS if m in PROP] + [f"{m} ci95_hi" for m in METRICS if m in PROP]
    with open(os.path.join(a.out, "all_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        for r in runs:
            for s in all_splits:
                m = data[r][0]["results"].get(s)
                if not m:
                    continue
                n = m.get("Total", 0)
                cis = {k: wilson(m[k], n) for k in METRICS if k in PROP and m.get(k) is not None}
                w.writerow([r, condition(r), s, n] + [m.get(k) for k in METRICS]
                           + [cis.get(k, ("", ""))[0] for k in METRICS if k in PROP]
                           + [cis.get(k, ("", ""))[1] for k in METRICS if k in PROP])

    # hyperparams.csv
    hp_head = ["run", "condition", "epochs", "lr", "bs_per_device", "grad_accum", "effective_bs", "max_seq_len",
               "lora_r", "lora_alpha", "lora_dropout", "best_checkpoint", "best_dev_exact_match_subset", "dev_subset_n"]
    hp_rows = []
    for r in runs:
        i = data[r][1]
        ga = i.get("grad_accum", 1)
        lora = i.get("lora", {})
        hp_rows.append([r, condition(r), i.get("epochs"), i.get("lr"), i.get("bs"), ga,
                        i.get("effective_bs", (i.get("bs") or 0) * ga if i.get("bs") else None), i.get("max_seq_len"),
                        lora.get("r"), lora.get("alpha"), lora.get("dropout"),
                        os.path.basename(i.get("best_checkpoint") or ""), i.get("best_dev_exact_match_subset"),
                        i.get("dev_subset_n")])
    with open(os.path.join(a.out, "hyperparams.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([hp_head] + hp_rows)

    # comparison.md (+ optional .tex)
    header = ["Condition"] + [SPLIT_NAME[c] for c in cols]
    n_note = {s: data[runs[0]][0]["results"].get(s, {}).get("Total") for s in cols if s in SPLIT_ORDER}
    parts = ["# Comparison of runs\n",
             "Values are percentages except Distance (mean A* length to the goal over valid predictions, "
             "lower is better, 1 = at goal). Best per column in bold. 'iid avg' and 'OOD avg' are unweighted "
             "means over their splits. 'Goal not reachable' examples are excluded from every split. "
             "Number of evaluated examples: " + ", ".join(f"{s}={n}" for s, n in n_note.items()) + ".\n",
             "## Hyperparameters\n",
             "| Condition | Run | Epochs | LR | Batch (per device x accum = effective) | Max seq len | LoRA r/alpha | Best dev EM (subset) |\n"
             "|---|---|---|---|---|---|---|---|"]
    for row in hp_rows:
        r, c, ep, lr, bs, ga, eb, ml, lr_r, lr_a, _, _, bdev, _ = (("-" if v is None else v) for v in row)
        batch = "-" if bs == "-" else f"{bs} x {ga} = {eb}"
        parts.append(f"| {c} | `{r}` | {ep} | {lr} | {batch} | {ml} | {lr_r}/{lr_a} | "
                     f"{'-' if bdev == '-' else f'{100 * bdev:.1f}'} |")
    tex_parts = []
    for m in METRICS:
        rows = metric_table(runs, data, m, cols)
        parts += [f"\n## {m}\n", md(header, rows)]
        tex_parts += [f"% {m}", tex(header, rows), ""]
    for m in ("Exact match", "Success rate"):
        d = delta_table(runs, data, m, cols)
        if d:
            parts += [f"\n## {m}: difference vs A (percentage points)\n", md(header, d)]
    text = "\n".join(parts) + "\n"
    with open(os.path.join(a.out, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(text)
    if a.latex:
        with open(os.path.join(a.out, "comparison.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(tex_parts))
    print(text)
    print(f"saved to {a.out}/: all_results.csv, hyperparams.csv, comparison.md" + (", comparison.tex" if a.latex else ""))


if __name__ == "__main__":
    main()
