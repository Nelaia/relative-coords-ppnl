import os
import re
import zipfile
import numpy as np
import pandas as pd
from scipy.stats import binomtest

PRED_DIR = "preds"


def extract_predictions(zip_path="all_predictions.zip", out_dir=PRED_DIR):
    """Unzip the per-sample predictions produced by generate_predictions.py."""
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
CONDS = ["a", "rnum", "rnl", "gnum", "gnl"]
SPLITS = ["test_seen", "test_unseen", "ood_5x5", "ood_7x7", "ood_moreobsts"]


def parse_a_input(inp):
    n = int(re.search(r"(\d+)\s*by\s*\d+", inp).group(1))
    head, tail = inp.split("Go from")
    obs = {tuple(map(int, m)) for m in re.findall(r"\((\d+),\s*(\d+)\)", head)}
    s, g = [tuple(map(int, m)) for m in re.findall(r"\((\d+),\s*(\d+)\)", tail)]
    return n, obs, s, g


def simulate_outcome(n, obs, s, g, pred):
    r, c = s
    toks = str(pred).strip().split()
    for t in toks:
        if t not in MOVES:
            return "bad_token"
        r += MOVES[t][0]
        c += MOVES[t][1]
        if not (0 <= r < n and 0 <= c < n):
            return "out_of_grid"
        if (r, c) in obs:
            return "collision"
    if (r, c) == g:
        return "success"
    manhattan = abs(s[0] - g[0]) + abs(s[1] - g[1])
    return "short" if len(toks) < manhattan else "wrong_end"


def load_df(cond, split):
    return pd.read_csv(f"{PRED_DIR}/pred_{cond}_{split}.csv")


def mcnemar_test(df, c1, c2, grp):
    a = df[(df["cond"] == c1) & (df["grp"] == grp)].sort_values(["split", "idx"])["sr"].values
    b = df[(df["cond"] == c2) & (df["grp"] == grp)].sort_values(["split", "idx"])["sr"].values
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if (n10 + n01) > 0 else 1.0
    diff = (a.mean() - b.mean()) * 100
    return diff, n10, n01, p


def build_frame():
    """One row per (condition, split, sample) with the simulated outcome, SR, EM, index-6 and border flags."""
    rows = []
    for s in SPLITS:
        df_a = load_df("a", s)
        geo = [parse_a_input(x) for x in df_a["input"]]
        for c in CONDS:
            d = load_df(c, s)
            assert (d["answer"].fillna("") == df_a["answer"].fillna("")).all()
            for i, (n, obs, st, g) in enumerate(geo):
                pred_str = "" if pd.isna(d["pred"].iloc[i]) else str(d["pred"].iloc[i])
                ans_str = "" if pd.isna(d["answer"].iloc[i]) else str(d["answer"].iloc[i])
                res = simulate_outcome(n, obs, st, g, pred_str)
                rows.append({
                    "cond": c,
                    "split": s,
                    "idx": i,
                    "n": n,
                    "res": res,
                    "sr": res == "success",
                    "em": "".join(pred_str.split()) == "".join(ans_str.split()),
                    "has6": any(max(p) == 6 for p in list(obs) + [st, g]),
                    "border": min(st[0], st[1], n - 1 - st[0], n - 1 - st[1]),
                })

    R = pd.DataFrame(rows)
    R["grp"] = R["split"].replace({"test_seen": "ID", "test_unseen": "ID"})
    return R


def main():
    extract_predictions()
    R = build_frame()

    # 1. Success Rate (SR) and Exact Match (EM)
    print("=== 1. SUCCESS RATE (SR %) & EXACT MATCH (EM %) ===")
    print("SR (%):")
    print((R.groupby(["cond", "split"])["sr"].mean().unstack() * 100).round(2))
    print("\nEM (%):")
    print((R.groupby(["cond", "split"])["em"].mean().unstack() * 100).round(2))

    # 2. Paired McNemar Significance Tests on SR
    print("\n=== 2. PAIRED MCNEMAR TESTS ON SR ===")
    for c in ["rnum", "rnl", "gnum", "gnl"]:
        for g in ["ID", "ood_5x5", "ood_7x7", "ood_moreobsts"]:
            diff, n10, n01, p = mcnemar_test(R, c, "a", g)
            print(f"{c:<5} vs a    | {g:<13} | diff={diff:+6.2f}% | n10={n10:>3}, n01={n01:>3} | p={p:.4e}")

    print("\n--- Pairwise Comparisons Among Relative Conditions ---")
    for c1, c2 in [("rnl", "rnum"), ("gnl", "gnum"), ("rnl", "gnl"), ("rnum", "gnum")]:
        for g in ["ID", "ood_5x5", "ood_7x7", "ood_moreobsts"]:
            diff, n10, n01, p = mcnemar_test(R, c1, c2, g)
            print(f"{c1:<5} vs {c2:<4} | {g:<13} | diff={diff:+6.2f}% | n10={n10:>3}, n01={n01:>3} | p={p:.4e}")

    # 3. Unseen Index '6' Hypothesis on 7x7 Grid
    print("\n=== 3. UNSEEN INDEX '6' HYPOTHESIS ON OOD_7x7 (SR %) ===")
    x7 = R[R["split"] == "ood_7x7"]
    print(x7.groupby(["cond", "has6"])["sr"].agg(sr=lambda s: round(s.mean() * 100, 2), count="size"))

    # 4. Boundary Information Loss Analysis
    print("\n=== 4. BOUNDARY DISTANCE ANALYSIS (SR %) ===")
    yb = R[R["split"].isin(["test_seen", "test_unseen", "ood_5x5", "ood_7x7"])]
    print("All 4 splits combined:")
    print((yb.groupby(["cond", "border"])["sr"].mean().unstack() * 100).round(2))

    for sp in ["test_seen", "test_unseen", "ood_5x5", "ood_7x7"]:
        sub = R[R["split"] == sp]
        print(f"\n--- Split: {sp} (n per condition = {len(sub[sub['cond'] == 'a'])}) ---")
        print("Sample counts by border distance:", sub[sub["cond"] == "a"].groupby("border").size().to_dict())
        print((sub.groupby(["cond", "border"])["sr"].mean().unstack() * 100).round(2))

    sub_no7 = R[R["split"].isin(["test_seen", "test_unseen", "ood_5x5"])]
    print("\nID (test_seen + test_unseen) + ood_5x5 (excluding 7x7):")
    print("Sample counts by border distance:", sub_no7[sub_no7["cond"] == "a"].groupby("border").size().to_dict())
    print((sub_no7.groupby(["cond", "border"])["sr"].mean().unstack() * 100).round(2))

    # Out-of-grid error rate vs. starting at border (border == 0) across all 5 conditions
    print("\n--- Out-of-Grid Errors: Share Starting at Border (border == 0) ---")
    og_all = R[R["res"] == "out_of_grid"]
    err_border0 = (og_all.groupby("cond")["border"].apply(lambda s: (s == 0).mean()) * 100).round(2)
    base_border0 = (R.groupby("cond")["border"].apply(lambda s: (s == 0).mean()) * 100).round(2)
    comp_df = pd.DataFrame({
        "Out-of-grid border=0 (%)": err_border0,
        "Baseline border=0 (%)": base_border0,
        "Total out-of-grid count": og_all.groupby("cond").size(),
    })
    print(comp_df)

    # 5. Detailed Failure Mode Breakdown (%)
    print("\n=== 5. FAILURE MODE BREAKDOWN (%) ===")
    print((pd.crosstab([R["split"], R["cond"]], R["res"], normalize="index") * 100).round(2))

    # 6. Dev Split Sanity Check (First 500 vs. Remaining 1,493 Samples)
    print("\n=== 6. DEV SPLIT CHECK: FIRST 500 VS. REMAINING SAMPLES (EM %) ===")
    for c in CONDS:
        d_dev = load_df(c, "dev")
        em_vec = [
            "".join(str(p).split()) == "".join(str(a).split())
            for p, a in zip(d_dev["pred"].fillna(""), d_dev["answer"].fillna(""))
        ]
        print(f"{c:<5} | dev[:500] = {np.mean(em_vec[:500])*100:6.2f}% | dev[500:] = {np.mean(em_vec[500:])*100:6.2f}% | full dev = {np.mean(em_vec)*100:6.2f}%")


if __name__ == "__main__":
    main()