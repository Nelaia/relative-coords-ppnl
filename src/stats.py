"""stats.py - paired significance tests on per-sample Success rate (paper Table 4).

For every comparison and group it reports the paired difference in SR (percentage points) with a 95% Wald
interval, the exact McNemar p-value, and the Holm-adjusted p-value. Holm is applied separately to the
16 comparisons with A and to the 16 comparisons among relative conditions.

Needs outputs/all_predictions.zip (generate_predictions.py). Writes table4.csv and table4.md.
  python stats.py --zip outputs/all_predictions.zip
"""
import argparse
import math

import pandas as pd
from scipy.stats import binomtest, norm

import evaluate_results as E

GROUPS = ["ID", "ood_5x5", "ood_7x7", "ood_moreobsts"]
LABEL = {"a": "A", "rnum": "R-num", "rnl": "R-nl", "gnum": "G-num", "gnl": "G-nl"}
VS_A = [(c, "a") for c in ("rnum", "rnl", "gnum", "gnl")]
AMONG_REL = [("rnum", "gnum"), ("rnl", "gnl"), ("rnl", "rnum"), ("gnl", "gnum")]


def holm(pvals):
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired(R, c1, c2, grp, z=norm.ppf(0.975)):
    a = R[(R["cond"] == c1) & (R["grp"] == grp)].sort_values(["split", "idx"])["sr"].values
    b = R[(R["cond"] == c2) & (R["grp"] == grp)].sort_values(["split", "idx"])["sr"].values
    assert len(a) == len(b)
    n = len(a)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    diff = (n10 - n01) / n
    var = (n10 + n01 - (n10 - n01) ** 2 / n) / n ** 2  # variance of a paired difference in proportions
    se = math.sqrt(max(var, 0.0))
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {"n": n, "diff_pp": 100 * diff, "lo": 100 * (diff - z * se), "hi": 100 * (diff + z * se),
            "n10": n10, "n01": n01, "p": p}


def family(R, pairs):
    rows = [dict(paired(R, c1, c2, g), comparison=f"{LABEL[c1]} - {LABEL[c2]}", group=g)
            for c1, c2 in pairs for g in GROUPS]
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="all_predictions.zip")
    ap.add_argument("--out", default="table4")
    a = ap.parse_args()

    E.extract_predictions(a.zip)
    R = E.build_frame()
    rows = [dict(r, family="vs A") for r in family(R, VS_A)] + \
           [dict(r, family="among relative") for r in family(R, AMONG_REL)]
    df = pd.DataFrame(rows)[["family", "comparison", "group", "n", "diff_pp", "lo", "hi", "n10", "n01", "p", "p_holm"]]
    df.to_csv(a.out + ".csv", index=False)

    wide = df.assign(cell=df.apply(lambda r: f"{r.diff_pp:+.2f} [{r.lo:.2f}, {r.hi:.2f}]", axis=1)) \
             .pivot(index=["family", "comparison"], columns="group", values="cell")[GROUPS]
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write(wide.reset_index().to_markdown(index=False))
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nsaved {a.out}.csv and {a.out}.md")


if __name__ == "__main__":
    main()
