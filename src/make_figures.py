"""make_figures.py - Figure 1 (SR by split) and Figure 2 (SR on OOD 7x7 with / without an index 6).

  python make_figures.py --results comparison/all_results.csv --zip outputs/all_predictions.zip
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import evaluate_results as E  # noqa: E402

ORDER = ["A", "R-num", "R-nl", "G-num", "G-nl"]
STYLE = {"A": ("#666666", "-", "D"), "R-num": ("#1f5fbf", "--", "s"), "R-nl": ("#0b3d91", "-", "o"),
         "G-num": ("#5b9bd5", "--", "s"), "G-nl": ("#3f7fd0", "-", "o")}
CODE = {"a": "A", "rnum": "R-num", "rnl": "R-nl", "gnum": "G-num", "gnl": "G-nl"}


def figure1(results_csv, out):
    df = pd.read_csv(results_csv)
    sr = df.pivot(index="condition", columns="split", values="Success rate") * 100
    sr["ID"] = sr[["test_seen", "test_unseen"]].mean(axis=1)  # unweighted mean, as in Table 3
    xs = ["ID", "ood_5x5", "ood_7x7", "ood_moreobsts"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c in ORDER:
        color, ls, mk = STYLE[c]
        ax.plot(range(4), [sr.loc[c, x] for x in xs], color=color, linestyle=ls, marker=mk, label=c)
    ax.set_xticks(range(4), ["ID (avg)", "5×5", "7×7", "More obstacles"])
    ax.set_ylabel("Success rate (%)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print("saved", out)


def figure2(zip_path, out):
    E.extract_predictions(zip_path)
    R = E.build_frame()
    x7 = R[R["split"] == "ood_7x7"]
    t = (x7.groupby(["cond", "has6"])["sr"].mean().unstack() * 100).rename(index=CODE).loc[ORDER]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, c in enumerate(ORDER):
        ax.plot([t.loc[c, False], t.loc[c, True]], [i, i], color="#0b3d91" if c == "A" else "#888888")
        ax.scatter(t.loc[c, False], i, facecolors="white", edgecolors="black", zorder=3)
        ax.scatter(t.loc[c, True], i, color="black", zorder=3)
    ax.set_yticks(range(len(ORDER)), ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Success rate (%) on OOD 7×7  (○ no index 6, ● index 6 present)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print("saved", out)
    print(t.round(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="comparison/all_results.csv")
    ap.add_argument("--zip", default="all_predictions.zip")
    a = ap.parse_args()
    figure1(a.results, "figure1_sr_by_split.png")
    figure2(a.zip, "figure2_index6_7x7.png")
