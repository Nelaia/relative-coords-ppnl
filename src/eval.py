"""eval.py: evaluate the QLoRA adapter on dev, test_seen, test_unseen and 3 OOD splits.

Metrics are ported from the repo's evaluate/executor-point-sg.py (Exact match, Success rate,
Distance, Feasible, Optimal; "Goal not reachable" examples are skipped). The functions below keep
the repo's logic (including its A* and its character-length 'Optimal') so numbers stay comparable.
An extra 'Optimal (steps)' compares the number of actions instead of characters.

Smoke test :  python eval.py --limit 64
Full run   :  python eval.py              (resumable: re-run the same command after an interruption)
From HF    :  python eval.py --hf_repo Nelaia/qwen2.5-3b-ppnl-lora --subfolder a_abs_single-goal-original
"""
import argparse
import heapq
import json
import os

import data as D

RUN_NAME = "a_abs_single-goal-original"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./working"
ACTS = ["up", "down", "left", "right"]
MOVES = {"left": (0, -1), "right": (0, 1), "down": (1, 0), "up": (-1, 0)}
METRICS = ["Exact match", "Success rate", "Distance", "Feasible", "Optimal", "Optimal (steps)"]


# ---------- metrics (ported from executor-point-sg.py) ----------
def _fix_concat(actions):
    for a in ACTS:
        for b in ACTS:
            actions = actions.replace(a + b, a + " " + b)
    return actions


def _start_goal(grid):
    start = goal = None
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == 2:
                start = (r, c)
            if v == 3:
                goal = (r, c)
    return start, goal


def is_goal(grid, pos, actions):
    """1 = reaches goal, 2 = hits obstacle, 0 = otherwise (incl. leaving the grid)."""
    nx, ny = len(grid), len(grid[0])
    for act in _fix_concat(actions).split(" "):
        dx, dy = MOVES.get(act, (0, 0))
        pos = (pos[0] + dx, pos[1] + dy)
        if pos[0] < 0 or pos[1] < 0 or pos[0] >= nx or pos[1] >= ny:
            return 0
        elif grid[pos[0]][pos[1]] == 1:
            return 2
    return 1 if grid[pos[0]][pos[1]] == 3 else 0


def a_star(grid, start, goal):
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    visited, heap = set(), []
    heapq.heappush(heap, (0, start, []))
    while heap:
        cost, current, path = heapq.heappop(heap)
        if grid[current[0]][current[1]] == 3:
            return len(path + [current])
        if current in visited:
            continue
        visited.add(current)
        for a in actions:
            nb = (current[0] + a[0], current[1] + a[1])
            if 0 <= nb[0] < len(grid) and 0 <= nb[1] < len(grid[1]):
                if grid[nb[0]][nb[1]] != 1:
                    heapq.heappush(heap, (cost + 1 + heuristic(nb), nb, path + [current]))
    return 100


def distance_from_goal(grid, pos, actions):
    """-1 obstacle, 100 off-grid, else A* length from the final position to the goal."""
    nx, ny = len(grid), len(grid[0])
    _, goal = _start_goal(grid)
    for act in actions.split(" "):
        dx, dy = MOVES.get(act, (0, 0))
        pos = (pos[0] + dx, pos[1] + dy)
        if pos[0] < 0 or pos[1] < 0 or pos[0] >= nx or pos[1] >= ny:
            return 100
        if grid[pos[0]][pos[1]] == 1:
            return -1
    return a_star(grid, pos, goal)


def example_metrics(ex, pred):
    grid, truth = ex["world"], ex["agent_as_a_point"]
    init, _ = _start_goal(grid)
    ok = is_goal(grid, init, pred) == 1
    dist = distance_from_goal(grid, init, pred)
    n_truth = len(truth.split())
    return {
        "em": truth.replace(" ", "") == pred.replace(" ", ""),
        "success": ok,
        "optimal": ok and len(pred.replace(" ", "")) <= len(truth.replace(" ", "")),
        "optimal_steps": ok and len(_fix_concat(pred).split()) <= n_truth,
        "dist": None if dist in (-1, 100) else dist,
    }


def aggregate(examples, preds):
    rows = [example_metrics(x, p) for x, p in zip(examples, preds) if D.is_reachable(x)]
    n = len(rows)
    valid = [r["dist"] for r in rows if r["dist"] is not None]
    return {
        "Total": n,
        "Exact match": sum(r["em"] for r in rows) / n,
        "Success rate": sum(r["success"] for r in rows) / n,
        "Distance": (sum(valid) / len(valid)) if valid else float("nan"),
        "Feasible": len(valid) / n,
        "Optimal": sum(r["optimal"] for r in rows) / n,
        "Optimal (steps)": sum(r["optimal_steps"] for r in rows) / n,
    }


# ---------- generation (resumable) ----------
def run_split(name, model, tok, root, out_dir, limit, chunk=256, bs=64):
    from train import generate_plans
    data = D.load_split(name, root)
    if limit:
        data = data[:limit]
    path = os.path.join(out_dir, f"{name}.jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done[r["i"]] = r["generated"]
            except Exception:  # truncated last line after an interruption
                pass
    todo = [i for i, x in enumerate(data) if D.is_reachable(x) and i not in done]
    print(f"[{name}] total={len(data)} done={len(done)} todo={len(todo)}")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as fb:
            fb.seek(-1, os.SEEK_END)
            if fb.read(1) != b"\n":  # partial last line after a crash: close it so new records stay intact
                with open(path, "a", encoding="utf-8") as fa:
                    fa.write("\n")
    with open(path, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), chunk):
            idx = todo[s:s + chunk]
            preds = generate_plans(model, tok, [D.build_prompt(data[i]) for i in idx], bs=bs)
            for i, p in zip(idx, preds):
                f.write(json.dumps({"i": i, "generated": p}) + "\n")
                done[i] = p
            f.flush()
            if (s // chunk) % 4 == 3 or s + chunk >= len(todo):
                print(f"[{name}] {len(done)}/{sum(D.is_reachable(x) for x in data)}")
    preds_all = [done.get(i, "") for i in range(len(data))]
    # same layout as the repo's executor input, so its executor-point-sg.py can cross-check
    with open(os.path.join(out_dir, f"preds_{name}.json"), "w", encoding="utf-8") as f:
        json.dump([{"english": x["nl_description"], "ground_truth": x["agent_as_a_point"],
                    "generated": [p]} for x, p in zip(data, preds_all)], f)
    return aggregate(data, preds_all)


def print_table(results):
    cols = ["Total"] + METRICS
    print("\n" + f"{'split':14s}" + "".join(f"{c:>17s}" for c in cols))
    for name, m in results.items():
        print(f"{name:14s}" + f"{m['Total']:>17d}" + "".join(f"{m[c]:>17.4f}" for c in METRICS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default=RUN_NAME)
    ap.add_argument("--adapter", default=None, help="local adapter dir (default: <WORK>/<run>/best_adapter)")
    ap.add_argument("--hf_repo", default=None)
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--splits", nargs="+", default=list(D.SPLITS))
    ap.add_argument("--limit", type=int, default=0, help=">0 = smoke test on the first N examples per split")
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    base_dir = os.path.join(WORK, a.run_name)
    out_dir = os.path.join(base_dir, "eval_smoke" if a.limit else "eval")
    os.makedirs(out_dir, exist_ok=True)

    tok = D.get_tokenizer()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(D.MODEL_NAME, quantization_config=bnb, device_map={"": 0},
                                                torch_dtype=torch.float16, attn_implementation="sdpa")
    if a.hf_repo:
        model = PeftModel.from_pretrained(base, a.hf_repo, subfolder=a.subfolder)
        src = f"{a.hf_repo}/{a.subfolder}"
    else:
        src = a.adapter or os.path.join(base_dir, "best_adapter")
        model = PeftModel.from_pretrained(base, src)
    model.eval()
    print("adapter:", src)

    root = D.find_root()
    results = {}
    for name in a.splits:
        results[name] = run_split(name, model, tok, root, out_dir, a.limit, bs=a.bs)
        with open(os.path.join(out_dir, "results.json"), "w") as f:  # save after every split
            json.dump({"adapter": src, "limit": a.limit, "results": results}, f, indent=2)
    print_table(results)
    print("saved:", os.path.join(out_dir, "results.json"))


if __name__ == "__main__":
    main()
