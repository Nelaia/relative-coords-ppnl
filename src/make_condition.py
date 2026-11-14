"""make_condition.py - build the R-num / G-num versions of single_goal_original.

Only `nl_description` is rewritten. `world`, `solution_coordinates`, `agent_as_a_point`, ... are kept,
so train.py / eval.py / push_hf.py work unchanged. The absolute description is kept in
`nl_description_abs` for traceability. Coordinates keep the repo's (row, col) order and each value is
"object minus reference" (row: + = down, col: + = right).

  rnum : reference = robot.  "Go from (0,0) to (+3,+3)", obstacles as offsets from the robot
  gnum : reference = goal.   "Go from (-3,-3) to (0,0)", obstacles as offsets from the goal
  rnl  : same offsets as rnum, written in words ("The goal is 3 cells to the right and 3 cells down from the robot.")
  gnl  : same offsets as gnum, written in words ("The robot is 3 cells to the left and 3 cells up from the goal.")
         words: horizontal (right/left) first, then vertical (down/up); a zero axis is omitted.

Every converted example is verified by mapping it back to absolute coordinates and comparing with the
`world` grid (the script stops on the first mismatch).

Kaggle usage:
  !python make_condition.py --cond rnum --dst /kaggle/working/data_rnum
  %env PPNL_ROOT=/kaggle/working/data_rnum
  !python data.py --check
  !python train.py --run_name r-num_single-goal-original
(same with --cond gnum). Add --drop_size to remove the "You are in a N by N world." sentence.
"""
import argparse
import json
import os
import re

import data as D

PAT = re.compile(r"^(You are in a \d+ by \d+ world\. )(There are obstacles that you have to avoid at: )"
                 r"(.*?)(\. Go from )\((\d+),(\d+)\)( to )\((\d+),(\d+)\)$")
TUP = re.compile(r"\(([+-]?\d+),([+-]?\d+)\)")
# Sentence stating the coordinate origin, inserted after the size sentence (covers obstacles, start and goal).
REF_SENTENCE = {
    "rnum": "All coordinates are relative to the robot. ",
    "gnum": "All coordinates are relative to the goal. ",
}


def sgn(v):
    return f"+{v}" if v > 0 else str(v)


def pt(r, c):
    return f"({sgn(r)},{sgn(c)})"


def cells(n):
    return f"{n} cell" + ("" if n == 1 else "s")


def phrase(off, ref_name):
    """(drow, dcol) offset -> natural language. Horizontal first, then vertical; a zero axis is omitted.
    Only-horizontal uses "of the <ref>", everything else "from the <ref>"."""
    dr, dc = off
    parts = []
    if dc:
        parts.append(f"{cells(abs(dc))} to the {'right' if dc > 0 else 'left'}")
    if dr:
        parts.append(f"{cells(abs(dr))} {'down' if dr > 0 else 'up'}")
    conn = "of" if (dc and not dr) else "from"
    return " and ".join(parts) + f" {conn} the {ref_name}"


def convert_nl(size, obs, s, g, cond):
    ref = s if cond == "rnl" else g
    ref_name = "robot" if cond == "rnl" else "goal"
    off = lambda p: (p[0] - ref[0], p[1] - ref[1])
    if cond == "rnl":
        head = f"The goal is {phrase(off(g), ref_name)}."
    else:
        head = f"The robot is {phrase(off(s), ref_name)}."
    obs_s = " ".join(f"Obstacle {i} is {phrase(off(o), ref_name)}." for i, o in enumerate(obs, 1))
    return size, f"{head} There are obstacles that you have to avoid. {obs_s} Go from the robot to the goal."


def parse_nl(desc):
    """Inverse of convert_nl: returns (obstacle offsets, offset of the non-reference point)."""
    def off(txt):
        h = re.search(r"(\d+) cells? to the (right|left)", txt)
        v = re.search(r"(\d+) cells? (down|up)", txt)
        dc = (int(h.group(1)) * (1 if h.group(2) == "right" else -1)) if h else 0
        dr = (int(v.group(1)) * (1 if v.group(2) == "down" else -1)) if v else 0
        return dr, dc
    head = re.search(r"The (?:goal|robot) is (.*?) (?:of|from) the (?:robot|goal)\.", desc)
    obs = re.findall(r"Obstacle \d+ is (.*?) (?:of|from) the (?:robot|goal)\.", desc)
    return [off(o) for o in obs], off(head.group(1))


def convert(desc, cond, drop_size=False):
    m = PAT.match(desc.strip())
    if not m:
        raise ValueError(f"unparsable description: {desc!r}")
    size, obs_pre, obs, mid1, sr, sc, mid2, gr, gc = m.groups()
    sr, sc, gr, gc = int(sr), int(sc), int(gr), int(gc)
    if cond in ("rnl", "gnl"):
        pts = [(int(a), int(b)) for a, b in TUP.findall(obs)]
        size, body = convert_nl(size, pts, (sr, sc), (gr, gc), cond)
        return ("" if drop_size else size) + body
    ref = (sr, sc) if cond == "rnum" else (gr, gc)
    new_obs = TUP.sub(lambda t: pt(int(t.group(1)) - ref[0], int(t.group(2)) - ref[1]), obs)
    start = pt(sr - ref[0], sc - ref[1])
    goal = pt(gr - ref[0], gc - ref[1])
    return ("" if drop_size else size) + REF_SENTENCE[cond] + f"{obs_pre}{new_obs}{mid1}{start}{mid2}{goal}"


def to_absolute(desc, cond, world):
    """Inverse mapping using the world grid for the reference point; returns (obstacles, start, goal)."""
    if cond in ("rnl", "gnl"):
        obs, other = parse_nl(desc)
        s, g = ((0, 0), other) if cond == "rnl" else (other, (0, 0))
    else:
        t = [(int(a), int(b)) for a, b in TUP.findall(desc)]
        obs, (s, g) = t[:-2], t[-2:]
    cells = {v: [] for v in (1, 2, 3)}
    for r, row in enumerate(world):
        for c, v in enumerate(row):
            if v in cells:
                cells[v].append((r, c))
    ref = cells[2][0] if cond in ("rnum", "rnl") else cells[3][0]
    add = lambda p: (p[0] + ref[0], p[1] + ref[1])
    return sorted(add(o) for o in obs), add(s), add(g), sorted(cells[1]), cells[2][0], cells[3][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["rnum", "gnum", "rnl", "gnl"])
    ap.add_argument("--dst", required=True)
    ap.add_argument("--src", default=None, help="folder containing iid/ and ood/ (default: auto-detect)")
    ap.add_argument("--drop_size", action="store_true")
    a = ap.parse_args()

    src = a.src or D.find_root()
    total = 0
    for split, rel in D.SPLITS.items():
        data = D.load_split(split, src)
        out = []
        for x in data:
            y = dict(x)
            y["nl_description_abs"] = x["nl_description"]
            y["nl_description"] = convert(x["nl_description"], a.cond, a.drop_size)
            o, s, g, wo, ws, wg = to_absolute(y["nl_description"], a.cond, x["world"])
            assert o == wo and s == ws and g == wg, f"round-trip mismatch in {split}: {x['nl_description']!r}"
            out.append(y)
        path = os.path.join(a.dst, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f)
        total += len(out)
        mx = max(len(y["nl_description"]) for y in out)
        print(f"{split:14s} n={len(out):5d} verified={len(out):5d} max_chars={mx}")
    print(f"done: {total} examples written to {a.dst}")
    ex = D.load_split("train", a.dst)[0]
    print("example ABS:", ex["nl_description_abs"])
    print(f"example {a.cond.upper()}:", ex["nl_description"], "->", ex["agent_as_a_point"].strip())
    print(f"next: %env PPNL_ROOT={a.dst}")


if __name__ == "__main__":
    main()
