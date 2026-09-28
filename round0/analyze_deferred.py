"""Deferred-materialization ceiling from the Round 0 Prism traces (no GPU).

For every executed forward we ask whether the row it was run on is an ancestor of (or is) a row
that the search still holds at the end. Work on rows that are not is 'commitment waste': it was
materialized and then pruned. Two accountings of 'needed':
  (a) search-pruned:   needed = ancestors of the final_K rows returned by generate_hts
  (b) top-1 only:      needed = ancestors of the rank-1 final row (best-of-K selection waste)
"""
import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict


def analyze(path, res_row):
    recs = [json.loads(l) for l in open(path)]
    fwd_by_seq = {r["seq"]: r for r in recs if r["t"] == "fwd"}
    states = [r for r in recs if r["t"] == "state"]
    prunes = iter([r for r in recs if r["t"] == "prune"])
    ver = [r for r in recs if r["t"] == "fwd" and r["kind"] in ("verifier", "verifier_final")]
    vrec = [r for r in recs if r["t"] == "verifier"]

    # rebuild tree; node = (event_idx, row)
    parent = {}
    node_flops = defaultdict(float)
    node_kind = {}
    events = []
    pending = None
    for ei, st in enumerate(states):
        fw = fwd_by_seq[st["seq"]]
        if st["kind"] == "denoise" and pending is not None:
            bmap = {b["new"]: b["src"] for b in pending["branch"]}
            par = lambda j: (ei - 1, bmap[j])
            pending = None
        else:
            par = (lambda j: (ei - 1, j)) if ei > 0 else (lambda j: None)
        for j in range(len(st["rows"])):
            parent[(ei, j)] = par(j)
            node_flops[(ei, j)] += fw["flops_row"]
            node_kind[(ei, j)] = st["kind"]
        events.append(st)
        if st["kind"] == "rough":
            pending = next(prunes)
    # verifier forwards attributed to the row scored at that prune event (rough event with same block/step)
    ev_index = {(st["kind"], st["block"], st["step"]): ei for ei, st in enumerate(states)}
    last_ei = len(states) - 1
    for r in ver:
        if r["kind"] == "verifier":
            node_flops[(ev_index[("rough", r["block"], r["step"])], r["row"])] += r["flops_row"] * r["n"]
        else:
            node_flops[(last_ei, r["row"])] += r["flops_row"] * r["n"]

    total = sum(node_flops.values())
    final_rows = [(last_ei, j) for j in range(len(states[last_ei]["rows"]))]

    # rank-1 final row: match verifier_final text hash to res.jsonl all_trajectories rank 1
    rank1_text = res_row["all_trajectories"][0]["resp"]
    h1 = hashlib.blake2b(rank1_text.encode(), digest_size=16).hexdigest()
    r1_rows = [v["row"] for v in vrec if v["block"] is None and v["h_text"] == h1]
    top1_rows = [(last_ei, r1_rows[0])] if r1_rows else final_rows[:1]

    def needed_set(roots):
        need = set()
        for n in roots:
            while n is not None and n not in need:
                need.add(n); n = parent[n]
        return need

    out = {}
    for name, roots in (("search", final_rows), ("top1", top1_rows)):
        need = needed_set(roots)
        waste = sum(f for n, f in node_flops.items() if n not in need)
        # split waste: rows never scored before (pre-evidence) vs rows that survived an earlier prune
        scored_before = set()
        pre = 0.0
        for n, f in node_flops.items():
            if n in need:
                continue
            # walk ancestry: has any ancestor been a 'rough' (scored) node?
            a, seen_score = parent[n], False
            while a is not None:
                if node_kind[a] == "rough":
                    seen_score = True; break
                a = parent[a]
            if not seen_score:
                pre += f
        # physical width per event under oracle deferral
        width_orig = [len(st["rows"]) for st in states]
        width_need = [sum(1 for j in range(len(st["rows"])) if (ei, j) in need) for ei, st in enumerate(states)]
        out[name] = {"W_commit": waste / total, "W_commit_pre_evidence": pre / total,
                     "peak_width_orig": max(width_orig), "peak_width_oracle": max(width_need),
                     "mean_width_orig": sum(width_orig) / len(width_orig), "mean_width_oracle": sum(width_need) / len(width_need),
                     "width_orig": width_orig, "width_oracle": width_need}
    # where in the run does the width collapse? event index of last prune vs total events
    last_prune_ei = max(ei for ei, st in enumerate(states) if st["kind"] == "rough")
    out["last_prune_progress"] = last_prune_ei / (len(states) - 1)
    out["flops_after_last_prune"] = sum(f for (ei, j), f in node_flops.items() if ei > last_prune_ei) / total
    out["total_flops"] = total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {json.loads(l)["problem_idx"]: json.loads(l) for l in open(os.path.join(args.out, "res.jsonl"))}
    per = []
    for p in sorted(glob.glob(os.path.join(args.out, "traces", "p*.jsonl"))):
        pid = int(os.path.basename(p)[1:6])
        if os.path.exists(p.replace(".jsonl", ".done")):
            per.append(analyze(p, res[pid]))
    T = sum(d["total_flops"] for d in per)
    wavg = lambda key, sub: sum(d[sub][key] * d["total_flops"] for d in per) / T
    summary = {"n_problems": len(per)}
    for sub in ("search", "top1"):
        summary[sub] = {
            "W_commit": wavg("W_commit", sub),
            "W_commit_pre_evidence": wavg("W_commit_pre_evidence", sub),
            "peak_width_orig": max(d[sub]["peak_width_orig"] for d in per),
            "peak_width_oracle": max(d[sub]["peak_width_oracle"] for d in per),
            "mean_width_orig": sum(d[sub]["mean_width_orig"] for d in per) / len(per),
            "mean_width_oracle": sum(d[sub]["mean_width_oracle"] for d in per) / len(per),
            "per_problem_W_commit": [round(d[sub]["W_commit"], 3) for d in per],
        }
    summary["last_prune_progress_mean"] = sum(d["last_prune_progress"] for d in per) / len(per)
    summary["flops_after_last_prune"] = sum(d["flops_after_last_prune"] * d["total_flops"] for d in per) / T
    summary["width_timeline_p0"] = {"orig": per[0]["search"]["width_orig"][:40], "oracle_search": per[0]["search"]["width_oracle"][:40],
                                    "oracle_top1": per[0]["top1"]["width_oracle"][:40]}
    with open(os.path.join(args.out, "deferred_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
