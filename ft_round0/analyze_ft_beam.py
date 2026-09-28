"""Survival-aware protection ceiling for FastTTS beam-search traces (see FT_PREREG.md)."""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

KV_1P5B = 28 * 2 * 128 * 2 * 2   # Qwen2.5-Math-1.5B: 28 KiB / token
KV_7B = 28 * 4 * 128 * 2 * 2     # Qwen2.5-Math-7B:   56 KiB / token


def analyze(path):
    recs = [json.loads(l) for l in open(path)]
    prob = [r for r in recs if r["t"] == "problem"][0]
    iters = [r for r in recs if r["t"] == "iter"]
    end = [r for r in recs if r["t"] == "end"][0]
    T = len(iters)
    dt = [r["gen_s"] + r["ver_s"] for r in iters]

    node = {}          # (i, uid) -> record
    parent = {}
    children = defaultdict(list)
    checks = {"tree_inconsistent": 0, "lineage_token_mismatch": 0, "active_count_bad": 0, "kept_count_bad": 0}
    for it in iters:
        i = it["i"]
        if len(it["rows"]) != prob["n"]:
            checks["active_count_bad"] += 1
        kept = sum(1 for r in it["rows"] if not r["pruned"] and not r["completed"])
        noncompleted = sum(1 for r in it["rows"] if not r["completed"])
        if kept != min(prob["n"] // prob["beam_width"], noncompleted) and i < T - 1:
            checks["kept_count_bad"] += 1
        for r in it["rows"]:
            n = (i, r["uid"])
            node[n] = r
            if i == 0:
                parent[n] = None
            else:
                p = (i - 1, r["src_uid"])
                if p not in node:
                    checks["tree_inconsistent"] += 1
                    parent[n] = None
                else:
                    parent[n] = p
                    children[p].append(n)

    # death iteration of each node = max iteration in its subtree
    death = {}
    for n in sorted(node, key=lambda x: -x[0]):
        d = n[0]
        for c in children[n]:
            d = max(d, death[c])
        death[n] = d

    completed_nodes = [(it["i"], r["uid"]) for it in iters for r in it["rows"] if r["completed"]]
    pred_node = [n for n in completed_nodes if n[1] == end["pred_uid"]]

    def ancestors(roots):
        s = set()
        for n in roots:
            while n is not None and n not in s:
                s.add(n); n = parent[n]
        return s

    strict = ancestors(completed_nodes)
    answer = ancestors(pred_node)

    # falsification 3: lineage tokens of each completed beam == its total tokens
    for n in completed_nodes:
        tot, m = 0, n
        while m is not None:
            tot += node[m]["step_tokens"]; m = parent[m]
        if tot != node[n]["total_tokens"]:
            checks["lineage_token_mismatch"] += 1

    def bytetime(pred, unit):
        s = 0.0
        for n, r in node.items():
            if not pred(n):
                continue
            i, d = n[0], death[n]
            s += r["step_tokens"] * ((d - i + 1) if unit == "iter" else sum(dt[i:d + 1]))
        return s

    out = {"pid": prob["pid"], "T": T, "prompt_tokens": prob["prompt_tokens"], "n_nodes": len(node),
           "n_completed": len(completed_nodes), "wall_s": sum(dt), "checks": checks}
    for unit in ("iter", "sec"):
        prompt_bt = prob["prompt_tokens"] * (T if unit == "iter" else sum(dt))
        u = bytetime(lambda n: True, unit) + prompt_bt
        s_ = bytetime(lambda n: n in strict, unit) + prompt_bt
        a_ = bytetime(lambda n: n in answer, unit) + prompt_bt
        out[unit] = {"uniform": u, "oracle_strict": s_, "oracle_answer": a_,
                     "W_strict": 1 - s_ / u, "W_answer": 1 - a_ / u}
        # late share and recompute cost over unprotected (strict) nodes
        late = 0.0; tot = 0.0
        for n, r in node.items():
            if n in strict:
                continue
            i, d = n[0], death[n]
            bt = r["step_tokens"] * ((d - i + 1) if unit == "iter" else sum(dt[i:d + 1]))
            tot += bt
            if d / max(1, T - 1) >= 0.5:
                late += bt
        out[unit]["late_share"] = late / tot if tot else 0.0
        out[unit]["unprotected_bt"] = tot
    # recompute cost: for each dying leaf lineage (unprotected node with no unprotected children
    # continuing), unique tokens below nearest protected ancestor, and its context size
    rec = []
    for n, r in node.items():
        if n in strict or any(c not in strict for c in children[n]):
            continue
        uniq, m = 0, n
        while m is not None and m not in strict:
            uniq += node[m]["step_tokens"]; m = parent[m]
        ctx = prob["prompt_tokens"] + r["total_tokens"]
        rec.append({"death_iter": death[n], "progress": death[n] / max(1, T - 1), "unique_tokens": uniq,
                    "context_tokens": ctx, "share": uniq / ctx, "bt": r["step_tokens"] * (death[n] - n[0] + 1)})
    out["recompute"] = rec
    # descriptive online probe: kept beam's lineage kept again next iteration
    kept_next = [0, 0]
    for it in iters[:-1]:
        i = it["i"]
        for r in it["rows"]:
            if r["pruned"] or r["completed"]:
                continue
            n = (i, r["uid"])
            kids = children[n]
            if not kids:
                continue
            kept_next[1] += 1
            kept_next[0] += int(any(not node[c]["pruned"] and not node[c]["completed"] for c in kids))
    out["kept_lineage_survives_next"] = kept_next
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="aime24")
    args = ap.parse_args()
    per = []
    for p in sorted(glob.glob(os.path.join(args.out, "traces", f"{args.dataset}_p*.jsonl"))):
        if os.path.exists(p.replace(".jsonl", ".traced.done")):
            per.append(analyze(p))
    if not per:
        print("no completed problems"); return
    summary = {"n_problems": len(per), "T_mean": st.mean(d["T"] for d in per), "T_list": [d["T"] for d in per]}
    for unit in ("iter", "sec"):
        U = sum(d[unit]["uniform"] for d in per); S = sum(d[unit]["oracle_strict"] for d in per); A = sum(d[unit]["oracle_answer"] for d in per)
        ub = sum(d[unit]["unprotected_bt"] for d in per)
        summary[unit] = {"W_protect_strict": 1 - S / U, "W_answer": 1 - A / U,
                         "late_share": sum(d[unit]["late_share"] * d[unit]["unprotected_bt"] for d in per) / ub if ub else 0.0,
                         "per_problem_W_strict": [round(d[unit]["W_strict"], 3) for d in per]}
    rec = [r for d in per for r in d["recompute"]]
    btw = sum(r["bt"] for r in rec)
    summary["recompute"] = {"n_dying_lineages": len(rec),
                            "median_unique_tokens": st.median(r["unique_tokens"] for r in rec) if rec else None,
                            "bt_weighted_mean_share": sum(r["share"] * r["bt"] for r in rec) / btw if btw else None,
                            "median_context_tokens": st.median(r["context_tokens"] for r in rec) if rec else None,
                            "death_progress_hist": {f"{k/4:.2f}-{(k+1)/4:.2f}": sum(1 for r in rec if k / 4 <= r["progress"] < (k + 1) / 4 + (1e-9 if k == 3 else 0)) for k in range(4)}}
    summary["kv_GiB_uniform_bytetime_iter_1p5B"] = sum(d["iter"]["uniform"] for d in per) * KV_1P5B / 2**30
    summary["kv_bytes_per_token"] = {"1.5B": KV_1P5B, "7B": KV_7B}
    kn = [sum(d["kept_lineage_survives_next"][0] for d in per), sum(d["kept_lineage_survives_next"][1] for d in per)]
    summary["kept_lineage_survives_next"] = {"hits": kn[0], "n": kn[1], "rate": kn[0] / kn[1] if kn[1] else None}
    summary["checks"] = {k: sum(d["checks"][k] for d in per) for k in per[0]["checks"]}
    summary["wall_min_per_problem"] = [round(d["wall_s"] / 60, 1) for d in per]
    with open(os.path.join(args.out, f"ft_beam_summary_{args.dataset}.json"), "w") as f:
        json.dump({"summary": summary, "per_problem": per}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
