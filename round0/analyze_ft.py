"""Survival-aware protection ceiling from the Round 0 Prism traces (no GPU).

Prism/LLaDA runs with use_cache=False: the physical per-branch state is a ~3 KB canvas, so the
trace provides only the *survival timeline* of branches. Byte costs below are modelled:
  const : every branch holds a fixed-size state (cached full canvas)      -> byte-time == row-steps
  ar    : branch state grows with decoded tokens (AR-style KV)             -> weighted by decoded tokens
Protection policies compared on protected byte-time (state x steps under protection) and on
write traffic (newly created state bytes that a replica must receive):
  uniform            protect every live branch every step
  oracle_next_prune  protect a branch only if it survives the next prune event
  oracle_strict      protect only ancestors of the final_K rows the search returns
  oracle_answer      protect only ancestors of the rank-1 (returned) final row
"""
import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict

KV_BYTES_PER_TOKEN = 32 * 2 * 4096 * 2  # 8B-class model, bf16, no GQA: 512 KiB


def build(path, res_row):
    recs = [json.loads(l) for l in open(path)]
    fwd_by_seq = {r["seq"]: r for r in recs if r["t"] == "fwd"}
    states = [r for r in recs if r["t"] == "state"]
    prunes = [r for r in recs if r["t"] == "prune"]
    prob = [r for r in recs if r["t"] == "problem"][0]
    vrec = [r for r in recs if r["t"] == "verifier"]
    P, T = prob["prompt_len"], prob["total_len"]

    parent, decoded, kind = {}, {}, {}
    pit = iter(prunes)
    pending = None
    prune_events = []
    for ei, st in enumerate(states):
        if st["kind"] == "denoise" and pending is not None:
            bmap = {b["new"]: b["src"] for b in pending["branch"]}
            par = lambda j: (ei - 1, bmap[j])
            pending = None
        else:
            par = (lambda j: (ei - 1, j)) if ei > 0 else (lambda j: None)
        for j, r in enumerate(st["rows"]):
            parent[(ei, j)] = par(j)
            decoded[(ei, j)] = (T - P) - r["nm_tot"]
            kind[(ei, j)] = st["kind"]
        if st["kind"] == "rough":
            pending = next(pit)
            prune_events.append((ei, pending))
    last = len(states) - 1
    finals = [(last, j) for j in range(len(states[last]["rows"]))]
    h1 = hashlib.blake2b(res_row["all_trajectories"][0]["resp"].encode(), digest_size=16).hexdigest()
    r1 = [v["row"] for v in vrec if v["block"] is None and v["h_text"] == h1]
    answer_root = [(last, r1[0])] if r1 else finals[:1]

    def ancestors(roots):
        s = set()
        for n in roots:
            while n is not None and n not in s:
                s.add(n); n = parent[n]
        return s

    strict = ancestors(finals)
    answer = ancestors(answer_root)
    # next-prune oracle: node protected iff it has a descendant at the next rough event's selection,
    # i.e. it is an ancestor of a selected row at the next prune (or of a final row if no prune follows)
    next_sel = set()
    for pi, (ei, pe) in enumerate(prune_events):
        sel_nodes = [(ei, j) for j in pe["selected"]]
        anc = ancestors(sel_nodes)
        lo = prune_events[pi - 1][0] if pi > 0 else -1
        next_sel |= {n for n in anc if lo < n[0] <= ei}
    last_prune_ei = prune_events[-1][0] if prune_events else -1
    next_sel |= {n for n in strict if n[0] > last_prune_ei}

    policies = {"uniform": lambda n: True, "oracle_next_prune": lambda n: n in next_sel,
                "oracle_strict": lambda n: n in strict, "oracle_answer": lambda n: n in answer}
    size = {"const": lambda n: T * KV_BYTES_PER_TOKEN, "ar": lambda n: (P + decoded[n]) * KV_BYTES_PER_TOKEN}
    out = {}
    for sm, sf in size.items():
        for pn, pf in policies.items():
            bt = 0.0; wr = 0.0
            for n in parent:
                if kind[n] != "denoise":
                    continue
                if pf(n):
                    bt += sf(n)
                    par = parent[n]
                    new_tokens = decoded[n] - (decoded[par] if par is not None else 0)
                    # prefix KV is shared by all branches (written once per problem); forks are copy-on-write,
                    # so a replica only receives the tokens this branch decoded itself
                    wr += max(new_tokens, 0) * KV_BYTES_PER_TOKEN
            wr += P * KV_BYTES_PER_TOKEN
            out[(sm, pn)] = {"byte_time": bt, "write": wr}
    # online-predictability probes at prune events
    probes = {"prune2_selected_unperturbed": [0, 0], "prune2_selected_from_rank1_parent": [0, 0],
              "answer_from_top_scored_survivor": [0, 0]}
    if len(prune_events) >= 2:
        (e1, p1), (e2, p2) = prune_events[0], prune_events[1]
        order1 = sorted(range(len(p1["scores"])), key=lambda i: -p1["scores"][i])
        bmap = {b["new"]: b for b in p1["branch"]}
        for j in p2["selected"]:
            b = bmap[j]
            probes["prune2_selected_unperturbed"][0] += int(len(b["perturbed"]) == 0)
            probes["prune2_selected_unperturbed"][1] += 1
            probes["prune2_selected_from_rank1_parent"][0] += int(b["src"] == order1[0])
            probes["prune2_selected_from_rank1_parent"][1] += 1
        # final answer lineage: which step-8 survivor does the rank-1 final row descend from?
        order2 = sorted(range(len(p2["scores"])), key=lambda i: -p2["scores"][i])
        n = answer_root[0]
        while n is not None and n[0] > e2:
            n = parent[n]
        probes["answer_from_top_scored_survivor"][0] += int(n is not None and n[1] == order2[0])
        probes["answer_from_top_scored_survivor"][1] += 1
    return out, probes, {"P": P, "T": T, "steps": sum(1 for s in states if s["kind"] == "denoise")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {json.loads(l)["problem_idx"]: json.loads(l) for l in open(os.path.join(args.out, "res.jsonl"))}
    agg = defaultdict(lambda: {"byte_time": 0.0, "write": 0.0})
    probes = defaultdict(lambda: [0, 0])
    n = 0
    for p in sorted(glob.glob(os.path.join(args.out, "traces", "p*.jsonl"))):
        if not os.path.exists(p.replace(".jsonl", ".done")):
            continue
        pid = int(os.path.basename(p)[1:6])
        out, pr, meta = build(p, res[pid])
        for k, v in out.items():
            agg[k]["byte_time"] += v["byte_time"]; agg[k]["write"] += v["write"]
        for k, v in pr.items():
            probes[k][0] += v[0]; probes[k][1] += v[1]
        n += 1
    summary = {"n_problems": n, "kv_bytes_per_token": KV_BYTES_PER_TOKEN, "policies": {}}
    for sm in ("const", "ar"):
        u = agg[(sm, "uniform")]
        for pn in ("uniform", "oracle_next_prune", "oracle_strict", "oracle_answer"):
            a = agg[(sm, pn)]
            summary["policies"][f"{sm}/{pn}"] = {
                "byte_time_GBs": a["byte_time"] / 2**30, "write_GB": a["write"] / 2**30,
                "byte_time_saving_vs_uniform": 1 - a["byte_time"] / u["byte_time"],
                "write_saving_vs_uniform": 1 - a["write"] / u["write"] if u["write"] else 0.0}
    summary["probes"] = {k: {"hits": v[0], "n": v[1], "rate": v[0] / v[1] if v[1] else None} for k, v in probes.items()}
    with open(os.path.join(args.out, "ft_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
