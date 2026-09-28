"""Offline analysis of Round 0 traces (see ROUND0_PREREG.md for definitions)."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_trace(path):
    recs = [json.loads(l) for l in open(path)]
    return recs


def analyze_problem(recs, npz_path=None):
    """Returns per-problem accounting dict."""
    # pair state records with their fwd records (same seq)
    fwd_by_seq = {r["seq"]: r for r in recs if r["t"] == "fwd"}
    states = [r for r in recs if r["t"] == "state"]
    prunes = [r for r in recs if r["t"] == "prune"]
    verifiers = [r for r in recs if r["t"] == "verifier"]
    misc = [r for r in recs if r["t"] == "misc"]
    end = [r for r in recs if r["t"] == "end"][0]
    prob = [r for r in recs if r["t"] == "problem"][0]

    out = {"pid": prob["pid"], "prompt_len": prob["prompt_len"], "total_len": prob["total_len"],
           "n_state_events": len(states), "checks": {"hash_mismatch_state_vs_fwd": 0, "tree_inconsistent": 0,
                                                     "dup_logits_checked": 0, "dup_logits_equal": 0, "dup_logits_maxdiff": 0.0,
                                                     "canvas_collisions": 0, "canvas_pairs_checked": 0}}

    # ---- tree + never-diverged classes -----------------------------------------------------
    # class id per row, refined at each event: class_t(row) = canon(class_{t-1}(parent), h_full_t(row))
    prune_iter = iter(prunes)
    pending_prune = None
    prev_rows = None          # list of dicts for rows of previous state event (with 'cls', 'node')
    node_counter = 0
    canon = {}
    events = []               # per state event: dict(kind, block, step, rows=[{h_fwd,h_full,cls,node,parent}], flops_row, lat_row)
    for st in states:
        fw = fwd_by_seq[st["seq"]]
        assert fw["n"] == len(st["rows"])
        rows = []
        # decide parent mapping
        if st["kind"] == "denoise" and pending_prune is not None:
            bmap = {b["new"]: b for b in pending_prune["branch"]}
            if sorted(bmap) != list(range(len(st["rows"]))):
                out["checks"]["tree_inconsistent"] += 1
            parent_of = lambda j: prev_rows[bmap[j]["src"]]
            born_perturbed = lambda j: len(bmap[j]["perturbed"]) > 0
            pending_prune = None
        else:
            parent_of = (lambda j: prev_rows[j]) if prev_rows is not None else (lambda j: None)
            born_perturbed = lambda j: False
        for j, r in enumerate(st["rows"]):
            if fw["hashes"][j] != r["h_fwd"]:
                out["checks"]["hash_mismatch_state_vs_fwd"] += 1
            par = parent_of(j)
            pcls = par["cls"] if par is not None else -1
            key = (pcls, r["h_full"])
            if key not in canon:
                canon[key] = len(canon)
            rows.append({"row": j, "h_fwd": r["h_fwd"], "h_full": r["h_full"], "cls": canon[key],
                         "node": node_counter, "parent": par["node"] if par is not None else None,
                         "nm_tot": r["nm_tot"], "nm_blk": r["nm_blk"], "born_perturbed": born_perturbed(j)})
            node_counter += 1
        ev = {"kind": st["kind"], "block": st["block"], "step": st["step"], "seq": st["seq"], "rows": rows,
              "flops_row": fw["flops_row"], "lat_row": fw["lat"] / fw["n"], "L": fw["L"]}
        events.append(ev)
        for d in fw["dup_checks"]:
            out["checks"]["dup_logits_checked"] += 1
            out["checks"]["dup_logits_equal"] += int(d["logits_equal"])
            out["checks"]["dup_logits_maxdiff"] = max(out["checks"]["dup_logits_maxdiff"], d["max_abs_diff"])
        if st["kind"] == "rough":
            pending_prune = next(prune_iter)
        prev_rows = rows

    node_parent = {r["node"]: r["parent"] for ev in events for r in ev["rows"]}
    node_event = {r["node"]: ei for ei, ev in enumerate(events) for r in ev["rows"]}

    def lca_event(u, v):
        anc = set()
        while u is not None:
            anc.add(u); u = node_parent[u]
        while v is not None and v not in anc:
            v = node_parent[v]
        return node_event[v] if v is not None else -1

    # idle rows: block already fully decoded but still forwarded (Prism inefficiency, same-path)
    idle_rows = sum(1 for ev in events if ev["kind"] == "denoise" for r in ev["rows"] if r["nm_blk"] == 0)
    idle_flops = sum(ev["flops_row"] for ev in events if ev["kind"] == "denoise" for r in ev["rows"] if r["nm_blk"] == 0)

    # ---- per-event grouping ------------------------------------------------------------------
    total_flops = 0.0
    total_lat = 0.0
    n_exec = 0
    n_unique = 0
    c1_flops = c2_flops = 0.0
    c1_lat = c2_lat = 0.0
    c1_rows = c2_rows = 0
    c2_groups = []
    seen_hfwd_first_event = {}
    cross_time_flops = 0.0
    cross_time_rows = 0
    c3_flops = 0.0
    c3_rows = 0
    event_flops = []
    for ei, ev in enumerate(events):
        n = len(ev["rows"])
        ef = n * ev["flops_row"]
        total_flops += ef
        total_lat += n * ev["lat_row"]
        event_flops.append(ef)
        n_exec += n
        groups = defaultdict(list)
        for r in ev["rows"]:
            groups[r["h_fwd"]].append(r)
        n_unique += len(groups)
        for h, g in groups.items():
            if len(g) > 1:
                k = len(set(r["cls"] for r in g))
                c1_rows += len(g) - k
                c2_rows += k - 1
                c1_flops += (len(g) - k) * ev["flops_row"]
                c2_flops += (k - 1) * ev["flops_row"]
                c1_lat += (len(g) - k) * ev["lat_row"]
                c2_lat += (k - 1) * ev["lat_row"]
                if k > 1:
                    reps = {}
                    for r in g:
                        reps.setdefault(r["cls"], r["node"])
                    rep_nodes = list(reps.values())
                    dists, lca_kinds = [], []
                    for a in range(len(rep_nodes)):
                        for b in range(a + 1, len(rep_nodes)):
                            le = lca_event(rep_nodes[a], rep_nodes[b])
                            dists.append(ei - le)
                            lca_kinds.append(events[le]["kind"] if le >= 0 else "root")
                    c2_groups.append({"event_idx": ei, "kind": ev["kind"], "block": ev["block"], "step": ev["step"],
                                      "n": len(g), "k": k, "h_full_equal": len(set(r["h_full"] for r in g)) == 1,
                                      "nm_tot": g[0]["nm_tot"], "nodes": [r["node"] for r in g],
                                      "born_perturbed": [r["born_perturbed"] for r in g],
                                      "events_since_lca": dists, "lca_kind": lca_kinds})
            key = (h, ev["kind"])
            if h in seen_hfwd_first_event and seen_hfwd_first_event[h] != (ev["block"], ev["step"]):
                cross_time_flops += len(g) * ev["flops_row"]
                cross_time_rows += len(g)
            seen_hfwd_first_event.setdefault(h, (ev["block"], ev["step"]))
        # C3: denoise right after rough at same (block, step): unperturbed copies re-run the same canvas
        if ev["kind"] == "denoise" and ei > 0 and events[ei - 1]["kind"] == "rough" \
                and events[ei - 1]["block"] == ev["block"] and events[ei - 1]["step"] == ev["step"]:
            rough_h = set(r["h_fwd"] for r in events[ei - 1]["rows"])
            for r in ev["rows"]:
                if r["h_fwd"] in rough_h:
                    c3_rows += 1
                    c3_flops += ev["flops_row"]

    # verifier forwards (exact key = text)
    ver_fwds = [r for r in recs if r["t"] == "fwd" and r["kind"] in ("verifier", "verifier_final")]
    ver_flops = sum(r["flops_row"] * r["n"] for r in ver_fwds)
    ver_lat = sum(r["lat"] for r in ver_fwds)
    ver_dup_flops = 0.0
    by_event = defaultdict(list)
    for v in verifiers:
        by_event[(v["block"], v["step"])].append(v)
    ver_fwd_by_key = defaultdict(list)
    for r in ver_fwds:
        ver_fwd_by_key[(r["block"], r["step"], r["row"])].append(r)
    for (b, s), vs in by_event.items():
        seen = set()
        for v in vs:
            if v["h_text"] in seen:
                for r in ver_fwd_by_key[(b, s, v["row"])]:
                    ver_dup_flops += r["flops_row"] * r["n"]
            seen.add(v["h_text"])

    grand_total_flops = total_flops + ver_flops
    grand_total_lat = total_lat + ver_lat

    # progress bins (by executed state-event ordinal) and remaining compute after each C2 merge
    cum = np.cumsum(event_flops)
    for g in c2_groups:
        ei = g["event_idx"]
        g["progress"] = ei / max(1, len(events) - 1)
        g["remaining_frac_after"] = float((total_flops - cum[ei]) / total_flops) if total_flops else 0.0
        g["flops_saved"] = (g["k"] - 1) * events[ei]["flops_row"]
    bins = {"0-25": 0.0, "25-50": 0.0, "50-75": 0.0, "75-100": 0.0}
    for g in c2_groups:
        p = g["progress"]
        b = "0-25" if p < 0.25 else "25-50" if p < 0.5 else "50-75" if p < 0.75 else "75-100"
        bins[b] += g["flops_saved"]

    # divergence dynamics: for duplicate groups at event t, are the children still equal at t+1?
    still_equal = {"C1": [0, 0], "C2": [0, 0]}
    for ei in range(len(events) - 1):
        ev, nx = events[ei], events[ei + 1]
        children = defaultdict(list)
        for r in nx["rows"]:
            children[r["parent"]].append(r)
        groups = defaultdict(list)
        for r in ev["rows"]:
            groups[r["h_fwd"]].append(r)
        for g in groups.values():
            if len(g) < 2:
                continue
            k = len(set(r["cls"] for r in g))
            cat = "C2" if k > 1 else "C1"
            # take one child per row (first child, e.g. unperturbed copy) if present
            ch = [children[r["node"]][0] for r in g if children[r["node"]]]
            if len(ch) >= 2:
                still_equal[cat][1] += 1
                still_equal[cat][0] += int(len(set(c["h_fwd"] for c in ch)) == 1)

    # canvas collision check for equal-hash pairs (hash = identity)
    if npz_path and os.path.exists(npz_path):
        z = np.load(npz_path)
        canv, index = z["canvases"], z["index"]
        pos = {(int(s), int(b)): i for i, (s, b) in enumerate(index)}
        byh = defaultdict(list)
        for ev in events:
            for r in ev["rows"]:
                byh[r["h_fwd"]].append(pos[(ev["seq"], r["row"])])
        for h, idxs in byh.items():
            for i in idxs[1:]:
                out["checks"]["canvas_pairs_checked"] += 1
                if not np.array_equal(canv[idxs[0]], canv[i]):
                    out["checks"]["canvas_collisions"] += 1

    # sampler determinism stat: share of transferred tokens equal to noise-free argmax
    n_rows_tr = sum(len(m["argmax_equal_sample"]) for m in misc if m.get("kind") == "transfer")
    n_rows_argmax = sum(sum(m["argmax_equal_sample"]) for m in misc if m.get("kind") == "transfer")

    out.update({
        "n_exec_rows": n_exec, "n_unique_states": n_unique,
        "R_state": 1 - n_unique / n_exec if n_exec else 0.0,
        "flops_denoise_rough": total_flops, "flops_verifier": ver_flops, "flops_total": grand_total_flops,
        "lat_denoise_rough": total_lat, "lat_verifier": ver_lat, "lat_total": grand_total_lat,
        "c1_flops": c1_flops, "c2_flops": c2_flops, "c1_lat": c1_lat, "c2_lat": c2_lat,
        "c1_rows": c1_rows, "c2_rows": c2_rows, "c3_rows": c3_rows, "c3_flops": c3_flops,
        "cross_time_rows": cross_time_rows, "cross_time_flops": cross_time_flops,
        "idle_rows": idle_rows, "idle_flops": idle_flops,
        "ver_dup_flops": ver_dup_flops,
        "S_C1": c1_flops / grand_total_flops, "S_C2": c2_flops / grand_total_flops,
        "S_C1_lat": c1_lat / grand_total_lat if grand_total_lat else 0.0, "S_C2_lat": c2_lat / grand_total_lat if grand_total_lat else 0.0,
        "c2_groups": c2_groups, "c2_bins_flops": bins,
        "still_equal_next": still_equal,
        "argmax_rows": n_rows_argmax, "transfer_rows": n_rows_tr,
        "nfe": end["stats"]["nfe"], "svf_calls": end["stats"]["svf_calls"],
        "steps_per_block": end["stats"]["steps_per_block"],
        "wall_s": end["time"] - prob["time"],
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.out, "traces", "p*.jsonl")))
    per = []
    for p in paths:
        if not os.path.exists(p.replace(".jsonl", ".done")):
            continue
        per.append(analyze_problem(load_trace(p), p.replace(".jsonl", "_canvases.npz")))
    if not per:
        print("no completed problems")
        return
    tot = lambda k: sum(d[k] for d in per)
    T = tot("flops_total")
    TL = tot("lat_total")
    summary = {
        "n_problems": len(per),
        "n_exec_rows": tot("n_exec_rows"), "n_unique_states": tot("n_unique_states"),
        "R_state": 1 - tot("n_unique_states") / tot("n_exec_rows"),
        "R_compute_denoise_rough": (tot("c1_flops") + tot("c2_flops")) / tot("flops_denoise_rough"),
        "S_oracle_C1": tot("c1_flops") / T, "S_oracle_C2": tot("c2_flops") / T,
        "S_oracle_C1_latency": tot("c1_lat") / TL, "S_oracle_C2_latency": tot("c2_lat") / TL,
        "S_C3_same_row_recompute": tot("c3_flops") / T,
        "S_cross_time_memo": tot("cross_time_flops") / T,
        "S_verifier_text_dup": tot("ver_dup_flops") / T,
        "S_idle_finished_rows": tot("idle_flops") / T,
        "c2_pairs_events_since_lca_hist": dict(sorted(__import__("collections").Counter(
            d_ for d in per for g in d["c2_groups"] for d_ in g["events_since_lca"]).items())),
        "c2_pairs_lca_kind": dict(__import__("collections").Counter(
            k_ for d in per for g in d["c2_groups"] for k_ in g["lca_kind"])),
        "flops_share_verifier": tot("flops_verifier") / T,
        "c1_rows": tot("c1_rows"), "c2_rows": tot("c2_rows"), "c3_rows": tot("c3_rows"),
        "n_c2_groups": sum(len(d["c2_groups"]) for d in per),
        "c2_groups_h_full_equal": sum(g["h_full_equal"] for d in per for g in d["c2_groups"]),
        "c2_bins_flops_share": {k: sum(d["c2_bins_flops"][k] for d in per) / T for k in ["0-25", "25-50", "50-75", "75-100"]},
        "c2_remaining_frac_after_merge": [g["remaining_frac_after"] for d in per for g in d["c2_groups"]],
        "still_equal_next": {c: [sum(d["still_equal_next"][c][0] for d in per), sum(d["still_equal_next"][c][1] for d in per)] for c in ["C1", "C2"]},
        "argmax_share_of_transfer_rows": tot("argmax_rows") / max(1, tot("transfer_rows")),
        "checks": {k: (max if k == "dup_logits_maxdiff" else sum)(d["checks"][k] for d in per) for k in per[0]["checks"]},
        "wall_min_per_problem": [round(d["wall_s"] / 60, 1) for d in per],
        "nfe": [d["nfe"] for d in per], "svf_calls": [d["svf_calls"] for d in per],
        "per_problem_S_C2": [round(d["S_C2"], 4) for d in per],
        "per_problem_S_C1": [round(d["S_C1"], 4) for d in per],
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "per_problem": per}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
