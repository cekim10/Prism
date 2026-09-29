"""Falsification test 1: compare res_orig.jsonl and res_traced.jsonl for the same problems."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    load = lambda m: {json.loads(l)["tag"]: json.loads(l) for l in open(os.path.join(args.out, f"res_{m}.jsonl"))}
    o, t = load("orig"), load("traced")
    common = sorted(set(o) & set(t))
    ok_all = True
    for tag in common:
        a, b = o[tag], t[tag]
        flat = lambda s: [v for beam in s for step in beam for v in (step if isinstance(step, list) else [step])]
        sa, sb = flat(a["scores"]), flat(b["scores"])
        max_score_diff = max((abs(x - y) for x, y in zip(sa, sb)), default=0.0) if len(sa) == len(sb) else float("inf")
        checks = {"pred": a["pred"] == b["pred"], "completions": a["completions"] == b["completions"],
                  "scores_exact": a["scores"] == b["scores"], "scores_within_1e-3": max_score_diff <= 1e-3,
                  "tokens": a["effective_num_tokens"] == b["effective_num_tokens"],
                  "n_completions": len(a["completions"]) == len(b["completions"])}
        ok = all(v for k, v in checks.items() if k != "scores_exact"); ok_all &= ok
        print(tag, "PASS" if ok else "FAIL", checks, f"max|score diff|={max_score_diff:.2e}")
    print("ALL PASS" if ok_all and common else "SOME FAIL" if common else "no common problems")


if __name__ == "__main__":
    main()
