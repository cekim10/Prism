"""Run FastTTS beam search (repo defaults, vanilla semantics) on AIME 2024 / MATH-500 with the
traced beam search, writing per-problem traces. Resumable per problem.

FASTTTS_DIR must point at a checkout of ihc-fan-lab/FastTTS (commit 1c01d54)."""
import argparse
import json
import os
import sys
import time

FASTTTS_DIR = os.environ.get("FASTTTS_DIR", os.path.expanduser("~/FastTTS"))
sys.path.insert(0, FASTTTS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_problems(name, start, n):
    from datasets import load_dataset
    if name == "aime24":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        rows = [{"id": str(r["id"]), "problem": r["problem"], "answer": str(r["answer"])} for r in ds]
    elif name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [{"id": str(r["unique_id"]), "problem": r["problem"], "answer": str(r["answer"])} for r in ds]
    else:
        raise ValueError(name)
    return rows[start:start + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["aime24", "math500"], default="aime24")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["traced", "orig"], default="traced")
    ap.add_argument("--num_iterations", type=int, default=None)
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    args = ap.parse_args()

    import fasttts as ft_mod
    from fasttts import create_fasttts
    from config import SearchConfig
    if args.mode == "traced":
        import traced_beam_search as tbs
        ft_mod.beam_search = tbs.beam_search

    os.makedirs(os.path.join(args.out, "traces"), exist_ok=True)
    problems = load_problems(args.dataset, args.start, args.n)

    fasttts = create_fasttts(
        generator_vllm_config={"gpu_memory_utilization": args.gpu_mem},
        verifier_vllm_config={"gpu_memory_utilization": args.gpu_mem},
        approach="beam_search",
        offload_enabled=False, spec_beam_extension=False, prefix_aware_scheduling=False,
    )
    cfg_kwargs = {}
    if args.num_iterations:
        cfg_kwargs["num_iterations"] = args.num_iterations
    search_config = SearchConfig(approach="beam_search", **cfg_kwargs)
    fasttts.initialize()
    res_path = os.path.join(args.out, f"res_{args.mode}.jsonl")
    try:
        for k, row in enumerate(problems):
            idx = args.start + k
            tag = f"{args.dataset}_p{idx:04d}"
            done = os.path.join(args.out, "traces", f"{tag}.{args.mode}.done")
            if os.path.exists(done):
                print(f"skip {tag}", flush=True)
                continue
            if args.mode == "traced":
                tbs.TRACER = tbs.Tracer(os.path.join(args.out, "traces", f"{tag}.jsonl"), pid=tag)
            t0 = time.time()
            results = fasttts.search([row["problem"]], search_config=search_config)
            dt = time.time() - t0
            if args.mode == "traced":
                tbs.TRACER.close()
            out = {"tag": tag, "idx": idx, "id": row["id"], "answer": row["answer"], "pred": results["pred"][0],
                   "completions": results["completions"][0], "scores": results["scores"][0],
                   "effective_num_tokens": results["effective_num_tokens"][0],
                   "total_num_tokens": results["total_num_tokens"], "gen_s": results["total_generator_latency_s"],
                   "ver_s": results["total_verifier_latency_s"], "wall_s": dt, "mode": args.mode}
            with open(res_path, "a") as f:
                f.write(json.dumps(out) + "\n")
            with open(done, "w") as f:
                f.write(json.dumps({"wall_s": dt}))
            print(f"{tag}: {dt/60:.1f} min, {len(results['completions'][0])} completions, "
                  f"tokens={results['total_num_tokens']}, gen={results['total_generator_latency_s']:.0f}s ver={results['total_verifier_latency_s']:.0f}s",
                  flush=True)
    finally:
        fasttts.shutdown()


if __name__ == "__main__":
    main()
