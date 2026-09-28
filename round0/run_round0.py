"""Round 0 driver: runs Prism (official GSM8K config) on GSM8K test problems with the traced
sampler, writing the official res.jsonl format plus per-problem traces. Resumable per problem."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prism_loader as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["traced", "orig"], default="traced")
    ap.add_argument("--gen_length", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--no_quant", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    os.makedirs(args.out, exist_ok=True)
    trace_dir = os.path.join(args.out, "traces")
    os.makedirs(trace_dir, exist_ok=True)

    model, tok = L.load_model_and_tokenizer(args.device)
    if not args.no_quant:
        from quant import quantize_model_inplace
        print("quantized linears:", quantize_model_inplace(model, args.device), flush=True)
    hts_mod, _ = L.load_prism_modules()
    from traced_sampler import TracedHTSSampler, Tracer

    ds = L.load_gsm8k_test()
    kwargs = dict(L.OFFICIAL_GSM8K_GEN_KWARGS)
    if args.gen_length: kwargs["gen_length"] = args.gen_length
    if args.steps: kwargs["steps"] = args.steps

    res_path = os.path.join(args.out, "res.jsonl")
    for idx in range(args.start, args.start + args.n):
        done_marker = os.path.join(trace_dir, f"p{idx:05d}.done")
        if os.path.exists(done_marker):
            print(f"skip {idx} (done)", flush=True)
            continue
        doc = ds[idx]
        prompt_text = L.gsm_prompt(doc)
        ids = L.encode_prompt(tok, prompt_text, args.device)
        L.set_seed(args.seed_base + idx)
        t0 = time.time()
        if args.mode == "traced":
            tracer = Tracer(os.path.join(trace_dir, f"p{idx:05d}"), pid=idx)
            sampler = TracedHTSSampler(model, tok, args.device, tracer)
        else:
            sampler = hts_mod.HTSSampler(model, tok, args.device)
        final_codes, stats = sampler.generate_hts(prompt_text=prompt_text, input_ids=ids, **kwargs)
        dt = time.time() - t0
        processed = [c.strip() for c in final_codes]
        out = {"doc": doc, "target": doc["answer"], "resps": [[c] for c in processed], "prompt": prompt_text,
               "entropy_history": stats.get("entropy_history", []), "pruning_history": stats.get("pruning_history", []),
               "final_scores": stats.get("final_scores", []), "all_trajectories": stats.get("all_trajectories", []),
               "nfe": stats.get("nfe", 0), "first_block_nfe": stats.get("first_block_nfe", 0),
               "svf_calls": stats.get("svf_calls", 0), "total_steps": stats.get("total_steps", 0),
               "num_gen_blocks": stats.get("num_gen_blocks", []), "steps_per_block": stats.get("steps_per_block", []),
               "problem_idx": idx, "wall_s": dt, "mode": args.mode}
        with open(res_path, "a") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        with open(done_marker, "w") as f:
            f.write(json.dumps({"wall_s": dt, "nfe": stats.get("nfe"), "svf_calls": stats.get("svf_calls")}))
        print(f"problem {idx}: {dt/60:.1f} min, nfe={stats.get('nfe')} svf={stats.get('svf_calls')} "
              f"blocks={stats.get('num_gen_blocks')} steps/block={stats.get('steps_per_block')} top='{processed[0][-80:]!r}'", flush=True)


if __name__ == "__main__":
    main()
