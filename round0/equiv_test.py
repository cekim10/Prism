"""Falsification test 1: the traced sampler must reproduce the untouched sampler bit-for-bit
(same seed) - final texts, scores, nfe, svf_calls, pruning history."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prism_loader as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="+", default=[0])
    ap.add_argument("--gen_length", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_quant", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tok = L.load_model_and_tokenizer(args.device)
    if not args.no_quant:
        from quant import quantize_model_inplace
        quantize_model_inplace(model, args.device)
    hts_mod, _ = L.load_prism_modules()
    from traced_sampler import TracedHTSSampler, Tracer

    ds = L.load_gsm8k_test()
    kwargs = dict(L.OFFICIAL_GSM8K_GEN_KWARGS, gen_length=args.gen_length)
    all_pass = True
    for idx in args.idx:
        doc = ds[idx]
        prompt_text = L.gsm_prompt(doc)
        ids = L.encode_prompt(tok, prompt_text, args.device)
        outs = {}
        for mode in ("orig", "traced"):
            L.set_seed(42 + idx)
            t0 = time.time()
            if mode == "orig":
                sampler = hts_mod.HTSSampler(model, tok, args.device)
            else:
                sampler = TracedHTSSampler(model, tok, args.device, Tracer(os.path.join(args.out, f"equiv_p{idx:05d}"), pid=idx))
            codes, stats = sampler.generate_hts(prompt_text=prompt_text, input_ids=ids, **kwargs)
            outs[mode] = {"codes": codes, "final_scores": stats["final_scores"], "nfe": stats["nfe"],
                          "svf_calls": stats["svf_calls"], "pruning_history": stats["pruning_history"],
                          "steps_per_block": stats["steps_per_block"], "wall_s": time.time() - t0}
            print(f"[{mode}] idx={idx} {outs[mode]['wall_s']/60:.1f} min nfe={stats['nfe']} svf={stats['svf_calls']} steps/block={stats['steps_per_block']}", flush=True)
        o, t = outs["orig"], outs["traced"]
        checks = {
            "codes_equal": o["codes"] == t["codes"],
            "scores_equal": o["final_scores"] == t["final_scores"],
            "nfe_equal": o["nfe"] == t["nfe"],
            "svf_equal": o["svf_calls"] == t["svf_calls"],
            "pruning_equal": o["pruning_history"] == t["pruning_history"],
            "steps_equal": o["steps_per_block"] == t["steps_per_block"],
        }
        ok = all(checks.values())
        all_pass &= ok
        print(f"idx={idx} EQUIVALENCE {'PASS' if ok else 'FAIL'} {checks}", flush=True)
        with open(os.path.join(args.out, f"equiv_p{idx:05d}_result.json"), "w") as f:
            json.dump({"checks": checks, "orig": o, "traced": t}, f, ensure_ascii=False, indent=1)
    print("ALL PASS" if all_pass else "SOME FAIL")


if __name__ == "__main__":
    main()
