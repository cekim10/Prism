"""Traced copy of Prism's HTSSampler.generate_hts / _branch_and_resample.

The control flow, tensor ops and RNG consumption are verbatim from
LLaDA/LLaDA_Prism/dllm_eval/models/hts_sampler.py; the only additions are `tracer.*` calls and
the TracedModel proxy, none of which draw random numbers or touch the tensors used by the search.
"""
import hashlib
import json
import math
import time

import numpy as np
import torch

from prism_loader import load_prism_modules

hts_mod, ver_mod = load_prism_modules()
HTSSampler = hts_mod.HTSSampler
CodeVerifier = ver_mod.CodeVerifier

# LLaDA-8B: d=4096, 32 layers, mlp 12288, vocab 126464 (untied head)
P_NONEMBED = 32 * (4 * 4096 * 4096 + 3 * 4096 * 12288) + 4096 * 126464
N_LAYERS, D_MODEL = 32, 4096


def flops_per_row(L):
    return 2.0 * P_NONEMBED * L + 4.0 * N_LAYERS * D_MODEL * L * L


def _h(t: torch.Tensor) -> str:
    return hashlib.blake2b(t.detach().contiguous().cpu().numpy().tobytes(), digest_size=16).hexdigest()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


class Tracer:
    def __init__(self, path_prefix, pid, keep_canvases=True):
        self.pid = pid
        self.f = open(path_prefix + ".jsonl", "w")
        self.canvas_path = path_prefix + "_canvases.npz"
        self.keep_canvases = keep_canvases
        self.canvases, self.canvas_index = [], []
        self.seq = 0
        self.ctx = {"kind": None, "block": None, "step": None, "row": None}
        self.prompt_length = None

    def _w(self, rec):
        rec["pid"] = self.pid
        self.f.write(json.dumps(rec) + "\n")

    def begin_problem(self, prompt_length, total_length, cfg):
        self.prompt_length = prompt_length
        self._w({"t": "problem", "prompt_len": prompt_length, "total_len": total_length, "cfg": cfg, "time": time.time()})

    def set_ctx(self, kind, block, step, row=None):
        self.ctx = {"kind": kind, "block": block, "step": step, "row": row}

    # called from generate_hts right before a batched forward on rows x[:bsz]
    def record_state(self, kind, block, step, x, conf, prompt_length, window_end, mask_id, block_length):
        bsz = x.shape[0]
        rows = []
        for b in range(bsz):
            xb, cb = x[b], conf[b]
            rows.append({
                "row": b,
                "h_fwd": _h(xb),
                "h_full": hashlib.blake2b(_h(xb).encode() + _h(cb.float()).encode(), digest_size=16).hexdigest(),
                "nm_blk": int((xb[window_end - block_length:window_end] == mask_id).sum().item()),
                "nm_tot": int((xb[prompt_length:] == mask_id).sum().item()),
            })
            if self.keep_canvases:
                self.canvas_index.append((self.seq, b))
                self.canvases.append(xb[prompt_length:].cpu().numpy().astype(np.int32))
        self._w({"t": "state", "seq": self.seq, "kind": kind, "block": block, "step": step, "rows": rows})

    # called by TracedModel for every model forward
    def record_forward(self, input_ids, latency, logits):
        n, L = input_ids.shape
        hashes = [_h(input_ids[i]) for i in range(n)]
        dup_checks = []
        seen = {}
        for i, h in enumerate(hashes):
            if h in seen:
                j = seen[h]
                dup_checks.append({"i": i, "j": j, "logits_equal": bool(torch.equal(logits[i], logits[j])),
                                   "max_abs_diff": float((logits[i].float() - logits[j].float()).abs().max().item())})
            else:
                seen[h] = i
        self._w({"t": "fwd", "seq": self.seq, **self.ctx, "n": n, "L": L, "lat": latency,
                 "flops_row": flops_per_row(L), "hashes": hashes, "dup_checks": dup_checks})
        self.seq += 1

    def record_verifier(self, block, step, row, score, text):
        self._w({"t": "verifier", "block": block, "step": step, "row": row, "score": float(score),
                 "h_text": hashlib.blake2b(text.encode(), digest_size=16).hexdigest()})

    def record_prune(self, block, step, scores, keys, selected, target_width, branch_map, current_bsz):
        self._w({"t": "prune", "block": block, "step": step, "scores": scores, "keys": [str(k) for k in keys],
                 "selected": selected, "target_width": target_width, "branch": branch_map, "bsz_before": current_bsz})

    def record_misc(self, **kw):
        self._w({"t": "misc", **kw})

    def end_problem(self, final, stats):
        self._w({"t": "end", "final": final, "stats": {k: v for k, v in stats.items() if k != "all_trajectories"},
                 "time": time.time()})
        self.f.close()
        if self.keep_canvases and self.canvases:
            np.savez_compressed(self.canvas_path, canvases=np.stack(self.canvases),
                                index=np.array(self.canvas_index, dtype=np.int64))


class TracedModel:
    """Callable proxy: identical outputs, records per-row input hashes, latency and duplicate-row logit equality."""

    def __init__(self, model, tracer):
        self._m = model
        self.tracer = tracer
        self.config = model.config
        self.device = model.device

    def __call__(self, input_ids=None, attention_mask=None, **kw):
        _sync()
        t0 = time.time()
        out = self._m(input_ids=input_ids, attention_mask=attention_mask, **kw)
        _sync()
        lat = time.time() - t0
        self.tracer.record_forward(input_ids, lat, out.logits)
        return out

    def eval(self):
        return self


class TracedHTSSampler(HTSSampler):
    def __init__(self, model, tokenizer, device, tracer):
        self.tracer = tracer
        self.raw_model = model
        self.model = TracedModel(model, tracer)
        self.tokenizer = tokenizer
        self.device = device
        self.verifier = CodeVerifier(self.model, tokenizer, device)

    def _branch_and_resample(self, x, conf_scores, survivor_indices, target_width, mask_id,
                             prompt_length, resample_window=6, task_type="code"):
        num_survivors = len(survivor_indices)
        self._last_branch_map = []
        if num_survivors == 0: return x[:target_width].clone(), conf_scores[:target_width].clone()

        if task_type == "math": resample_window = 2
        elif task_type == "reasoning": resample_window = 6
        elif task_type == "code": resample_window = 2

        base_repeat = target_width // num_survivors
        remainder = target_width % num_survivors
        new_x_list = []
        new_conf_list = []

        for i in range(num_survivors):
            count = base_repeat + (1 if i < remainder else 0)
            if count == 0: continue

            survivor_x = x[survivor_indices[i]]
            survivor_conf = conf_scores[survivor_indices[i]]

            new_x_list.append(survivor_x.unsqueeze(0))
            new_conf_list.append(survivor_conf.unsqueeze(0))
            self._last_branch_map.append({"new": len(new_x_list) - 1, "src": int(survivor_indices[i].item()), "perturbed": []})

            if count > 1:
                gen_part = survivor_x[prompt_length:]
                gen_conf = survivor_conf[prompt_length:]
                non_mask_indices = (gen_part != mask_id).nonzero(as_tuple=True)[0]

                for _ in range(count - 1):
                    perturbed_x = survivor_x.clone()
                    perturbed_conf = survivor_conf.clone()
                    perturbed_positions = []

                    if len(non_mask_indices) > 0:
                        pool_size = min(resample_window * 3, len(non_mask_indices))
                        current_token_confs = gen_conf[non_mask_indices]

                        _, candidate_indices = torch.topk(current_token_confs, k=pool_size, largest=False)

                        num_to_perturb = min(resample_window, pool_size)
                        rand_indices = torch.randperm(pool_size, device=self.device)[:num_to_perturb]
                        selected_sub_indices = candidate_indices[rand_indices]

                        target_indices_in_x = prompt_length + non_mask_indices[selected_sub_indices]
                        perturbed_x[target_indices_in_x] = mask_id
                        perturbed_conf[target_indices_in_x] = 0.0
                        perturbed_positions = sorted(int(v) for v in (target_indices_in_x - prompt_length).tolist())

                    new_x_list.append(perturbed_x.unsqueeze(0))
                    new_conf_list.append(perturbed_conf.unsqueeze(0))
                    self._last_branch_map.append({"new": len(new_x_list) - 1, "src": int(survivor_indices[i].item()), "perturbed": perturbed_positions})

        return torch.cat(new_x_list, dim=0), torch.cat(new_conf_list, dim=0)

    @torch.no_grad()
    def generate_hts(self, prompt_text, input_ids, problem_data=None,
                     initial_N=1, final_K=1, survivor_K=None,
                     prune_step_pct=0.0, reward_mode="confidence",
                     temperature=0.7, block_length=32, steps=64, gen_length=1024,
                     top_p=0.95, top_k=None, minimal_topk=1, threshold=0.9,
                     eos_id=156892, mask_id=156895,
                     hts_mode=False, hts_start_pct=0.1, hts_end_pct=0.6, decay_factor=1.5,
                     hts_survivor_k=4, task_type="code", until=None, pruning_interval=0):
        tracer = self.tracer

        input_ids = input_ids.to(self.device)
        if input_ids.shape[0] == 1: input_ids = input_ids.repeat(initial_N, 1)

        schedule_map = {}
        ts_start, tr_end = 0, 0
        if not hts_mode:
            final_K_list = [final_K] if not isinstance(final_K, list) else final_K
            prune_pct_list = [prune_step_pct] if not isinstance(prune_step_pct, list) else prune_step_pct
            survivor_K_list = final_K_list if survivor_K is None else ([survivor_K] if not isinstance(survivor_K, list) else survivor_K)
            if len(survivor_K_list) < len(final_K_list): survivor_K_list.extend(final_K_list[len(survivor_K_list):])
            for pct, width, parents in zip(prune_pct_list, final_K_list, survivor_K_list):
                if pct > 0:
                    s = int(steps * pct)
                    schedule_map[s] = (width, parents)
        else:
            final_K_list = [final_K] if not isinstance(final_K, int) else [final_K]
            ts_start, tr_end = int(steps * hts_start_pct), int(steps * hts_end_pct)

        steps = min(steps, gen_length // minimal_topk)
        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        x = torch.full((initial_N, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()

        conf_scores = torch.zeros((initial_N, total_length), dtype=torch.float32, device=self.device)
        conf_scores[:, :prompt_length] = 1.0

        prefill_blocks = prompt_length // block_length
        num_gen_blocks = max(1, num_blocks - prefill_blocks)
        current_bsz = initial_N

        next_allowed_pruning_step = ts_start if hts_mode else 0

        stats = {
            "initial_n": initial_N, "final_k": final_K_list[-1],
            "pruning_history": [], "entropy_history": [], "nfe": 0.0,
            "svf_calls": 0, "final_scores": [], "total_steps": steps,
            "first_block_nfe": 0.0, "num_gen_blocks": [], "steps_per_block": []
        }

        tracer.begin_problem(prompt_length, total_length, {
            "initial_N": initial_N, "final_K": final_K, "hts_survivor_k": hts_survivor_k, "hts_mode": hts_mode,
            "ts_start": ts_start, "tr_end": tr_end, "decay_factor": decay_factor, "pruning_interval": pruning_interval,
            "steps": steps, "block_length": block_length, "gen_length": gen_length, "temperature": temperature,
            "threshold": threshold, "task_type": task_type, "reward_mode": reward_mode, "num_blocks": num_blocks,
            "prefill_blocks": prefill_blocks})

        for num_block in range(prefill_blocks, num_blocks):
            stats["num_gen_blocks"].append(num_block)

            window_end = (num_block + 1) * block_length
            schedule = self._get_num_transfer_tokens(block_length, steps)

            steps_this_block = 0
            for step in range(steps):
                steps_this_block += 1
                cur_full_x = x[:current_bsz, :]

                perform_pruning = False
                num_parents_to_select = 0

                if hts_mode and step >= next_allowed_pruning_step and step < tr_end:
                    target_width = max(final_K_list[-1], math.ceil(initial_N * (decay_factor ** -(step - ts_start))))
                    if current_bsz > target_width:
                        perform_pruning = True
                        num_parents_to_select = hts_survivor_k
                elif not hts_mode and step in schedule_map:
                    target_width, num_parents_to_select = schedule_map[step]
                    if current_bsz > target_width: perform_pruning = True

                if perform_pruning:
                    stats["nfe"] += current_bsz
                    if num_block == prefill_blocks:
                        stats["first_block_nfe"] += current_bsz

                    stats["svf_calls"] += current_bsz

                    tracer.record_state("rough", num_block, step, cur_full_x, conf_scores[:current_bsz], prompt_length, window_end, mask_id, block_length)
                    tracer.set_ctx("rough", num_block, step)
                    gen_logits = self._chunked_forward(cur_full_x, chunk_size=64, slice_indices=(prompt_length, total_length))
                    rough_ids = torch.argmax(gen_logits, dim=-1)
                    rough_codes_snippet = self.tokenizer.batch_decode(rough_ids, skip_special_tokens=True)
                    candidates = []
                    for i in range(current_bsz):
                        full_code = rough_codes_snippet[i]
                        tracer.set_ctx("verifier", num_block, step, row=i)
                        s = self._safe_scalar(self.verifier.get_reward(prompt_text, full_code, mode=reward_mode, problem_data=problem_data, current_logits=gen_logits[i] if reward_mode != "svf" else None, task_type=task_type))
                        tracer.record_verifier(num_block, step, i, s, full_code)
                        s += self._analyze_structure(full_code, task_type=task_type)
                        clean_content = full_code.strip().replace(" ", "").replace("\n", "")
                        candidates.append({'score': s, 'idx': i, 'key': hash(clean_content[:200] + clean_content[-200:])})

                    stats["pruning_history"].append({"step": step, "scores": [c['score'] for c in candidates]})
                    candidates.sort(key=lambda x: x['score'], reverse=True)

                    selected_indices, seen_keys = [], set()
                    for cand in candidates:
                        if len(selected_indices) >= num_parents_to_select: break
                        if cand['key'] not in seen_keys:
                            selected_indices.append(cand['idx']); seen_keys.add(cand['key'])

                    if len(selected_indices) < num_parents_to_select:
                        for cand in candidates:
                            if len(selected_indices) >= num_parents_to_select: break
                            if cand['idx'] not in selected_indices: selected_indices.append(cand['idx'])

                    top_indices = torch.tensor(selected_indices, device=self.device)
                    x, conf_scores = self._branch_and_resample(x, conf_scores, top_indices, target_width, mask_id, prompt_length, task_type=task_type)
                    tracer.record_prune(num_block, step, [c['score'] for c in sorted(candidates, key=lambda c: c['idx'])],
                                        [c['key'] for c in sorted(candidates, key=lambda c: c['idx'])],
                                        selected_indices, target_width, self._last_branch_map, current_bsz)

                    current_bsz = target_width
                    cur_full_x = x[:current_bsz, :]
                    next_allowed_pruning_step = step + 1 + pruning_interval

                active_mask = x[:current_bsz, window_end-block_length:window_end] == mask_id
                if active_mask.sum() == 0: break

                stats["nfe"] += current_bsz
                if num_block == prefill_blocks:
                    stats["first_block_nfe"] += current_bsz

                tracer.record_state("denoise", num_block, step, cur_full_x, conf_scores[:current_bsz], prompt_length, window_end, mask_id, block_length)
                tracer.set_ctx("denoise", num_block, step)
                active_logits = self._chunked_forward(cur_full_x, chunk_size=32, slice_indices=(window_end-block_length, window_end))

                active_logits[:, :, eos_id] = -1e10

                with torch.no_grad():
                    if len(stats["entropy_history"]) < 32:
                        probs_for_stats = torch.softmax(active_logits.float(), dim=-1)
                        entropy_per_branch = (-(probs_for_stats * torch.log(probs_for_stats + 1e-10)).sum(dim=-1).mean(dim=-1)).cpu().numpy().tolist()
                        stats["entropy_history"].append(entropy_per_branch)

                x0, x0_p = self._sample_with_temperature(active_logits, temperature, top_k, top_p)

                num_transfer = schedule[step].item()
                confidence = torch.where(active_mask, x0_p, -torch.inf)
                transfer_idx = torch.zeros_like(x0, dtype=torch.bool)

                for b in range(current_bsz):
                    k_transfer = min(num_transfer, active_mask[b].sum().item())
                    active_indices = torch.where(active_mask[b])[0]
                    if (confidence[b] > threshold).sum().item() >= k_transfer:
                        conf_indices = torch.where((confidence[b] > threshold) & active_mask[b])[0]; transfer_idx[b, conf_indices] = True
                    elif len(active_indices) > 0:
                        _, topk_indices = torch.topk(confidence[b][active_indices], k=min(k_transfer, len(active_indices))); transfer_idx[b, active_indices[topk_indices]] = True

                if transfer_idx.any():
                    x[:current_bsz, window_end-block_length:window_end][transfer_idx] = x0[transfer_idx]
                    conf_scores[:current_bsz, window_end-block_length:window_end][transfer_idx] = x0_p[transfer_idx]

                tracer.record_misc(kind="transfer", block=num_block, step=step,
                                   n_transfer=transfer_idx.sum(dim=1).tolist(),
                                   argmax_equal_sample=[bool(v) for v in ((torch.argmax(active_logits, dim=-1) == x0) | ~transfer_idx).all(dim=1).tolist()])

                if task_type in ["math", "reasoning"]:
                    for b in range(current_bsz):
                        gen_span = x[b, prompt_length:window_end]
                        text_snippet = self.tokenizer.decode(gen_span, skip_special_tokens=True)
                        should_stop = False
                        if task_type == "reasoning" and ("###" in text_snippet):
                            should_stop = True
                        if task_type == "math" and ("\\boxed{" in text_snippet and "}" in text_snippet.split("\\boxed{")[-1]):
                            should_stop = True

                        if should_stop:
                            non_mask_indices = (gen_span != mask_id).nonzero(as_tuple=True)[0]
                            if len(non_mask_indices) > 0:
                                last_idx = non_mask_indices[-1].item()
                                if last_idx + 1 < len(gen_span):
                                    x[b, prompt_length + last_idx + 1 : window_end] = eos_id
                                if window_end < total_length:
                                    x[b, window_end:] = eos_id
                                    conf_scores[b, window_end:] = 1.0

                for b in range(current_bsz):
                    gen_window = x[b, prompt_length:window_end]
                    eos_indices = (gen_window == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_indices) > 0:
                        first_eos_idx = eos_indices[0].item()
                        if first_eos_idx + 1 < len(gen_window):
                            x[b, prompt_length + first_eos_idx + 1 : window_end] = eos_id

            stats["steps_per_block"].append(steps_this_block)
            x = x[:current_bsz]

        stats["nfe"] = int(round(stats["nfe"]))
        stats["first_block_nfe"] = int(round(stats["first_block_nfe"]))

        final_gen_tokens = x[:current_bsz, prompt_length:]
        final_codes = self.tokenizer.batch_decode(final_gen_tokens, skip_special_tokens=True)
        final_candidates = []

        stats["svf_calls"] += len(final_codes)

        for i in range(len(final_codes)):
            txt = final_codes[i]
            if until:
                for term in until:
                    if term in txt: txt = txt.split(term)[0]
            tracer.set_ctx("verifier_final", None, None, row=i)
            s = self._safe_scalar(self.verifier.get_reward(prompt_text, txt, mode=reward_mode, task_type=task_type))
            tracer.record_verifier(None, None, i, s, txt)
            s += self._analyze_structure(txt, task_type)
            final_candidates.append({'resp': txt, 'score': s})

        final_candidates.sort(key=lambda x: x['score'], reverse=True)
        stats["final_scores"] = [c['score'] for c in final_candidates]
        stats["all_trajectories"] = [{"rank": i+1, "resp": c['resp'], "score": c['score']} for i, c in enumerate(final_candidates)]

        tracer.end_problem([c['resp'] for c in final_candidates], stats)
        return [c['resp'] for c in final_candidates], stats
