"""Traced copy of FastTTS search/beam_search.py::_beam_search / beam_search (commit 1c01d54).

Control flow, sampling parameters and RNG consumption are verbatim; the only additions are uid
bookkeeping on Beam objects and `TRACER` calls. Import after FastTTS's root directory is on
sys.path (its modules use top-level `from config import ...` imports)."""
import copy
import json
import logging
import time
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from vllm import SamplingParams
import torch.cuda.nvtx as nvtx

from config import SearchConfig
from models.vllm_wrapper import GeneratorVLLMModelWrapper, VerifierVLLMModelWrapper
from search.beam import Beam
from search.utils import build_conversation, aggregate_scores, split_string_by_separator, truncate_sentence_by_tokens
from search.beam_search import beam_is_completed, score_beam, generate_beam

logger = logging.getLogger(__name__)

TRACER = None


class Tracer:
    def __init__(self, path, pid):
        self.f = open(path, "w")
        self.pid = pid
        self._uid = 0

    def uid(self):
        self._uid += 1
        return self._uid

    def w(self, rec):
        rec["pid"] = self.pid
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _log_iter(tracer, i, iter_beams, agg_index, search_config, gen_time, ver_time, n_completed):
    rows = []
    for b in iter_beams:
        rows.append({"uid": b.uid, "src_uid": b.src_uid, "step_tokens": b.completion_tokens,
                     "total_tokens": b.total_completion_tokens, "all_scores": [float(s) for s in b.all_scores],
                     "agg_score": float(aggregate_scores(b.all_scores[:agg_index], search_config.agg_strategy)) if b.all_scores else None,
                     "completed": bool(b.completed), "pruned": bool(b.pruned), "n_chars": len(b.current_text)})
    tracer.w({"t": "iter", "i": i, "gen_s": gen_time, "ver_s": ver_time, "n_completed_total": n_completed,
              "time": time.time(), "rows": rows})


def _beam_search(
    batch_of_prompts: List[str],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> List[Beam]:
    """Beam search implementation with optimizations."""
    tracer = TRACER

    base_sampling_params = SamplingParams(
        temperature=search_config.temperature,
        max_tokens=search_config.max_tokens,
        top_p=search_config.top_p,
        stop=[search_config.stop],
        include_stop_str_in_output=True,
        n=1,
    )

    final_sampling_params = SamplingParams(
        temperature=search_config.temperature,
        max_tokens=search_config.max_tokens,
        top_p=search_config.top_p,
        n=1,
    )

    beams: List[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(search_config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                    total_completion_tokens=0,
                    completion_time=0.0,
                )
            )
            beams[-1].uid = tracer.uid(); beams[-1].src_uid = None

    completed_beams: List[Beam] = []
    total_generator_latency_s = 0
    total_verifier_latency_s = 0
    n_generator_latency_s = 0
    n_verifier_latency_s = 0
    total_num_tokens = 0
    n_completion_tokens = 0
    extended_tokens_list = []

    tokenizer = generator.get_tokenizer()

    conv = build_conversation(batch_of_prompts[0], "", search_config.system_prompt)
    conv_str = tokenizer.apply_chat_template(conv,
                                             add_generation_prompt=True,
                                             continue_final_message=False, tokenize=True)
    prompt_token_length = len(conv_str)
    tracer.w({"t": "problem", "prompt_tokens": prompt_token_length, "n": search_config.n, "beam_width": search_config.beam_width,
              "num_iterations": search_config.num_iterations, "temperature": search_config.temperature,
              "max_tokens": search_config.max_tokens, "agg_strategy": search_config.agg_strategy, "time": time.time()})

    logger.info("Starting beam search iterations")
    for i in tqdm(range(search_config.num_iterations), desc="Beam search iterations"):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]
        for b in active_beams: b.src_uid = b.uid if i > 0 else None

        if (len(completed_beams) >= search_config.n or len(active_beams) == 0) and n_generator_latency_s == 0:
            n_generator_latency_s = total_generator_latency_s
            n_verifier_latency_s = total_verifier_latency_s
            n_completion_tokens = total_num_tokens
            logger.info(f"Reached target n: {len(completed_beams)} completed beams after {n_generator_latency_s + n_verifier_latency_s:.2f}s, {n_completion_tokens} total tokens")

        if len(active_beams) != search_config.n:
            repeats = (search_config.n // len(active_beams))

            if getattr(generator.config, 'prefix_aware_scheduling', False):
                final_beams = []
                remainder = search_config.n % len(active_beams)

                for x, beam in reversed(list(enumerate(active_beams))):
                    repeats_for_this_beam = repeats + (1 if x < remainder else 0)
                    duplicates = []
                    for _ in range(repeats_for_this_beam-1):
                        duplicate = copy.deepcopy(beam)
                        duplicate.uid = tracer.uid(); duplicate.src_uid = beam.uid
                        if beam.future_texts:
                            last_text = truncate_sentence_by_tokens(
                                beam.future_texts[-1][0], tokenizer,
                                mean_ratio=search_config.truncation_mean_ratio,
                                std_ratio=search_config.truncation_std_ratio,
                                min_tokens=search_config.truncation_min_tokens,
                            )
                            duplicate.future_texts[-1] = (last_text, False)
                        duplicates.append(duplicate)
                    final_beams.extend([beam] + duplicates)
                active_beams = final_beams
            else:
                extended_active_beams = [
                    copy.deepcopy(b) for b in (active_beams * repeats)
                ]
                for b in extended_active_beams:
                    b.uid = tracer.uid()  # deepcopy carried src_uid = source beam's uid
                    if b.future_texts:
                        last_text = truncate_sentence_by_tokens(
                            b.future_texts[-1][0], tokenizer,
                            mean_ratio=search_config.truncation_mean_ratio,
                            std_ratio=search_config.truncation_std_ratio,
                            min_tokens=search_config.truncation_min_tokens,
                        )
                        b.future_texts[-1] = (last_text, False)
                active_beams = (active_beams + extended_active_beams)[:search_config.n]

            if len(active_beams) != search_config.n:
                raise ValueError(
                    f"Expected {search_config.n} active beams, but got {len(active_beams)}"
                )
        iter_beams = list(active_beams)

        extended_beams = 0
        extended_tokens = []
        for beam in active_beams:
            if len(beam.future_texts) > 0:
                extended_beams += 1
                next_text, is_finished_this_step = beam.future_texts[0]
                if i == search_config.num_iterations - 1:
                    while beam.future_texts:
                        next_text, _ = beam.future_texts.pop(0)
                        num_tokens = len(tokenizer.encode(next_text))
                        beam.completion_tokens = num_tokens
                        beam.total_completion_tokens += beam.completion_tokens
                        beam.current_text += next_text
                        extended_tokens.append(num_tokens)
                    beam.skipped_this_step = beam.completed
                elif is_finished_this_step:
                    beam.skipped_this_step = True
                else:
                    num_tokens = len(tokenizer.encode(next_text))
                    beam.completion_tokens = num_tokens
                    beam.total_completion_tokens += beam.completion_tokens
                    beam.current_text += next_text
                    beam.future_texts.pop(0)
                    extended_tokens.append(num_tokens)

        current_sampling_params = final_sampling_params if i == search_config.num_iterations - 1 else base_sampling_params

        convs = [
            build_conversation(b.prompt, b.current_text, search_config.system_prompt)
            for b in active_beams if not b.skipped_this_step
        ]
        add_generation_prompt = i == 0
        continue_final_message = i > 0

        gen_time = 0.0
        if convs:
            if hasattr(search_config, 'custom_chat_template') and search_config.custom_chat_template is not None:
                tokenizer.chat_template = search_config.custom_chat_template
            templated_convs = tokenizer.apply_chat_template(
                convs,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=False,
            )

            lookahead = 0 if i == search_config.num_iterations - 1 else search_config.lookahead

            gen_results, gen_time = generate_beam(
                    templated_convs, lookahead, generator, current_sampling_params, tokenizer=tokenizer
                )
            total_generator_latency_s += gen_time

        prompts, completions = [], []
        skipped_beams = 0
        verified_beams = 0
        counter = 0
        for beam in active_beams:
            skipped_this_step = beam.skipped_this_step
            if skipped_this_step and i < search_config.num_iterations - 1:
                next_text, _ = beam.future_texts.pop(0)
                beam.current_text += next_text
                num_tokens = len(tokenizer.encode(next_text))
                beam.completion_tokens = num_tokens
                beam.total_completion_tokens += beam.completion_tokens
                beam.history.append("")
                beam.skipped_this_step = False
                is_completed = beam.completed
                skipped_beams += 1
                extended_tokens.append(num_tokens)
            else:
                gen_result = gen_results[counter]
                counter += 1
                is_completed = beam_is_completed(gen_result, prompt_token_length + gen_result.completion_tokens[0])
                if i == search_config.num_iterations - 1:
                    current_text = gen_result.next_texts[0]
                    future_texts = []
                else:
                    current_text, future_texts, _ = split_string_by_separator(
                        gen_result.next_texts[0], search_config.stop
                    )
                beam.future_texts = future_texts
                beam.next_texts = gen_result.next_texts
                beam.stop_reasons = gen_result.stop_reasons
                beam.lookahead_texts = gen_result.lookahead_texts
                num_tokens = len(tokenizer.encode(current_text))
                beam.completion_tokens = num_tokens
                beam.total_completion_tokens += beam.completion_tokens
                beam.current_text += current_text
                beam.history.append(gen_result.next_texts[0])

            if is_completed:
                if not beam.completed:
                    beam.completed = True
                    beam.completion_time = total_generator_latency_s + total_verifier_latency_s
                if not beam.future_texts:
                    completed_beams.append(beam)

            if len(beam.all_scores) >= i + 1 and i < search_config.num_iterations - 1:
                verified_beams += 1
            elif beam.future_texts and beam.future_texts[0][1]:
                prompts.append(beam.prompt)
                completions.append([beam.current_text + beam.future_texts[0][0]])
            else:
                prompts.append(beam.prompt)
                completions.append([beam.current_text])

        extended_tokens_list.append(extended_tokens)

        verifier_time = 0.0
        if prompts:
            scores, verifier_time = score_beam(verifier, prompts, completions, tokenizer)
            total_verifier_latency_s += verifier_time
        else:
            scores = []

        agg_index = i + 1 if i < search_config.num_iterations - 1 else max([len(s) for s in scores])
        counter = 0
        agg_scores = []
        for beam in active_beams:
            if i == search_config.num_iterations - 1 or len(beam.all_scores) < agg_index:
                score = scores[counter]
                agg_scores.append(aggregate_scores(score[0][:agg_index], search_config.agg_strategy))
                beam.all_scores = score[0]
                counter += 1
            else:
                agg_scores.append(aggregate_scores(beam.all_scores[:agg_index], search_config.agg_strategy))
        assert counter == len(scores), f"counter: {counter}, len(scores): {len(scores)}"

        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        if len(active_beams) == 0:
            logger.info(f"Early exit: {len(active_beams)} active, {len(completed_beams)} completed")
            _log_iter(tracer, i, iter_beams, agg_index, search_config, gen_time, verifier_time, len(completed_beams))
            break

        top_indices = np.argsort(np.array(agg_scores).flatten())[
            -(search_config.n // search_config.beam_width) :
        ]
        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True
        _log_iter(tracer, i, iter_beams, agg_index, search_config, gen_time, verifier_time, len(completed_beams))

        num_steps = [beam.current_text.count("\n\n") for beam in active_beams if not beam.pruned]
        agg_scores_length = [len(beam.all_scores) for beam in active_beams if not beam.pruned]
        stop_reasons = [beam.stop_reasons[0] for beam in active_beams if not beam.pruned]
        logger.info(f"-" * 100)
        logger.info(f"Iteration {i} completed beams: {len(completed_beams)}, skipped beams: {skipped_beams}, extended beams: {extended_beams}, verifier beams: {verified_beams}, total latency: {total_generator_latency_s + total_verifier_latency_s:.2f}s, length of agg_scores: {agg_scores_length}, num_steps: {num_steps}, stop reasons: {stop_reasons}")
        for x, beam in enumerate([b for b in active_beams if not b.pruned]):
            if num_steps[x] != i + 1:
                logger.warning(f"Beam {x} has {num_steps[x]} steps, expected {i + 1}")
                logger.warning(f"Beam {x} current text: {beam.current_text}")
                logger.warning(f"Beam {x} history: {beam.history}, stop reasons: {beam.stop_reasons}")
    total_num_tokens += sum([b.completion_tokens for b in completed_beams])
    n_completion_tokens = total_num_tokens if n_generator_latency_s == 0 else n_completion_tokens
    n_generator_latency_s = total_generator_latency_s if n_generator_latency_s == 0 else n_generator_latency_s
    n_verifier_latency_s = total_verifier_latency_s if n_verifier_latency_s == 0 else n_verifier_latency_s

    if search_config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, search_config.agg_strategy),
            reverse=True,
        )

    return completed_beams, total_generator_latency_s, total_verifier_latency_s, n_generator_latency_s, n_verifier_latency_s, total_num_tokens, n_completion_tokens, extended_tokens_list


def beam_search(
    examples: Dict[str, Any],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> Dict[str, Any]:
    """Beam search for a batch of examples."""
    problems = examples["problem"]
    assert len(problems) == 1, "batch_of_prompts should be a list of length 1 for now"

    nvtx.range_push("Total")
    completed_beams, total_generator_latency_s, total_verifier_latency_s, n_generator_latency_s, n_verifier_latency_s, total_num_tokens, n_completion_tokens, extended_tokens_list = _beam_search(problems, search_config, generator, verifier)
    nvtx.range_pop()

    grouped_results = defaultdict(list)
    for results in completed_beams:
        grouped_results[results.prompt].append(results)

    results = {
        "completions": [],
        "pred": [],
        "completion_tokens": [],
        "scores": [],
        "effective_num_tokens": [],
        "total_num_tokens": total_num_tokens,
        "n_completion_tokens": n_completion_tokens,
        "total_generator_latency_s": total_generator_latency_s,
        "total_verifier_latency_s": total_verifier_latency_s,
        "n_generator_latency_s": n_generator_latency_s,
        "n_verifier_latency_s": n_verifier_latency_s,
        "completion_time": [],
        "vllm_metrics": {},
        "vllm_metrics_summary": {},
        "extended_tokens_list": extended_tokens_list,
    }

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, search_config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["pred"].append(pred)
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["completion_tokens"].append([b.completion_tokens for b in beams])
        results["completion_time"].append([b.completion_time for b in beams])
        results["effective_num_tokens"].append([b.total_completion_tokens for b in beams])
        TRACER.w({"t": "end", "completed_uids": [b.uid for b in beams], "pred_uid": beams[int(np.argmax(agg_scores))].uid,
                  "agg_scores": [float(s) for s in agg_scores], "total_tokens": [b.total_completion_tokens for b in beams],
                  "gen_s": total_generator_latency_s, "ver_s": total_verifier_latency_s, "time": time.time()})

    return results
