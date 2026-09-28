# Round 0 preregistration: exact post-divergence state convergence in Prism search

Frozen 2026-09-26, before any instrumentation code was written or any trace was collected.
Thresholds below are not to be changed after seeing results.

## Question

In Prism's hierarchical tree search for dLLMs, do two search branches that have *diverged*
(passed through different states after their last common ancestor) later land on the *exact same
computational state*, often enough that executing that state once instead of once-per-branch
would remove a substantial fraction of forward compute?

Only exact, bit-level equivalence of the forward-pass input counts (T1). Text-level, verifier-level
or semantic similarity (T2/T3) is descriptive only and never enters the GO/KILL decision.

## Search algorithm audited (LLaDA/LLaDA_Prism/dllm_eval/models/hts_sampler.py)

Prism keeps a batch `x[0:current_bsz, 0:total_length]` of `current_bsz` canvases (token ids, mask id
126336 at undecoded positions) plus `conf_scores` of the same shape. Generation is
semi-autoregressive over blocks of `block_length` positions; inside a block, every step runs one
forward pass per row on the *full* canvas (stateless model, `use_cache=false`, attention mask all
ones), samples `x0 = argmax(logits/T + Gumbel)` and unmasks in each row the top-`num_transfer`
positions by noise-free max-prob `x0_p` (or all positions with `x0_p > threshold`).

Pruning (hts mode) happens at steps `s` with `ts_start <= s < tr_end` once `current_bsz` exceeds
`target_width(s) = max(final_K, ceil(N * decay^-(s - ts_start)))`, gated by `pruning_interval`.
A pruning event runs a second forward per row over the generated span (rough argmax decode), scores
each row with the self-verifier (`svf`: one extra forward on a judge prompt built from the decoded
text), picks `hts_survivor_k` parents (dedup by a text-hash key), and calls `_branch_and_resample`:
each survivor is copied unchanged once, and `count-1` further copies are made by re-masking
`resample_window` (=2 for math) positions chosen at random from the `3*resample_window` lowest
confidence decoded positions. Rows are otherwise independent: nothing is shared across rows except
the batch dimension of the forward.

Global search-control state (`current_bsz`, `next_allowed_pruning_step`, `stats`) is identical for
all rows of a problem and is a function of (block, step) only, so it never distinguishes rows.

## Exact computational state (frozen definition)

For one problem, the future computation of a row is fully determined by

```text
S_row = ( problem_id,                      # fixes prompt tokens, total_length, config
          canvas  = x[b, :]                # token ids incl. mask ids, full length
          conf    = conf_scores[b, :]      # only consumed at branch events (which tokens to re-mask)
          block, step )                    # fixes window_end, num_transfer, pruning eligibility
```

plus exogenous randomness (Gumbel noise, `randperm` at branch events), which is *not* state.

The forward pass itself depends only on `(problem_id, canvas)`. Therefore:

- **T1 forward key** `h_fwd = blake2b(problem_id, canvas_gen_part)` (prompt is constant per problem).
  Two executed forwards with equal `h_fwd` have bit-identical `input_ids`; this is the key used for
  all compute-sharing numbers.
- **T1 full key** `h_full = blake2b(problem_id, canvas_gen_part, conf_gen_part_float32)`.
  Reported next to `h_fwd` to show whether canvas-identical rows are also trajectory-identical.
- A merge is only admitted when the two forwards happen at the same `(block, step)` (same event),
  because that is the only time a serving system could execute one forward for both. Equal
  `h_fwd` at different events is reported separately as a memoisation ceiling, not as headroom.

Verifier forwards (`svf_score`) are keyed by `h_ver = blake2b(problem_id, verify_input_ids)`; they
are exact-equivalent iff the decoded text is identical. This is tracked but reported separately
(it is a text-level key, so it is closer to T2 in spirit even though the forward is exact).

## Duplicate taxonomy (frozen)

Executed forwards at one event are grouped by `h_fwd`. Every group of size `n_s > 1` is split by
the ancestry of its rows in the branch tree (parent of a row at event t is the same row at event
t-1, or the survivor it was copied from at a branch event):

- **C1 creation duplicates**: rows whose state sequences have been `h_full`-identical at every
  common event since their lowest common ancestor. They were born identical (unperturbed copy vs
  a copy whose random re-mask happened to pick the same positions, or two copies with the same
  re-mask) and never differed. This is search-level duplicate generation, *not* convergence.
- **C2 true post-divergence convergence**: rows whose `h_full` differed at some common event after
  their LCA and are now `h_fwd`-equal. This is the only category that supports the hypothesis.
- **C3 same-row recompute**: the pruning rough-decode forward and the denoising forward of the
  same step run on the same canvas for the unperturbed survivor copy. Prism-internal, same path,
  excluded from everything except a descriptive line.

For a group with `n_s` rows partitioned into `k_s` never-diverged classes: `C1 saving = (n_s - k_s) C(s)`,
`C2 saving = (k_s - 1) C(s)`.

## Primary metric

```text
C(s)        = forward FLOP proxy = 2 * P_nonembed * L + 4 * n_layers * d * L^2   (P=7.5e9, L=total_length)
R_compute   = sum_s (n_s - 1) C(s) / sum_s n_s C(s)                       over all denoise+rough forwards
S_oracle_C2 = sum_s (k_s - 1) C(s) / sum_all_forwards C(s)               <- PRIMARY
S_oracle_C1 = sum_s (n_s - k_s) C(s) / sum_all_forwards C(s)
```

The denominator is every forward actually executed (denoise, rough-decode, verifier) for the
problem set; nothing hypothetical is counted. The oracle shares only the forward pass among rows
with equal `h_fwd` at the same event; each row keeps its own Gumbel sample afterwards, so branch
multiplicity, scores, selection, voting and final answers are unchanged by construction.
Wall-clock: forward latency is measured per chunk and attributed per row; a latency-weighted
version of the same ratios is reported alongside, but FLOP-weighted `S_oracle_C2` is the decision
metric.

Also reported: `R_state = 1 - unique/executed`, per-progress bins of `S_oracle_C2` (0-25/25-50/
50-75/75-100% of executed steps of the problem), and for every C2 merge event the share of the
problem's forward compute that remains after it.

## Workload and configuration

- Model: `GSAI-ML/LLaDA-8B-Instruct`, bf16, on Apple MPS (only accelerator available here).
- Task: GSM8K test split, problems in dataset order starting at index 0. Prompt = `gsm_prompt(doc)`
  exactly as `dllm_eval/tasks/gsm8k/utils.py`, tokenised without special tokens, no chat template,
  matching `scripts/run_gsm8k.sh` (which does not pass `--apply_chat_template`).
- Generation kwargs exactly as `scripts/run_gsm8k.sh` through `LLaDA.generate_until`:
  `hts_N=16, final_K=4, hts_survivor_k=2, hts_mode=True, hts_start_pct=0.1, hts_end_pct=0.6,
  pruning_interval=3, decay_factor=1.8, reward_mode=svf, task_type=math, steps=32,
  block_length=32, gen_length=256, temperature=0.7, top_p=0.95, top_k=None, threshold=0.85,
  mask_id=126336, eos_id=126081`.
- Sample size: as many problems as fit the local compute budget, executed in index order and cut
  only at a problem boundary; target 30, hard minimum 12 for a verdict. The count is reported.
- Seed 42 (`torch.manual_seed`, `random`, `numpy`) as in `evaluation_script.py`.
- Secondary workloads (MATH500, HumanEval, MEDAL) only if the primary is GRAY or GO.

## Decision rule (frozen)

Computed on `S_oracle_C2` pooled over all completed problems:

- **KILL**: `S_oracle_C2 < 0.10`; or C2 is a minority of observed exact redundancy and the rest is
  C1/C3; or a search-level dedup of creation duplicates (retaining multiplicity as weight) removes
  at least as much compute as the runtime oracle.
- **GRAY**: `0.10 <= S_oracle_C2 < 0.20`.
- **GO**: `S_oracle_C2 >= 0.20`, search semantics preserved, and not explained by C1.
- **STRONG GO**: `S_oracle_C2 >= 0.30` under the same conditions.

No config other than the official one above is used for the verdict.

## Falsification tests (run before believing any number)

1. Instrumentation is inert: the traced sampler and the untouched original produce identical final
   token ids, scores, `nfe`, `svf_calls` on the same seed for the smoke problems.
2. Hash = identity: for every pair of forwards with equal `h_fwd`, the stored canvases are compared
   element-wise (zero tolerated collisions).
3. Hash → identical computation: whenever two rows with equal `h_fwd` sit in the same forward
   chunk, their logits rows are compared with `torch.equal`; any mismatch is reported (this also
   tests batch-invariance of the MPS kernels, which a sharing runtime would rely on).
4. No same-path counting: C3 and parent/child identity are excluded by construction; the C1/C2
   split uses the reconstructed branch tree, and the tree is checked for consistency
   (every row at event t has exactly one parent at event t-1).
5. Only executed forwards count: rows removed at pruning have no forwards after removal, and the
   denominator is the executed set.
6. Prism's own dedup (survivor key) is not counted as our saving: it only affects selection; we
   count forwards actually run.
7. FLOP vs wall-clock: both reported; the verdict uses FLOPs.
8. Divergence dynamics: for each C1/C2 group, whether the rows are still `h_fwd`-equal at the next
   event (measures how quickly exact equality is destroyed by the temperature-0.7 sampler).
