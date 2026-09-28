# Round 0 report: exact post-divergence state convergence in Prism search

Run 2026-09-28 on one NVIDIA GPU (bf16, no quantisation), analysed against the rule frozen in
`ROUND0_PREREG.md`. Raw outputs: `out/gsm8k_official/{summary.json,res.jsonl,run.log,traces/}`.

## Verdict: KILL

`S_oracle_C2 = 4.0%` (preregistered KILL threshold: `< 10%`). Two further KILL clauses also fire:
most observed exact redundancy is creation duplicates (C1 11.1%) rather than convergence, and a
search-level dedup of creation duplicates alone removes more compute than the convergence oracle.

## Exact state definition (as frozen)

Forward input of a row = `(problem_id, canvas x[b, :])`; full trajectory state adds the
per-position confidence `conf_scores[b, :]` (consumed only at branch events) and `(block, step)`.
`h_fwd = blake2b(canvas)`, `h_full = blake2b(canvas, conf)`. A merge is admitted only between rows
with equal `h_fwd` at the same `(kind, block, step)` event; it is C2 (post-divergence) only if the
rows' `h_full` differed at some common event after their lowest common ancestor.

## Scale

| | |
|---|---|
| Problems | 30 (GSM8K test, idx 0-29, official `run_gsm8k.sh` config) |
| Trajectories | 16 initial per problem, pruned 16 -> 9 -> 4 in the first block |
| Executed row-forwards (denoise + rough) | 18,208 |
| Verifier forwards | 870 (5.2% of FLOPs) |
| Mean NFE / SVF per problem | 607 / 29 |
| Wall-clock per problem | 0.3-0.6 min |

## T1 exact redundancy

| Metric | Value |
|---|---|
| `R_state` = 1 - unique/executed states | 16.1% |
| `R_compute` over denoise+rough forwards (C1 + C2) | 16.0% |
| `S_oracle_C1` creation duplicates (FLOP / latency weighted) | 11.1% / 11.9% |
| **`S_oracle_C2` true post-divergence convergence (FLOP / latency)** | **4.0% / 4.0%** |
| per-problem `S_C2`: mean, 95% CI, max | 4.2%, +-0.9 pp, 9.4% |
| C3 same-row rough/denoise recompute | 0.6% |
| Verifier forwards with identical text at one prune event | 2.4% |

Descriptive, same-path, not headroom for the hypothesis: 24.4% of all FLOPs are denoise forwards
on rows whose current block is already fully decoded (Prism keeps forwarding finished rows until
the slowest row finishes the block); this is what the 26.4% "same `h_fwd` at a different event"
memoisation ceiling consists of.

## True post-divergence convergence: what it actually is

530 C2 groups (777 saved row-forwards). Of the 1,118 class pairs inside them, 1,079 have their
LCA at a prune/branch event and 39 at the root. Concretely:

- `_branch_and_resample` copies a survivor and re-masks 2 low-confidence tokens in the copy. In the
  next 2-5 steps the model re-decodes the same tokens (92% of transferred tokens equal the
  noise-free argmax at T=0.7), so the perturbed copy lands exactly on its unperturbed sibling.
  Median distance since LCA is 4-5 events; 81% of merged pairs are still identical one event later.
- The 39 root-LCA pairs are initial rows that unmasked the same tokens in a different order.
- Nothing else. No two rows descended from different survivors ever became exactly equal; no
  convergence was observed after block 6 (of 9-10 generated blocks).

Progress bins of `S_oracle_C2`: 0-25%: 3.98 pp, 25-50%: 0.06 pp, 50-75%: 0, 75-100%: 0. Mean
compute remaining after a merge event is 75%, but the merges are all within the first block(s),
where the `S -> A -> C`, `S -> B -> D -> X` pattern reduces to "the perturbation was undone".

## Duplicate-aware search baseline

- A (search-level prevention, weight-preserving dedup at creation): removes exactly the C1
  forwards = 11.1%, with identical search semantics (identical rows have identical logits).
- B (runtime consolidation of everything exact): C1 + C2 = 15.2%; the part attributable to
  convergence is 4.0%.

A alone beats the convergence oracle by ~3x, which is a preregistered KILL condition. Skipping
finished rows (a trivial scheduler fix, 24.4%) dwarfs both.

## Search-semantics / quality preservation

The oracle shares a forward only between rows with byte-identical `input_ids`. For every one of the
2,924 such pairs that co-occurred in a forward chunk, the returned logits were `torch.equal`
(max abs diff 0.0), so scores, selection, branch multiplicity, voting and final answers are
unchanged by construction. Official metric script on these 30 problems: voted accuracy 63.3%,
pass@4 83.3% (small n; not a claim about the paper's numbers).

## Falsification results

1. Traced sampler == original sampler: static diff of the two files has zero non-tracer
   differences; the dynamic bit-equality run (`equiv_test.py`) was **not executed** on the server
   (`out/equiv/` is empty). Pending; see below.
2. Hash = identity: 7,822 equal-hash canvas pairs compared element-wise, 0 collisions.
3. Hash -> identical computation: 2,924 / 2,924 duplicate pairs had `torch.equal` logits.
4. Branch tree consistent (0 inconsistencies); C3 and parent/child identity excluded.
5. Only executed forwards counted (denominator = all forwards actually run).
6. Prism's survivor-key dedup only affects selection; not counted.
7. FLOP-weighted 4.03% vs latency-weighted 4.02%.
8. Divergence dynamics: C2 pairs persist to the next event 397/486, C1 pairs 138/207.

## Three biggest caveats

1. Falsification test 1 is only established statically; run
   `python round0/equiv_test.py --idx 0 1 --gen_length 64 --no_quant --device cuda --out round0/out/equiv`
   (expects `ALL PASS`). A failure would invalidate the trace, but could not raise `S_C2`.
2. One model, one task, 30 problems. The CI is tight (+-0.9 pp) and the mechanism is structural
   (stochastic T=0.7 sampling never re-merges different-survivor lineages), but MATH500/HumanEval
   were not run because the primary is KILL.
3. The FLOP proxy charges the full forward per row; a real sharing runtime would still pay per-row
   sampling and bookkeeping, so 4.0% is an upper bound on the realisable saving.

## Does this justify a system implementation?

No: exact post-divergence convergence in Prism is a 4% upper-bound artefact of 2-token
perturbations being undone within a few steps, and the preregistered rule says KILL.
