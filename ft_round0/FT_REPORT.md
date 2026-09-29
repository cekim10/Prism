# FT Round 0 report: survival-aware protection of beam-search KV state (FastTTS)

Run 2026-09-28 on one 44 GB GPU (vLLM 0.9.2, bf16), analysed against the rule frozen in
`FT_PREREG.md`. Raw outputs in `out/{aime24,math500}/` (traces, `res_*.jsonl`, summaries).

## Verdict: KILL (frozen "cheap recompute" clause)

The strict oracle clears the 30% byte-time gate (35.7% AIME24, 37.8% MATH-500, seconds-weighted),
but the state it declines to protect is small and dies throughout the run rather than late:
median unique KV of a dying lineage is 85 tokens (rule: >= 256), and 44% of the unprotected
byte-time comes from deaths in the second half of the run (rule: >= 50%). Both frozen conditions
for "FT candidate" fail; the cheap-recompute clause fires. The user's stated stance was to close
the FT-search idea if FastTTS did not clear the gate rather than hunt for a third workload.

## Setup (as preregistered)

FastTTS `ihc-fan-lab/FastTTS@1c01d54`, Qwen2.5-Math-1.5B-Instruct generator + Skywork-o1-Open-PRM
1.5B verifier, repo-default beam search (`n=8, beam_width=4`, keep 2 per iteration, 40 iterations
max, T=0.8, stop `\n\n`), vanilla semantics (spec beam extension / prefix-aware scheduling off).
The Skywork PRM architecture is not in vLLM 0.9.2; a plugin (`prm_plugin/`) registers it with
the checkpoint's value head. Primary: AIME 2024, all 30 problems. Secondary: MATH-500 0-49.

| | AIME24 | MATH-500 |
|---|---|---|
| problems / iterations per problem (mean, range) | 30 / 11.3 (5-32) | 50 / 11.5 (4-40) |
| completed candidates per problem | 8-27 | similar |
| wall-clock per problem | 0.1-0.5 min | 0.0-0.4 min |
| pred accuracy (boxed match, descriptive) | 3/30 | 35/50 |

## Primary metrics

| metric | AIME24 | MATH-500 | gate |
|---|---|---|---|
| `W_protect` strict oracle, seconds-weighted | **35.7%** (per-problem mean 37.6% +-4.5) | 37.8% (39.2% +-2.7) | >= 30% pass |
| `W_protect` strict, iteration-weighted | 32.7% | 32.9% | |
| `W_answer` (answer-preserving, descriptive) | 60.9% | 56.0% | |
| `late_share` (unprotected byte-time with death at >= 50% of run) | 43.8% | 35.8% | >= 50% **fail** |
| dying lineage unique tokens: median | **85** | 59 | >= 256 **fail** |
| dying lineage unique tokens: byte-time-weighted median | 139 | 105 | |
| byte-time-weighted mean of unique / context | 0.26 | 0.24 | >= 0.20 pass |
| share of unprotected byte-time from lineages >= 256 unique tokens | 23% | 13% | |
| death-iteration histogram (quarters of run) | 526 / 429 / 444 / 302 | 887 / 698 / 738 / 493 | |

Absolute scale at 28 KiB/token (1.5B): uniform protected byte-time is 12.9 GiB x iterations over
30 AIME problems, i.e. ~0.43 GiB x iteration per problem with iterations of 0.1-0.5 s; a typical
beam context is ~640 tokens = 18 MB, a typical dying lineage 85 tokens = 2.3 MB (4.7 MB at 7B).

## What the tree looks like

Each iteration spawns 8 candidates from 2 survivors; 6 die at the end of the same iteration
with one reasoning step (often 8-60 tokens) of unique KV. Lineages that survive a prune and die
later exist (23% of unprotected byte-time is in lineages with >= 256 unique tokens) but are the
minority, and the two survivors are usually siblings, so the divergent, uniquely-owned part of a
losing lineage stays short. Deaths are spread almost uniformly over the run (slightly
front-loaded), which is why `late_share` sits just under one half.

## Online predictability (descriptive)

P(a kept beam's lineage is kept again at the next prune) = 0.750 observed vs 0.795 chance on
AIME24 (0.729 vs 0.791 on MATH-500, chance computed per event from the number of non-completed
children). PRM rank at iteration i carries no information about survival at i+1 beyond chance in
this workload, so even the 36% oracle would not be approachable by a score-based policy.

## Falsification results

1. Traced vs original: `pred`, all completions, token counts and completion counts identical on
   the 2 test problems; PRM scores drift by up to 7.5e-3. Two *original* runs with the same seed
   also drift (p0 exact, p1 up to 1.18e-2), so the drift is verifier nondeterminism in the
   vLLM stack, not the tracing. Search path reproduced exactly; scores reported as degraded pass
   (prereg item 1).
2. Tree consistency: 0 inconsistent parents, 0 bad active counts, 0 bad kept counts (80 problems).
3. Unique-KV accounting: lineage token sums equal `total_completion_tokens` for every completed
   beam (0 mismatches).
4. Residency stops at the subtree's last active iteration; completed beams accrue nothing after
   completion (by construction).
5. Iteration- and seconds-weighted numbers both reported; verdict uses seconds.

## Three biggest caveats

1. The 1.5B generator solves 3/30 AIME problems; on AIME the search is mostly ranking noise.
   MATH-500 (35/50) gives the same numbers, so the conclusion does not hinge on this, but a 7B
   generator with longer steps could shift the unique-token distribution upward.
2. Absolute headroom is tiny at this scale: the whole per-problem protected state is tens of MB
   for tenths of a second. Ratios transfer to larger models, but the premise that protection is
   expensive needs 7B+ models and long contexts, which were not run.
3. FastTTS's own optimizations were disabled to keep the tree exact; they alter scheduling and
   speculative extension, not which beams are pruned, but their effect on byte-time was not
   measured.

## Does this justify a system implementation?

No. The oracle saving exists (36%) but consists of many short-lived, single-step KV fragments
(median 2.3 MB, dying uniformly through the run) that are cheaper to regenerate than to protect,
and the search's own scores give no early signal of which lineages will die.
