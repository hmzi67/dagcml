# DACL: Dynamic Alignment-gated Curriculum Learning

This document describes the novelty added on top of Doan & Le (2024, Knowledge-Based
Systems), "Curriculum meta-learning for zero-shot cross-lingual transfer" (the base
method implemented in `CurriculumGenerator` / `MAML` / `CMLTrainer.meta_train_with_curriculum`
in `train_xnli.py`, kept intact as the `cross_review` ablation baseline). It is the
source material for rewriting Section 3.2 of `main.tex`.

Implementation lives in `curriculum_dacl.py` (`SelfPacedDifficultyEstimator`,
`CompetenceScheduler`, `AlignmentGate`) and is wired into `train_xnli.py` behind config
flags (`curriculum_mode`, `window_scheduling`, `alignment_gating`) so every component can
be ablated independently.

## Problem with the baseline mechanism

`CurriculumGenerator.cross_review_evaluate` trains `n_teachers` separate models on
disjoint shards of each auxiliary-language dataset and cross-evaluates them to get a
difficulty score per sample. This is:

1. **Not novel** — a direct re-implementation of the paper's Section 3.2.1.
2. **Expensive** — `n_teachers` full model trainings (`teacher_epochs` each) plus
   `n_teachers * (n_teachers - 1)` cross-evaluation passes, per auxiliary language.
3. **Static** — difficulty is computed once before meta-training and never revisited,
   even though the notion of "difficulty" for a sample is relative to the current model
   and should drift as the meta-model improves.

DACL replaces this with three independently-switchable components.

## Component 1 — Teacher-free dynamic difficulty estimation

Replaces multi-teacher cross-review with a self-paced score computed by the *current*
meta-model itself, in a single `torch.no_grad()` forward pass, optionally re-estimated
periodically during meta-training:

```
difficulty_j(t) = lambda_loss * CE(f_theta_t(x_j), y_j) + lambda_ent * H(softmax(f_theta_t(x_j)))
```

where `H` is Shannon entropy of the predicted class distribution, `theta_t` is the
current meta-model at re-scoring time `t`, and `lambda_loss` / `lambda_ent` are config
weights (defaults 1.0 / 0.5). Both terms are z-normalized (zero mean, unit variance)
across the dataset before combining so the scales are comparable.

- `curriculum_mode = 'dacl_static'`: score computed once, like the cross-review baseline
  but teacher-free (ablates cost, isolates the effect of the estimator itself).
- `curriculum_mode = 'dacl_dynamic'`: re-scored every `rescore_every_n_episodes` episodes
  (default 50); the annealed buckets/windows are rebuilt after each re-scoring. The
  Spearman rank correlation between the previous and new difficulty ranking is logged to
  `training_history['difficulty_drift']` — a non-1.0 (and typically decreasing) value
  demonstrates that difficulty is non-stationary, which is the empirical justification
  for the dynamic approach over the static baseline.

Implemented by `SelfPacedDifficultyEstimator.score()` in `curriculum_dacl.py`, called
from `CMLTrainer._generate_curriculum_dacl` and re-invoked inside the episode loop of
`CMLTrainer.meta_train_with_curriculum` for the dynamic case.

## Component 2 — Competence-based window scheduling

`meta_train_with_curriculum` originally called `random.choice(task_data['annealed_batches'])`
every episode, discarding the easy-to-hard ordering the annealing arrangement produces.
DACL replaces this with a competence-based scheduler adapted from Platanios et al. (2019),
"Competence-based Curriculum Learning for Neural Machine Translation", applied here to
meta-task window selection instead of per-sentence NMT batching:

```
competence(e) = min(1.0, sqrt(e * (1 - c0^2) / n_episodes + c0^2))
```

with `c0 = initial_competence` (default 0.1) and `e` the current episode. At episode `e`,
only annealed windows whose maximum difficulty percentile is `<= competence(e)` are
eligible; the window is then sampled uniformly among eligible windows. Since windows are
built from `window_step` contiguous difficulty buckets out of `n_buckets` total, window
`i`'s max percentile is `min(1, (i + window_step) / n_buckets)`. At `e = n_episodes`,
`competence = 1.0` and every window is eligible.

Implemented by `CompetenceScheduler` in `curriculum_dacl.py`. Selected window index and
competence value are logged per episode to `training_history['curriculum_schedule']`.
Controlled by `window_scheduling = 'competence' | 'random'`.

## Component 3 — Alignment-gated meta-updates

The strongest component. In low-resource meta-learning a noisy or unusually hard task
batch can produce a query gradient that points away from the generalization direction,
degrading `theta`. The base paper only mitigates this indirectly via curriculum
ordering. DACL gates each meta-update by its alignment with a probe gradient computed on
a small, fixed held-out subset of the *high-resource* (English) validation set — never
any target or auxiliary language — so the zero-shot protocol is fully preserved.

Per meta-episode:

1. A fixed probe set (`probe_size = 64` examples) is sampled once from `en_val` with the
   global seed at the start of meta-training and excluded from any other use.
2. After the existing per-task `backward()` accumulation, the gradient of the
   classifier-head parameters only (names containing `classifier` or `pooler`,
   discovered and logged once at `AlignmentGate` construction) is extracted from
   `self.model.parameters()` — head-only keeps this memory-cheap.
3. The probe gradient of the classifier head is computed separately via
   `torch.autograd.grad(..., create_graph=False, retain_graph=False)` on the probe set,
   and cached; it is recomputed only every `probe_refresh_every` episodes (default 10) to
   bound the extra cost.
4. Cosine similarity `s` between the flattened task head-gradient and the cached probe
   head-gradient gates the meta-update by scaling the entire model's accumulated
   gradient before `meta_optimizer.step()`:

```
gate = clamp((1 + s) / 2, gate_min, 1.0)
for p in model.parameters():
    if p.grad is not None: p.grad.mul_(gate)
```

with `gate_min = 0.1` so no update is ever fully zeroed. Implemented by `AlignmentGate`
in `curriculum_dacl.py`; the scaling itself happens in `MAML.meta_update` (kept out of
`AlignmentGate` so the class stays a pure, testable helper). Per-episode cosine
similarity, gate, and meta-loss are logged to `training_history['alignment_stats']`.
Controlled by `alignment_gating = True | False`. No `create_graph=True` is introduced
anywhere — the probe gradient computation follows the same first-order style as the
existing `MAML.inner_loop`.

## Ablation table skeleton

Filled in by `run_ablation_suite()` (writes `results/ablation_comparison.csv` /
`.json` and `plots/ablation_comparison.png`). Populate after a full GPU run — the
numbers below are placeholders.

| Config | curriculum_mode | window_scheduling | alignment_gating | en | fr | es | de | el | bg | ru | tr | ar | vi | th | zh | hi | sw | ur | Average |
|--------|-----------------|--------------------|-------------------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|---------|
| A (Doan & Le baseline) | cross_review | random | False | | | | | | | | | | | | | | | | |
| B (difficulty estimator alone) | dacl_static | random | False | | | | | | | | | | | | | | | | |
| C (dynamic curriculum) | dacl_dynamic | competence | False | | | | | | | | | | | | | | | | |
| D (full DACL) | dacl_dynamic | competence | True | | | | | | | | | | | | | | | | |

Compute-cost accounting (wall-clock + forward/backward passes consumed by curriculum
generation) is recorded per config/language in the same JSON, under `compute_cost`. The
expected result is that `dacl_static`/`dacl_dynamic` remove the `n_teachers x
teacher_epochs` full training runs and `n_teachers x (n_teachers - 1)` cross-evaluation
passes entirely, replacing them with a single forward pass over each auxiliary dataset
(plus, for `dacl_dynamic`, one extra forward pass per aux language every
`rescore_every_n_episodes` episodes) — this is the quantitative "resource-efficient"
claim for the paper.

## Positioning vs. Doan & Le (2024)

| Aspect | Doan & Le (2024) | DACL |
|---|---|---|
| Difficulty scoring | Static, multi-teacher cross-review (`n_teachers` trained models) | Teacher-free, self-paced, using the current meta-model |
| When scored | Once, before meta-training | Once (`dacl_static`) or periodically inside the meta-loop (`dacl_dynamic`) |
| Window sampling | Uniform random over annealed windows | Competence-gated, easy-to-hard progression |
| Meta-update weighting | Uniform (implicit trust in curriculum ordering) | Gated by alignment with a high-resource probe gradient |

## Related work

Gradient-agreement methods for meta-learning (e.g. GA-MAML-style approaches) weight or
filter task gradients by their pairwise cosine agreement to reduce interference between
conflicting tasks during meta-training. DACL's Component 3 is in this family but differs
in three ways relevant to zero-shot cross-lingual transfer: (a) the reference signal is a
*probe* gradient from a small fixed high-resource-language set rather than agreement
among the sampled tasks themselves, which keeps the gate cheap (no pairwise comparisons)
and keeps target-language data fully out of the training loop, preserving the zero-shot
evaluation protocol; (b) gating is restricted to the classifier head gradient for the
alignment computation (though the resulting scalar gate is applied to the whole model),
keeping the extra memory/compute cost compatible with first-order MAML on a single 16GB
GPU; (c) it composes with a dynamic, teacher-free curriculum rather than operating on a
fixed task distribution. Platanios et al. (2019), "Competence-based Curriculum Learning
for Neural Machine Translation," is the nearest prior work for Component 2; DACL adapts
their per-sentence competence formula to gate which *annealed difficulty window* is
eligible for meta-task sampling at each episode, rather than gating individual NMT
training examples, and composes it with a dynamically re-scored curriculum rather than a
fixed, precomputed difficulty ranking.
