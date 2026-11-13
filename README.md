# DAGCML: Dynamic Alignment-Gated Curriculum Meta-Learning for Zero-Shot Cross-Lingual NLI

Research code for **DACL** (Dynamic Alignment-gated Curriculum Learning). DACL is a curriculum meta-learning method for zero-shot cross-lingual natural language inference on [XNLI](https://huggingface.co/datasets/facebook/xnli), built on `microsoft/mdeberta-v3-base`.

The pipeline follows the warmup-then-meta-train setup of Doan & Le (2024), *Curriculum meta-learning for zero-shot cross-lingual transfer*:

1. **Warmup.** Fine-tune mDeBERTa-v3-base on the English XNLI training set.
2. **Curriculum meta-training.** Run first-order MAML on tasks drawn from the Russian and Greek XNLI dev sets.
3. **Zero-shot evaluation.** Evaluate on the test sets of all 15 XNLI languages.

DACL replaces the base method's static multi-teacher cross-review curriculum with three components, each of which can be switched on or off independently:

| Component | What it does | Config flag |
|---|---|---|
| 1. Self-paced difficulty | Teacher-free difficulty score from the meta-model's own loss and predictive entropy, optionally re-scored during training | `curriculum_mode`: `cross_review` \| `dacl_static` \| `dacl_dynamic` \| `none` |
| 2. Competence scheduling | Restricts which difficulty windows can be sampled as training progresses (Platanios et al., 2019 style) | `window_scheduling`: `random` \| `competence`; `curriculum_direction`: `hard_first` \| `easy_first` |
| 3. Alignment gating | Scales each meta-update by its gradient alignment with a small English probe set | `alignment_gating`: `True` \| `False` |

See [`NOVELTY.md`](NOVELTY.md) for the method description and equations.

> **Status:** the methodology and evaluation protocol are under revision. The files in `results/` come from the original single-seed (42) runs and will be regenerated with the revised protocol.

## Repository structure

```
├── train_xnli.py                  # End-to-end pipeline: data, warmup, curriculum, FOMAML, evaluation, plots
├── curriculum_dacl.py             # DACL components: difficulty estimator, competence scheduler, alignment gate
├── run_ablation.py                # CLI driver for the A/B/C/D and pipeline-stage ablations
├── inference.py                   # Load a checkpoint; single/batch prediction and dataset evaluation
├── finish_from_checkpoint.py      # Recovery: run final evaluation from saved checkpoints
├── merge_ablation_results.py      # Merge per-config ablation results into one table
├── merge_pipeline_ablation_results.py
├── debug_backbone.py              # Backbone sanity checks
├── client/                        # Streamlit demo (Hugging Face Space)
├── results*/                      # Metrics JSON/CSV and per-episode training histories
├── plots*/                        # Generated figures
├── checkpoints/                   # Model weights (git-ignored)
├── NOVELTY.md                     # Method description
└── requirements.txt
```

## Setup

Requires Python 3.10+ (developed with 3.12) and a CUDA GPU. Training runs in fp32, because mDeBERTa's disentangled attention overflows in fp16.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # pick the wheel for your CUDA version
pip install -r requirements.txt
```

The XNLI data (`facebook/xnli`) and the backbone download automatically from the Hugging Face Hub on first run.

## Usage

### Full pipeline (warmup + DACL meta-training + evaluation)

```bash
python train_xnli.py
```

Hyperparameters live in the `Config` class at the top of `train_xnli.py`. The run writes:
- `checkpoints/`: `best_warmup_model.pt`, `best_meta_model.pt`, `meta_episode_<N>.pt`
- `results/`: `baseline_results.json` (after warmup), `final_results.json`, `results_table.csv`, `training_history.json`
- `plots/`: training curves, per-language comparison, difficulty drift, alignment statistics

For a quick smoke test on a reduced dataset with 10 episodes, call `quick_test()` from `train_xnli.py`.

### Ablations

```bash
# Component ablation: A = cross-review, B = static self-scored difficulty,
# C = dynamic difficulty + competence scheduling, D = full DACL
python run_ablation.py --mode full --labels A,B,C,D --seed 42

# Pipeline-stage ablation
python run_ablation.py --mode full --labels no_curriculum_maml,no_warmup_dacl,floor_no_warmup_no_curriculum

# Smoke test
python run_ablation.py --mode debug --labels D
```

Each configuration writes to its own `checkpoints_ablation_<label>/`, `results_ablation_<label>/` and `plots_ablation_<label>/` folders. The combined tables go to `results/ablation_comparison.*` and `results/pipeline_ablation_comparison.*`.

### Multiple seeds

`run_multiple_seeds()` in `train_xnli.py` reruns the pipeline for seeds `{42, 26, 11, 2000, 10}` and writes the mean and standard deviation for each language.

### Inference

```python
from inference import NLIInference

nli = NLIInference("checkpoints/best_meta_model.pt", device="cuda")
nli.predict_single("A man is playing a guitar on stage.",
                   "A person is performing music.", return_probs=True)
```

Checkpoints store a pickled `Config` object from `train_xnli.py`. If you load one from your own script and get `Can't get attribute 'Config'`, register the class first:

```python
import __main__, train_xnli
__main__.Config = train_xnli.Config
```

### Streamlit demo

```bash
cd client && streamlit run app.py
```

The demo downloads trained weights from the Hugging Face Hub (`hmzi67/mdeberta-xnli`). See [`client/README.md`](client/README.md).

## Citation

The paper is under review; citation details will be added on publication.

This work builds on:
- Doan & Le (2024). *Curriculum meta-learning for zero-shot cross-lingual transfer.* Knowledge-Based Systems.
- He et al. (2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing.* ICLR.
- Conneau et al. (2018). *XNLI: Evaluating cross-lingual sentence representations.* EMNLP.
- Platanios et al. (2019). *Competence-based curriculum learning for neural machine translation.* NAACL.

## License

[MIT](LICENSE)
