"""
Driver for the DACL ablation suite (A/B/C/D) and the pipeline-stage ablation
suite (PIPELINE_ABLATION_CONFIGS: no_curriculum_maml, no_warmup_dacl,
floor_no_warmup_no_curriculum). --labels accepts keys from either dict.

Pipeline-label runs write to results/plots/pipeline_ablation_comparison.*
instead of ablation_comparison.*, so they never overwrite the existing
merged A/B/C/D comparison.

This does NOT modify any DACL component logic, model, dataset, or the
zero-shot protocol in train_xnli.py. It only fixes two operational issues
in the stock `run_ablation_suite()`:

  1. Directory isolation: the stock function reuses config.checkpoint_dir /
     results_dir / plots_dir ('./checkpoints', './results', './plots') for
     every ablation config, which would overwrite the main run's
     best_warmup_model.pt / best_meta_model.pt / results/final_results.json.
     Here each config gets its own checkpoints_ablation_<label> /
     results_ablation_<label> / plots_ablation_<label> directory, mirroring
     the existing run_multiple_seeds() pattern for per-seed directories.

  2. Disk footprint: the shared machine has very little free disk. Stock
     checkpointing writes a new ~1.1GB meta_episode_<N>.pt snapshot every
     save_every_n_episodes (8 files for a 400-episode full run) plus
     best_warmup_model.pt and best_meta_model.pt. We monkeypatch
     CMLTrainer.save_checkpoint to skip writing the numbered periodic
     snapshot file (the periodic *evaluation* that feeds training_history /
     best_meta_model.pt tracking still runs exactly as before -- only the
     redundant extra .pt file on disk is skipped). After each config
     finishes and its results are captured into the ablation JSON/CSV, that
     config's checkpoint .pt files are deleted to free disk before the next
     config starts.

The final results/ablation_comparison.csv, results/ablation_comparison.json,
and plots/ablation_comparison.png are written to the MAIN results_dir /
plots_dir (./results, ./plots) as specified in the task, not the per-label
directories.
"""
import argparse
import json
import os
import shutil
import time

import train_xnli as tx

MAIN_RESULTS_DIR = './results'
MAIN_PLOTS_DIR = './plots'
MIN_FREE_GB_TO_START = 2.5

# A/B/C/D vary the curriculum mechanism inside meta-training (see
# ABLATION_CONFIGS). PIPELINE_ABLATION_CONFIGS vary whether entire pipeline
# stages (warmup, curriculum) run at all -- a coarser, separate axis. Both
# are addressable through the same --labels flag; the two dicts don't share
# keys.
ALL_ABLATION_CONFIGS = {**tx.ABLATION_CONFIGS, **tx.PIPELINE_ABLATION_CONFIGS}


def free_gb(path='.'):
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


_original_save_checkpoint = tx.CMLTrainer.save_checkpoint


def _thin_save_checkpoint(self, filename, **kwargs):
    if filename.startswith('meta_episode_') and filename.endswith('.pt'):
        return None
    return _original_save_checkpoint(self, filename, **kwargs)


def install_thin_checkpointing():
    tx.CMLTrainer.save_checkpoint = _thin_save_checkpoint


def cleanup_label_checkpoints(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith('.pt'):
            fpath = os.path.join(checkpoint_dir, fname)
            os.remove(fpath)
            print(f"  [cleanup] removed {fpath}")


def run_suite(debug_mode, labels, seed=42):
    install_thin_checkpointing()

    os.makedirs(MAIN_RESULTS_DIR, exist_ok=True)
    os.makedirs(MAIN_PLOTS_DIR, exist_ok=True)

    ablation_results = {}
    ablation_compute_cost = {}
    ablation_timing = {}

    for label in labels:
        if label not in ALL_ABLATION_CONFIGS:
            raise KeyError(
                f"Unknown ablation label {label!r}. Valid labels: "
                f"{sorted(ALL_ABLATION_CONFIGS)}"
            )
        overrides = ALL_ABLATION_CONFIGS[label]
        print("\n" + "=" * 70)
        print(f"ABLATION {label}: {overrides}  (debug_mode={debug_mode}, seed={seed})")
        print("=" * 70)

        free = free_gb()
        print(f"[disk] {free:.2f} GB free before config {label}")
        if free < MIN_FREE_GB_TO_START:
            raise RuntimeError(
                f"Aborting before config {label}: only {free:.2f} GB free "
                f"(< {MIN_FREE_GB_TO_START} GB threshold). Not safe to start "
                f"a run that saves ~1GB checkpoints on this shared machine."
            )

        label_ckpt_dir = f'./checkpoints_ablation_{label}'
        label_results_dir = f'./results_ablation_{label}'
        label_plots_dir = f'./plots_ablation_{label}'
        for d in (label_ckpt_dir, label_results_dir, label_plots_dir):
            os.makedirs(d, exist_ok=True)

        tx.config.checkpoint_dir = label_ckpt_dir
        tx.config.results_dir = label_results_dir
        tx.config.plots_dir = label_plots_dir

        for key, value in overrides.items():
            setattr(tx.config, key, value)

        tx.config.seed = seed
        tx.set_seed(seed)

        t0 = time.time()
        trainer, final_results, baseline_results = tx.run_complete_experiment(debug_mode=debug_mode)
        elapsed = time.time() - t0

        ablation_results[label] = final_results
        ablation_compute_cost[label] = trainer.training_history.get('compute_cost', {})
        ablation_timing[label] = elapsed

        print(f"\n[timing] Config {label} took {elapsed/60:.1f} minutes")

        # Free disk before starting the next config.
        cleanup_label_checkpoints(label_ckpt_dir)
        free_after = free_gb()
        print(f"[disk] {free_after:.2f} GB free after cleaning up config {label}")

    # ------------------------------------------------------------------
    # Comparison CSV/JSON/plot, written to the MAIN results/plots dirs.
    # ------------------------------------------------------------------
    tx.config.checkpoint_dir = './checkpoints'
    tx.config.results_dir = MAIN_RESULTS_DIR
    tx.config.plots_dir = MAIN_PLOTS_DIR

    rows = []
    for lang in tx.config.all_languages + ['average']:
        row = {'Language': lang}
        for label in labels:
            row[label] = ablation_results[label].get(lang, None)
        rows.append(row)
    df = tx.pd.DataFrame(rows)

    # Pipeline-stage labels (PIPELINE_ABLATION_CONFIGS) are a different axis
    # from the A/B/C/D curriculum-mechanism labels and must not overwrite the
    # existing results/ablation_comparison.csv|json, which already holds the
    # merged A/B/C/D comparison.
    is_pipeline_run = any(label not in tx.ABLATION_CONFIGS for label in labels)
    comparison_stem = 'pipeline_ablation_comparison' if is_pipeline_run else 'ablation_comparison'

    csv_path = os.path.join(MAIN_RESULTS_DIR, f'{comparison_stem}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")

    json_path = os.path.join(MAIN_RESULTS_DIR, f'{comparison_stem}.json')
    tx.save_json({
        'configs': {label: ALL_ABLATION_CONFIGS[label] for label in labels},
        'results': ablation_results,
        'compute_cost': ablation_compute_cost,
        'wall_clock_seconds': ablation_timing,
        'debug_mode': debug_mode,
        'seed': seed,
    }, json_path)

    if len(labels) > 1:
        visualizer = tx.ResultsVisualizer(tx.config)
        visualizer.plot_ablation_comparison(ablation_results, save_name=f'{comparison_stem}.png')

    print("\n" + df.to_string(index=False))
    return ablation_results, ablation_compute_cost, ablation_timing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['debug', 'full'], required=True)
    parser.add_argument('--labels', default='A,B,C,D')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    labels = args.labels.split(',')
    debug_mode = (args.mode == 'debug')

    results, costs, timing = run_suite(debug_mode=debug_mode, labels=labels, seed=args.seed)
