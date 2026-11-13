"""
Recovery script: finish STEP 5-7 of run_complete_experiment() using
checkpoints that already exist on disk, without redoing warmup or
meta-training.

Background: the original run completed warmup + all meta-episodes and
saved best_meta_model.pt / meta_episode_400.pt correctly, but then
crashed while loading best_meta_model.pt for the final evaluation due
to a path-doubling bug in CMLTrainer.load_checkpoint (now fixed in
train_xnli.py). This script picks up right where that run left off.

Run with: python finish_from_checkpoint.py
"""

import os
import sys
from train_xnli import (
    config, Config, DatasetManager, CMLTrainer, ResultsVisualizer, save_json,
)

# The checkpoints were saved while train_xnli.py ran as __main__, so pickle
# recorded the checkpoint's 'config' value as __main__.Config. Since this
# script is now __main__, register the class here so torch.load can resolve
# it.
sys.modules['__main__'].Config = Config

BEST_META_CKPT = 'best_meta_model.pt'
LAST_META_CKPT = 'meta_episode_400.pt'
BASELINE_RESULTS_PATH = os.path.join(config.results_dir, 'baseline_results.json')


def main():
    print("=" * 70)
    print("RESUMING: STEP 5-7 ONLY (no warmup / no meta-training)")
    print("=" * 70)

    # STEP 1: Load datasets (needed for evaluate_all_languages)
    dataset_manager = DatasetManager(config)
    dataset_manager.load_all_datasets()

    # STEP 2: Build trainer (fresh model; weights come from checkpoints below)
    trainer = CMLTrainer(config, dataset_manager)

    # Recover the full training history (through episode 400) from the
    # last meta-episode checkpoint, since the in-memory history was lost
    # when the process crashed.
    last_ckpt_path = os.path.join(config.checkpoint_dir, LAST_META_CKPT)
    if os.path.exists(last_ckpt_path):
        full_history = trainer.load_checkpoint(LAST_META_CKPT)
        trainer.training_history = full_history.get('training_history', trainer.training_history)
        print(f"✓ Recovered full training_history from {LAST_META_CKPT}")
    else:
        print(f"⚠ {LAST_META_CKPT} not found — training_history/plots will be incomplete")

    # Reuse the baseline results already saved during this run's warmup step
    if not os.path.exists(BASELINE_RESULTS_PATH):
        raise FileNotFoundError(
            f"Expected baseline results at {BASELINE_RESULTS_PATH} but none found. "
            "Cannot recover without redoing warmup."
        )
    with open(BASELINE_RESULTS_PATH) as f:
        import json
        baseline_results = json.load(f)
    print(f"✓ Loaded existing baseline results from {BASELINE_RESULTS_PATH}")

    # Load best_meta_model.pt weights for final evaluation (matches what
    # config.select_best_meta_checkpoint intends).
    best_meta_path = os.path.join(config.checkpoint_dir, BEST_META_CKPT)
    if config.select_best_meta_checkpoint and os.path.exists(best_meta_path):
        print(f"\n[Best checkpoint] Loading {BEST_META_CKPT} for final evaluation...")
        trainer.load_checkpoint(BEST_META_CKPT)
    else:
        print(f"\n[Best checkpoint] Using {LAST_META_CKPT} weights (last episode)")
        trainer.load_checkpoint(LAST_META_CKPT)

    # STEP 5: Final Evaluation
    print("\n" + "=" * 70)
    print("STEP 5: FINAL ZERO-SHOT EVALUATION")
    print("=" * 70)

    final_results = trainer.evaluate_all_languages()

    final_path = os.path.join(config.results_dir, 'final_results.json')
    save_json(final_results, final_path)

    history_path = os.path.join(config.results_dir, 'training_history.json')
    save_json(trainer.training_history, history_path)

    # STEP 6: Visualizations
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 70)

    visualizer = ResultsVisualizer(config)
    visualizer.plot_training_curves(trainer.training_history)
    visualizer.plot_language_comparison(final_results, baseline_results)
    visualizer.create_results_table(final_results, baseline_results)
    visualizer.plot_difficulty_drift(trainer.training_history)
    visualizer.plot_alignment_stats(trainer.training_history)

    # STEP 7: Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\nBaseline (after warmup):")
    print(f"  Average Accuracy: {baseline_results.get('average', 0) * 100:.2f}%")

    print(f"\nFinal (after meta-learning"
          f"{', best_meta_model.pt' if config.select_best_meta_checkpoint else ', last episode'}):")
    print(f"  Average Accuracy: {final_results.get('average', 0) * 100:.2f}%")

    improvement = (final_results.get('average', 0) - baseline_results.get('average', 0)) * 100
    print(f"\nImprovement: {improvement:+.2f}%")

    print(f"\n✓ All results saved to: {config.results_dir}")
    print(f"✓ All plots saved to: {config.plots_dir}")
    print("\n" + "=" * 70)
    print("RECOVERY COMPLETE!")
    print("=" * 70)

    return trainer, final_results, baseline_results


if __name__ == "__main__":
    main()
