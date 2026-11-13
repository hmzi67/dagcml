"""
Merge pre-existing, already-completed ablation runs A, B, C (found in
results_ablation_A/B/C/, dated Jul 23-24) with the already-completed run D
(currently the sole entry in results/ablation_comparison.json, dated Jul 27)
into a single combined results/ablation_comparison.json / .csv, and
regenerate plots/ablation_comparison.png with all four configs.

This does NOT retrain anything. A, B, C, and D were all produced by the same,
unmodified train_xnli.py (last modified before any of the four runs), with
config.seed defaulting to 42, so the four are directly comparable.
"""
import json
import os

import train_xnli as tx

MAIN_RESULTS_DIR = './results'
MAIN_PLOTS_DIR = './plots'

LABELS = ['A', 'B', 'C', 'D']

# Existing D data (already the correct combined-file format).
with open(os.path.join(MAIN_RESULTS_DIR, 'ablation_comparison.json')) as f:
    existing = json.load(f)

assert list(existing['results'].keys()) == ['D'], (
    f"Expected results/ablation_comparison.json to contain only D, found "
    f"{list(existing['results'].keys())}. Aborting to avoid clobbering "
    f"unexpected state."
)

ablation_results = {'D': existing['results']['D']}
ablation_compute_cost = {'D': existing['compute_cost']['D']}
ablation_timing = {'D': existing['wall_clock_seconds']['D']}
timing_notes = {
    'D': 'measured by run_ablation.py run_suite() timing wrapper (authoritative)'
}

for label in ['A', 'B', 'C']:
    label_dir = f'./results_ablation_{label}'
    with open(os.path.join(label_dir, 'final_results.json')) as f:
        ablation_results[label] = json.load(f)
    with open(os.path.join(label_dir, 'training_history.json')) as f:
        hist = json.load(f)
    ablation_compute_cost[label] = hist.get('compute_cost', {})
    ablation_timing[label] = None
    timing_notes[label] = (
        'not recorded: this run predates being executed through '
        'run_ablation.py run_suite() and has no timing wrapper. File mtimes '
        'only span the final eval phase (~80-90 min), not total training '
        'time, so no reliable wall-clock figure exists for this run.'
    )

# Sanity: recompute each config's average from its own per-language numbers
# and confirm it matches the stored average field.
for label in LABELS:
    res = ablation_results[label]
    langs = tx.config.all_languages
    recomputed = sum(res[lang] for lang in langs) / len(langs)
    stored = res.get('average')
    if stored is None or abs(recomputed - stored) > 1e-6:
        print(f"  [WARNING] config {label}: recomputed average {recomputed:.6f} "
              f"!= stored average {stored}")
    else:
        print(f"  [OK] config {label}: average {stored:.6f} verified from per-language numbers")

# ------------------------------------------------------------------
# Comparison CSV/JSON/plot, written to the MAIN results/plots dirs.
# ------------------------------------------------------------------
rows = []
for lang in tx.config.all_languages + ['average']:
    row = {'Language': lang}
    for label in LABELS:
        row[label] = ablation_results[label].get(lang, None)
    rows.append(row)
df = tx.pd.DataFrame(rows)

csv_path = os.path.join(MAIN_RESULTS_DIR, 'ablation_comparison.csv')
df.to_csv(csv_path, index=False)
print(f"\n✓ Saved: {csv_path}")

json_path = os.path.join(MAIN_RESULTS_DIR, 'ablation_comparison.json')
tx.save_json({
    'configs': {label: tx.ABLATION_CONFIGS[label] for label in LABELS},
    'results': ablation_results,
    'compute_cost': ablation_compute_cost,
    'wall_clock_seconds': ablation_timing,
    'wall_clock_notes': timing_notes,
    'debug_mode': False,
    'seed': 42,
    'merge_note': (
        'A, B, C were run to completion on 2026-07-23/24 (results_ablation_A/B/C) '
        'prior to this file existing; D was run on 2026-07-27 via run_ablation.py '
        'and originally written here alone. This file was produced by merging the '
        'four pre-existing completed runs, not by retraining any of them.'
    ),
}, json_path)

visualizer = tx.ResultsVisualizer(tx.config)
visualizer.plot_ablation_comparison(ablation_results)

print("\n" + df.to_string(index=False))
