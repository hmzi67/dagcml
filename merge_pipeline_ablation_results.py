"""
Merge the pre-existing, already-completed pipeline-stage ablation run
no_curriculum_maml (results_ablation_no_curriculum_maml/, dated 2026-07-28
19:26) with the two runs no_warmup_dacl and floor_no_warmup_no_curriculum
(run via run_ablation.py's run_suite() on 2026-07-29, currently the sole
entries in results/pipeline_ablation_comparison.json because run_suite()
overwrites rather than merges) into a single combined
results/pipeline_ablation_comparison.json / .csv, and regenerate
plots/pipeline_ablation_comparison.png with all three configs.

This does NOT retrain anything.
"""
import json
import os

import train_xnli as tx

MAIN_RESULTS_DIR = './results'
MAIN_PLOTS_DIR = './plots'

LABELS = ['no_curriculum_maml', 'no_warmup_dacl', 'floor_no_warmup_no_curriculum']

# Existing file currently holds only no_warmup_dacl and
# floor_no_warmup_no_curriculum, written by run_ablation.py's run_suite()
# with an authoritative timing wrapper.
with open(os.path.join(MAIN_RESULTS_DIR, 'pipeline_ablation_comparison.json')) as f:
    existing = json.load(f)

assert set(existing['results'].keys()) == {'no_warmup_dacl', 'floor_no_warmup_no_curriculum'}, (
    f"Expected results/pipeline_ablation_comparison.json to contain exactly "
    f"no_warmup_dacl and floor_no_warmup_no_curriculum, found "
    f"{list(existing['results'].keys())}. Aborting to avoid clobbering "
    f"unexpected state."
)

ablation_results = dict(existing['results'])
ablation_compute_cost = dict(existing['compute_cost'])
ablation_timing = dict(existing['wall_clock_seconds'])
timing_notes = {
    label: 'measured by run_ablation.py run_suite() timing wrapper (authoritative)'
    for label in existing['results']
}

label = 'no_curriculum_maml'
label_dir = f'./results_ablation_{label}'
with open(os.path.join(label_dir, 'final_results.json')) as f:
    ablation_results[label] = json.load(f)
with open(os.path.join(label_dir, 'training_history.json')) as f:
    hist = json.load(f)
ablation_compute_cost[label] = hist.get('compute_cost', {})
# This run predates the current results/pipeline_ablation_comparison.json;
# its wall-clock timing was recorded in that earlier version of the file
# (23860.58s) before this label's entry was overwritten by the
# no_warmup_dacl / floor_no_warmup_no_curriculum run.
ablation_timing[label] = 23860.57993197441
timing_notes[label] = (
    'measured by run_ablation.py run_suite() timing wrapper on 2026-07-28; '
    'recovered from the pipeline_ablation_comparison.json version that '
    'existed before this run overwrote it with no_warmup_dacl / '
    'floor_no_warmup_no_curriculum only.'
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

csv_path = os.path.join(MAIN_RESULTS_DIR, 'pipeline_ablation_comparison.csv')
df.to_csv(csv_path, index=False)
print(f"\n✓ Saved: {csv_path}")

json_path = os.path.join(MAIN_RESULTS_DIR, 'pipeline_ablation_comparison.json')
tx.save_json({
    'configs': {label: tx.PIPELINE_ABLATION_CONFIGS[label] for label in LABELS},
    'results': ablation_results,
    'compute_cost': ablation_compute_cost,
    'wall_clock_seconds': ablation_timing,
    'wall_clock_notes': timing_notes,
    'debug_mode': False,
    'seed': 42,
    'merge_note': (
        'no_curriculum_maml was run to completion on 2026-07-28 '
        '(results_ablation_no_curriculum_maml); no_warmup_dacl and '
        'floor_no_warmup_no_curriculum were run on 2026-07-29 via '
        'run_ablation.py and originally overwrote this file with only those '
        'two. This file was produced by merging the three pre-existing '
        'completed runs, not by retraining any of them.'
    ),
}, json_path)

visualizer = tx.ResultsVisualizer(tx.config)
visualizer.plot_ablation_comparison(ablation_results, save_name='pipeline_ablation_comparison.png')

print("\n" + df.to_string(index=False))
