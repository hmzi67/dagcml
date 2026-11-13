## Results

Store experiment metadata here:
- Evaluation reports (`*.json`, `*.csv`, `*.tsv`) produced by `train_xnli.py` or `inference.py`.
- Aggregated metrics per language/run.
- Notes on ablations or hyperparameter sweeps.

Recommended convention:
```
results/
  run-20250101_120000/
    metrics.json
    eval_en.csv
    eval_ru.csv
```

Everything besides this README is ignored by git so you can safely keep large result sets locally while documenting the run structure.

