## Checkpoints

This directory stores serialized model weights saved during curriculum/meta-training (see `Config.save_every_n_episodes`). Keep large `.pt`/`.bin` files here so they stay out of git history—only this README is tracked.

Suggested layout:
- `run-<timestamp>/model.pt` – full state dict for a single run
- `run-<timestamp>/optimizer.pt` – optional optimizer snapshot
- `best.pt` – symlink or copy of the best-performing checkpoint

Clean up outdated checkpoints periodically to conserve disk space.

