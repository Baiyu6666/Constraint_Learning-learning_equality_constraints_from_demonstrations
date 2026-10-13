# Dataset Caches

This directory stores generated caches that can be recreated from code and config.
Do not put canonical real-robot data here.

- `ur5_upward/`: cache for `6d_spatial_arm_up_n6(_py)(_traj)` train/grid samples.
  Files are keyed by dataset name, backend, seed, `n_train`, and `n_grid`.

It is safe to delete these files; `experiments/dataset_resolve.py` regenerates a
missing cache on demand.
