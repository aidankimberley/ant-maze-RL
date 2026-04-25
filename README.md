## ant-maze-RL (McGill course project)

This repo is a course project (McGill University) exploring a **sim-to-real-style transfer** pipeline in simulation:

- **Offline pre-training**: train a goal-conditioned **HIQL** policy on OGBench AntMaze under *nominal* physics.
- **Physics shift**: modify MuJoCo XML physics parameters (e.g., friction/damping/gear) to create a controlled dynamics mismatch.
- **Online fine-tuning**: continue HIQL training in the shifted environment to recover performance.

The report lives in `final_report.tex` (NeurIPS style via `neurips.sty`).

## Key results (plots used in the report)

### Offline sensitivity + learning curve

![HIQL Offline Sweep](Screenshot%202026-04-23%20at%206.39.06%20PM.png)

![Offline HIQL performance vs training steps](Screenshot%202026-04-23%20at%206.39.58%20PM.png)

### Nominal vs shifted physics (frozen baseline vs online finetuned)

![Success across nominal and moderate shift](Screenshot%202026-04-24%20at%202.16.39%20PM.png)

![Success across nominal/friction/composite shifts](Screenshot%202026-04-24%20at%202.27.42%20PM.png)

## Setup

### Python deps

There are a few different environments in this repo; the minimal set for the HIQL + OGBench scripts is roughly:

- `jax`, `flax`, `distrax`, `ml_collections`
- `gymnasium`
- `matplotlib` (for plotting)
- `imageio` (for writing mp4 videos)

We keep a reference list in `requirements-hiql.txt`.

### OGBench import path

This project depends on the local `ogbench/` checkout. Because the OGBench Python package is nested (`ogbench/ogbench`),
add it to `PYTHONPATH` before running scripts:

```bash
export PYTHONPATH="$(pwd)/ogbench:$PYTHONPATH"
```

## Reproduce evaluations (nominal vs shifted)

Evaluate a checkpoint across AntMaze tasks 1–5 under nominal physics and one-or-more shifts:

```bash
python scripts/eval_hiql_nominal_vs_shifted.py \
  --checkpoint_pkl ./experiments/checkpoint_300000/params_300000.pkl \
  --env_name antmaze-medium-navigate-v0 \
  --source_xml ./generated_assets/antmaze-medium-navigate-v0_composite_shift_moderate.xml \
  --shift_family friction \
  --shift_levels moderate_low severe_high \
  --episodes 10 \
  --seed 0 \
  --out_csv ./experiments/eval_nominal_vs_shifted_step300000_seed0.csv
```

## Plotting CSVs

### Online training logs (`raw_log.csv`)

`test_online_transfer.py` writes a `raw_log.csv`. Plot it with:

```bash
python scripts/plot_raw_log_csv.py --csv experiments/<run_dir>/raw_log.csv
```

### Eval summary CSVs (compare checkpoints / seeds)

If you have multiple eval CSVs from `scripts/eval_hiql_nominal_vs_shifted.py`, plot paper-ready comparisons:

```bash
python scripts/plot_raw_log_csv.py --csv \
  experiments/eval_nominal_vs_shifted_step300000_seed0.csv \
  experiments/eval_nominal_vs_shifted_step300000_seed1.csv \
  experiments/eval_nominal_vs_shifted_step300000_seed2.csv \
  experiments/eval_nominal_vs_shifted_step400000_seed0.csv \
  experiments/eval_nominal_vs_shifted_step400000_seed1.csv \
  experiments/eval_nominal_vs_shifted_step400000_seed2.csv \
  --out_dir experiments/paper_plots
```

## Generate videos (task rollouts)

To render an agent doing a specific AntMaze task and save mp4s into `eval_videos/`:

- Nominal physics: `watch_ant_checkpoint_eval.py`
- Shifted physics + XML manipulation: `watch_shifted_ant_checkpoint_eval.py`

Example (task 2, one recorded episode):

```bash
python watch_shifted_ant_checkpoint_eval.py \
  --checkpoint_pkl ./experiments/checkpoint_300000/params_300000.pkl \
  --env_name antmaze-medium-navigate-v0 \
  --task_id 2 \
  --episodes 0 \
  --video_episodes 1 \
  --max_episode_steps 1000 \
  --nominal \
  --source_xml ./generated_assets/antmaze-medium-navigate-v0_composite_shift_moderate.xml
```

## Online fine-tuning

Online fine-tuning entry point:
- `test_online_transfer.py`

It restores a HIQL checkpoint, runs online interaction in a shifted environment, updates HIQL online, and saves:
- fine-tuned weights under `--save_dir`
- `raw_log.csv` for plotting