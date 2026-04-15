#!/usr/bin/env python3
"""
watch_ant_checkpoint_eval.py

Load a saved OGBench HIQL checkpoint (.pkl) and visualize it using
OGBench's own evaluation helper.

Example:
PYTHONPATH="$(pwd)/ogbench:$PYTHONPATH" python watch_ant_checkpoint_eval.py \
  --checkpoint_pkl ./experiments/checkpoint_300000/params_300000.pkl \
  --env_name antmaze-medium-navigate-v0 \
  --task_id 1 \
  --episodes 2 \
  --video_episodes 2
"""

import argparse
import os
import re
import sys
from pathlib import Path
import imageio.v2 as imageio
import numpy as np


CONFIG = {
    "env_name": "antmaze-medium-navigate-v0",
    "seed": 0,
    "batch_size": 512,
    "subgoal_steps": 25,
    "expectile": 0.7,
    "high_alpha": 3.0,
    "low_alpha": 3.0,
    "actor_p_trajgoal": 0.5,
    "actor_p_randomgoal": 0.5,
    "actor_geom_sample": True,
    "discount": 0.99,
}


def setup_imports():
    impls_dir = os.environ.get("OGBENCH_IMPLS", "")
    if not impls_dir:
        this_file = os.path.abspath(__file__)
        impls_dir = os.path.join(os.path.dirname(this_file), "ogbench", "impls")

    impls_dir = os.path.abspath(impls_dir)
    repo_dir = os.path.dirname(impls_dir)

    if not os.path.isdir(impls_dir):
        raise FileNotFoundError(f"Could not find ogbench/impls at: {impls_dir}")

    for p in (repo_dir, impls_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    print(f"[setup] ogbench/impls loaded from: {impls_dir}")


def parse_step(checkpoint_pkl: Path) -> int:
    m = re.search(r"params_(\d+)\.pkl$", checkpoint_pkl.name)
    if not m:
        raise ValueError(
            f"Could not parse step from checkpoint filename: {checkpoint_pkl.name}"
        )
    return int(m.group(1))


def build_hiql_agent_config():
    from agents.hiql import get_config

    cfg = get_config()
    cfg.encoder = None
    cfg.frame_stack = None

    for k, v in (
        ("subgoal_steps", CONFIG["subgoal_steps"]),
        ("expectile", CONFIG["expectile"]),
        ("high_alpha", CONFIG["high_alpha"]),
        ("low_alpha", CONFIG["low_alpha"]),
        ("actor_p_trajgoal", CONFIG["actor_p_trajgoal"]),
        ("actor_p_randomgoal", CONFIG["actor_p_randomgoal"]),
        ("actor_geom_sample", CONFIG["actor_geom_sample"]),
        ("discount", CONFIG["discount"]),
        ("batch_size", CONFIG["batch_size"]),
    ):
        cfg[k] = v

    return cfg


def make_agent(env, seed, agent_cfg):
    from agents import agents as agent_registry

    ex_obs = np.asarray([env.observation_space.sample()], dtype=np.float32)
    ex_act = np.asarray([env.action_space.sample()], dtype=np.float32)

    agent_cls = agent_registry["hiql"]
    return agent_cls.create(seed, ex_obs, ex_act, agent_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_pkl", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="antmaze-medium-navigate-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task_id", type=int, default=1, help="AntMaze eval task id, usually 1..5")
    parser.add_argument("--episodes", type=int, default=2, help="Number of eval episodes")
    parser.add_argument("--video_episodes", type=int, default=2, help="Number of episodes to record")
    args = parser.parse_args()

    checkpoint_pkl = Path(args.checkpoint_pkl).resolve()
    if not checkpoint_pkl.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_pkl}")

    setup_imports()

    import ogbench
    from utils.flax_utils import restore_agent
    from utils.evaluation import evaluate as ogbench_evaluate

    print(f"Loading env and dataset: {args.env_name}...")
    env, _, _ = ogbench.make_env_and_datasets(
        args.env_name,
        compact_dataset=False,
    )
    env.reset(seed=args.seed)

    agent_cfg = build_hiql_agent_config()
    agent = make_agent(env, args.seed, agent_cfg)

    checkpoint_dir = checkpoint_pkl.parent
    step = parse_step(checkpoint_pkl)

    agent = restore_agent(agent, str(checkpoint_dir), step)
    print(f"[restore] Restored checkpoint step {step} from {checkpoint_dir}")

    eval_stats, trajs, renders = ogbench_evaluate(
        agent,
        env,
        task_id=args.task_id,
        config=agent_cfg,
        num_eval_episodes=args.episodes,
        num_video_episodes=args.video_episodes,
    )

    video_dir = Path("eval_videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    for i, render in enumerate(renders):
        if render is None:
            continue

        out_path = video_dir / f"task{args.task_id}_ep{i + 1}_step{step}.mp4"

        # Handle common cases:
        # 1) render is a list of frames
        # 2) render is a numpy array of shape (T, H, W, C)
        if isinstance(render, list):
            frames = render
        elif isinstance(render, np.ndarray):
            if render.ndim == 4:
                frames = list(render)
            else:
                print(f"Skipping render {i}: unexpected ndarray shape {render.shape}")
                continue
        else:
            print(f"Skipping render {i}: unsupported type {type(render)}")
            continue

        if len(frames) == 0:
            print(f"Skipping render {i}: empty frame list")
            continue

        imageio.mimsave(out_path, frames, fps=30)
        print(f"Saved video to {out_path}")

    print("\n[eval]")
    for k, v in eval_stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()