#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

# Allow imports from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.shifted_antmaze import make_shifted_antmaze_env
from envs.shifted_maze_factory import make_maze_env
from envs.xml_utils import read_default_geom_friction, read_floor_friction

from gymnasium.wrappers import TimeLimit


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
        impls_dir = os.path.join(os.path.dirname(os.path.dirname(this_file)), "ogbench", "impls")

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
        raise ValueError(f"Could not parse step from checkpoint filename: {checkpoint_pkl.name}")
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


def parse_maze_type(env_name: str) -> str:
    parts = env_name.split("-")
    if len(parts) < 2:
        raise ValueError(f"Could not parse maze_type from env_name: {env_name}")
    return parts[1]


def make_nominal_antmaze_env(
    env_name: str,
    render_mode: str = "rgb_array",
    max_episode_steps: int = 1000,
):
    maze_type = parse_maze_type(env_name)
    env = make_maze_env(
        loco_env_type="ant",
        maze_env_type="maze",
        maze_type=maze_type,
        ob_type="states",
        render_mode=render_mode,
        add_noise_to_goal=False,
    )
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def evaluate_across_tasks(agent, env, agent_cfg, num_eval_episodes: int):
    from utils.evaluation import evaluate as ogbench_evaluate

    per_task_success = []
    per_task_stats = []

    for task_id in range(1, 6):
        eval_stats, _, _ = ogbench_evaluate(
            agent,
            env,
            task_id=task_id,
            config=agent_cfg,
            num_eval_episodes=num_eval_episodes,
            num_video_episodes=0,
        )
        success = float(eval_stats.get("success", 0.0))
        per_task_success.append(success)
        per_task_stats.append((task_id, eval_stats))

    return {
        "success": float(np.mean(per_task_success)),
        "per_task_success": per_task_success,
        "per_task_stats": per_task_stats,
    }


def save_csv(rows: list[dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "env_name",
        "checkpoint_pkl",
        "seed",
        "variant",
        "shift_family",
        "shift_level",
        "task_id",
        "success",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] Wrote results to {out_csv}")


def print_summary(label: str, result: dict):
    print(f"\n[{label}]")
    print(f"  mean success: {result['success']:.4f}")
    for i, s in enumerate(result["per_task_success"], start=1):
        print(f"  task {i}: {s:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_pkl", type=str, required=True)
    parser.add_argument("--source_xml", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="antmaze-medium-navigate-v0")
    parser.add_argument("--generated_assets_dir", type=str, default="generated_assets")
    parser.add_argument("--shift_family", type=str, default="friction")
    parser.add_argument("--shift_levels", nargs="+", default=["moderate_low"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    args = parser.parse_args()

    checkpoint_pkl = Path(args.checkpoint_pkl).resolve()
    if not checkpoint_pkl.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_pkl}")

    setup_imports()

    from utils.flax_utils import restore_agent

    print(f"Base default geom friction: {read_default_geom_friction(args.source_xml)}")
    print(f"Base floor friction: {read_floor_friction(args.source_xml)}")

    agent_cfg = build_hiql_agent_config()

    # Nominal env + agent
    nominal_env = make_nominal_antmaze_env(args.env_name, max_episode_steps=args.max_episode_steps,)
    nominal_env.reset(seed=args.seed)
    agent = make_agent(nominal_env, args.seed, agent_cfg)

    checkpoint_dir = checkpoint_pkl.parent
    step = parse_step(checkpoint_pkl)
    agent = restore_agent(agent, str(checkpoint_dir), step)
    print(f"[restore] Restored checkpoint step {step} from {checkpoint_dir}")

    rows = []

    # Nominal evaluation
    nominal_result = evaluate_across_tasks(
        agent=agent,
        env=nominal_env,
        agent_cfg=agent_cfg,
        num_eval_episodes=args.episodes,
    )
    print_summary("nominal", nominal_result)

    for task_id, eval_stats in nominal_result["per_task_stats"]:
        rows.append(
            {
                "env_name": args.env_name,
                "checkpoint_pkl": str(checkpoint_pkl),
                "seed": args.seed,
                "variant": "nominal",
                "shift_family": "none",
                "shift_level": "base",
                "task_id": task_id,
                "success": float(eval_stats.get("success", 0.0)),
            }
        )

    # Shifted evaluations
    for shift_level in args.shift_levels:
        shifted_env, spec = make_shifted_antmaze_env(
            env_name=args.env_name,
            source_xml_path=args.source_xml,
            generated_assets_dir=args.generated_assets_dir,
            shift_family=args.shift_family,
            shift_level=shift_level,
            render_mode="rgb_array",
            add_noise_to_goal=False,
        )
        shifted_env = TimeLimit(shifted_env, max_episode_steps=args.max_episode_steps)
        shifted_env.reset(seed=args.seed)

        print(f"\nShift level: {shift_level}")
        print(f"  shifted default geom friction: {read_default_geom_friction(spec.generated_xml_path)}")
        print(f"  shifted floor friction: {read_floor_friction(spec.generated_xml_path)}")
        print(f"  shifted XML path: {spec.generated_xml_path}")

        shifted_result = evaluate_across_tasks(
            agent=agent,
            env=shifted_env,
            agent_cfg=agent_cfg,
            num_eval_episodes=args.episodes,
        )
        print_summary(f"shifted::{shift_level}", shifted_result)

        for task_id, eval_stats in shifted_result["per_task_stats"]:
            rows.append(
                {
                    "env_name": args.env_name,
                    "checkpoint_pkl": str(checkpoint_pkl),
                    "seed": args.seed,
                    "variant": "shifted",
                    "shift_family": args.shift_family,
                    "shift_level": shift_level,
                    "task_id": task_id,
                    "success": float(eval_stats.get("success", 0.0)),
                }
            )

        shifted_env.close()

    nominal_env.close()

    if args.out_csv is not None:
        save_csv(rows, Path(args.out_csv))


if __name__ == "__main__":
    main()