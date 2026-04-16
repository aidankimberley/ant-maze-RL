from __future__ import annotations

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.shifted_antmaze import make_shifted_antmaze_env
from envs.xml_utils import read_default_geom_friction


def safe_print_geom_friction(env, geom_name: str) -> None:
    try:
        friction = env.model.geom(geom_name).friction
        print(f"{geom_name} runtime friction: {friction}")
    except Exception as e:
        print(f"Could not inspect geom '{geom_name}': {e}")


def run_short_rollout(env, num_steps: int = 10) -> None:
    total_reward = 0.0
    terminated = False
    truncated = False

    for t in range(num_steps):
        action = env.action_space.sample()
        ob, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        xy = info.get("xy", None)
        success = info.get("success", None)

        print(
            f"step={t+1:02d} "
            f"reward={reward:.2f} "
            f"success={success} "
            f"xy={xy}"
        )

        if terminated or truncated:
            print(f"Episode ended early at step {t+1}.")
            break

    print(f"Short rollout total reward: {total_reward:.2f}")
    print(f"terminated={terminated}, truncated={truncated}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_xml", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="antmaze-medium-navigate-v0")
    parser.add_argument("--generated_assets_dir", type=str, default="generated_assets")
    parser.add_argument("--shift_level", type=str, default="moderate_low")
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout_steps", type=int, default=10)
    args = parser.parse_args()

    print("=== XML CHECKS ===")
    print(f"Base XML: {args.source_xml}")
    print(f"Base default geom friction: {read_default_geom_friction(args.source_xml)}")

    env, spec = make_shifted_antmaze_env(
        env_name=args.env_name,
        source_xml_path=args.source_xml,
        generated_assets_dir=args.generated_assets_dir,
        shift_family="friction",
        shift_level=args.shift_level,
    )

    print("\nGenerated shifted XML:")
    print(spec.generated_xml_path)
    print(f"Shift values: {spec.shift_values}")
    print(f"Generated default geom friction: {read_default_geom_friction(spec.generated_xml_path)}")

    print("\n=== ENV RESET CHECKS ===")
    ob, info = env.reset(seed=args.seed, options={"task_id": args.task_id})
    print("Shifted env successfully reset.")
    print(f"Observation shape: {ob.shape}")
    print(f"Observation dtype: {ob.dtype}")
    print(f"Num tasks: {env.num_tasks}")
    print(f"Current task id: {env.cur_task_id}")
    print(f"Goal shape: {info['goal'].shape}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Initial agent xy: {env.get_xy()}")
    print(f"Current goal xy: {env.cur_goal_xy}")

    print("\n=== RUNTIME MUJOCO CHECKS ===")
    safe_print_geom_friction(env, "floor")
    safe_print_geom_friction(env, "torso_geom")
    safe_print_geom_friction(env, "aux_1_geom")

    print("\n=== SHORT ROLLOUT CHECK ===")
    run_short_rollout(env, num_steps=args.rollout_steps)

    env.close()


if __name__ == "__main__":
    main()