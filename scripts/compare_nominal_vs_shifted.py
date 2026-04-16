from __future__ import annotations

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.shifted_antmaze import make_shifted_antmaze_env
from envs.shifted_maze_factory import make_maze_env
from envs.xml_utils import read_default_geom_friction


def run_rollout(env, num_steps: int, seed: int):
    """
    Run a short rollout with random actions.
    Returns summary stats and xy trajectory.
    """
    rng = np.random.default_rng(seed)

    total_reward = 0.0
    final_success = 0.0
    terminated = False
    truncated = False

    xy_traj = []
    rewards = []

    for _ in range(num_steps):
        # Random action baseline for now
        low = env.action_space.low
        high = env.action_space.high
        action = rng.uniform(low=low, high=high)

        ob, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        rewards.append(reward)
        final_success = info.get("success", 0.0)

        xy = info.get("xy", None)
        if xy is not None:
            xy_traj.append(np.array(xy, dtype=np.float32))

        if terminated or truncated:
            break

    if len(xy_traj) > 0:
        final_xy = xy_traj[-1]
    else:
        final_xy = None

    return {
        "total_reward": float(total_reward),
        "final_success": float(final_success),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "num_steps_executed": len(rewards),
        "final_xy": final_xy,
        "xy_traj": np.array(xy_traj, dtype=np.float32) if len(xy_traj) > 0 else np.zeros((0, 2), dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
    }


def print_summary(name: str, result: dict):
    print(f"\n=== {name.upper()} ===")
    print(f"steps executed: {result['num_steps_executed']}")
    print(f"total reward: {result['total_reward']:.2f}")
    print(f"final success: {result['final_success']}")
    print(f"terminated: {result['terminated']}")
    print(f"truncated: {result['truncated']}")
    print(f"final xy: {result['final_xy']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_xml", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="antmaze-medium-navigate-v0")
    parser.add_argument("--generated_assets_dir", type=str, default="generated_assets")
    parser.add_argument("--shift_level", type=str, default="moderate_low")
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout_steps", type=int, default=200)
    parser.add_argument("--save_npz", type=str, default=None)
    args = parser.parse_args()

    print("=== SETUP ===")
    print(f"env_name: {args.env_name}")
    print(f"task_id: {args.task_id}")
    print(f"seed: {args.seed}")
    print(f"rollout_steps: {args.rollout_steps}")
    print(f"base XML friction: {read_default_geom_friction(args.source_xml)}")

    # Parse maze type from env_name: antmaze-medium-navigate-v0 -> medium
    maze_type = args.env_name.split("-")[1]

    # Nominal env
    nominal_env = make_maze_env(
        loco_env_type="ant",
        maze_env_type="maze",
        maze_type=maze_type,
        ob_type="states",
        render_mode="rgb_array",
        add_noise_to_goal=False,
    )

    # Shifted env
    shifted_env, spec = make_shifted_antmaze_env(
        env_name=args.env_name,
        source_xml_path=args.source_xml,
        generated_assets_dir=args.generated_assets_dir,
        shift_family="friction",
        shift_level=args.shift_level,
        render_mode="rgb_array",
        add_noise_to_goal=False,
    )

    print(f"shifted XML friction: {read_default_geom_friction(spec.generated_xml_path)}")
    print(f"shifted XML path: {spec.generated_xml_path}")

    # Reset both on same task/seed and verify obs, info, and xy/goal are consistent
    nominal_ob, nominal_info = nominal_env.reset(seed=args.seed, options={"task_id": args.task_id})
    shifted_ob, shifted_info = shifted_env.reset(seed=args.seed, options={"task_id": args.task_id})

    shared_init_xy = nominal_env.get_xy().copy()
    shared_goal_xy = np.array(nominal_env.cur_goal_xy).copy()

    nominal_env.set_goal(goal_xy=shared_goal_xy)
    shifted_env.set_goal(goal_xy=shared_goal_xy)

    nominal_env.set_xy(shared_init_xy)
    shifted_env.set_xy(shared_init_xy)

    nominal_ob = nominal_env.get_ob()
    shifted_ob = shifted_env.get_ob()

    print("\n=== RESET CHECK ===")
    print(f"nominal obs shape: {nominal_ob.shape}")
    print(f"shifted obs shape: {shifted_ob.shape}")
    print(f"nominal current task id: {nominal_env.cur_task_id}")
    print(f"shifted current task id: {shifted_env.cur_task_id}")
    print(f"shared goal xy: {shared_goal_xy}")
    print(f"shared init xy: {shared_init_xy}")
    print(f"nominal init xy after sync: {nominal_env.get_xy()}")
    print(f"shifted init xy after sync: {shifted_env.get_xy()}")

    # Run rollouts
    nominal_result = run_rollout(nominal_env, num_steps=args.rollout_steps, seed=args.seed)
    shifted_result = run_rollout(shifted_env, num_steps=args.rollout_steps, seed=args.seed)

    print_summary("nominal", nominal_result)
    print_summary("shifted", shifted_result)

    # Simple comparison
    print("\n=== DELTA ===")
    print(f"reward delta (shifted - nominal): {shifted_result['total_reward'] - nominal_result['total_reward']:.2f}")
    if nominal_result["final_xy"] is not None and shifted_result["final_xy"] is not None:
        print(f"final xy delta: {shifted_result['final_xy'] - nominal_result['final_xy']}")

    # Optional save
    if args.save_npz is not None:
        np.savez(
            args.save_npz,
            nominal_xy=nominal_result["xy_traj"],
            shifted_xy=shifted_result["xy_traj"],
            nominal_rewards=nominal_result["rewards"],
            shifted_rewards=shifted_result["rewards"],
        )
        print(f"\nSaved trajectories to: {args.save_npz}")

    nominal_env.close()
    shifted_env.close()


if __name__ == "__main__":
    main()