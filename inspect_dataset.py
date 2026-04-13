import argparse
from collections.abc import Mapping
from typing import Any

import numpy as np
import minari


DEFAULT_DATASET_ID = "D4RL/antmaze/medium-play-v1"


def arr_stats(name: str, x: Any) -> None:
    arr = np.asarray(x)
    print(f"{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        print(f"  min:   {arr.min():.6f}")
        print(f"  max:   {arr.max():.6f}")
        print(f"  mean:  {arr.mean():.6f}")
    flat = arr.reshape(-1)[:8] if arr.size > 0 else arr
    print(f"  first values: {flat}")


def inspect_nested(prefix: str, value: Any) -> None:
    if isinstance(value, Mapping):
        print(f"{prefix}: dict with keys {list(value.keys())}")
        for k, v in value.items():
            inspect_nested(f"{prefix}.{k}", v)
    else:
        arr_stats(prefix, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--list-antmaze", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("MINARI DATASET INSPECTION")
    print("=" * 80)

    if args.list_antmaze:
        return
        print("\nRemote datasets with prefix 'D4RL/antmaze':")
        remote = minari.list_remote_datasets(prefix="D4RL/antmaze", latest_version=False)
        # after printing the remote list
        for ds_id, meta in remote.items():
            print(f"- {ds_id}")
            if isinstance(meta, dict):
                for key in ["total_episodes", "minari_version", "env_spec"]:
                    if key in meta:
                        print(f"    {key}: {meta[key]}")
        print()

    if args.download:
        print(f"\nDownloading dataset: {args.dataset_id}")
        minari.download_dataset(args.dataset_id)

    print(f"\nLoading dataset: {args.dataset_id}")
    dataset = minari.load_dataset(args.dataset_id, download=True)

    print("\n" + "=" * 80)
    print("DATASET-LEVEL INFO")
    print("=" * 80)
    print(f"dataset_id: {dataset._dataset_id}")
    print(f"total_episodes: {dataset.total_episodes}")
    print(f"total_steps: {dataset.total_steps}")

    if hasattr(dataset, "env_spec"):
        print(f"env_spec: {dataset.env_spec}")

    if hasattr(dataset, "observation_space"):
        print(f"observation_space: {dataset.observation_space}")
    if hasattr(dataset, "action_space"):
        print(f"action_space: {dataset.action_space}")

    if hasattr(dataset, "recover_environment"):
        try:
            env = dataset.recover_environment()
            print(f"recovered_env: {env}")
            print(f"recovered_env observation_space: {env.observation_space}")
            print(f"recovered_env action_space: {env.action_space}")
            env.close()
        except Exception as exc:
            print(f"could not recover environment: {exc}")

    print("\n" + "=" * 80)
    print("EPISODE INSPECTION")
    print("=" * 80)

    num_to_show = min(args.episodes, dataset.total_episodes)
    returns = []
    lengths = []

    for i, episode in enumerate(dataset.iterate_episodes()):
        if i >= num_to_show:
            break

        print(f"\nEpisode {i}")
        if hasattr(episode, "id"):
            print(f"  id: {episode.id}")
        if hasattr(episode, "total_timesteps"):
            print(f"  total_timesteps: {episode.total_timesteps}")
            lengths.append(int(episode.total_timesteps))

        if hasattr(episode, "observations"):
            inspect_nested("observations", episode.observations)
        if hasattr(episode, "actions"):
            inspect_nested("actions", episode.actions)
        if hasattr(episode, "rewards"):
            inspect_nested("rewards", episode.rewards)
            returns.append(float(np.asarray(episode.rewards).sum()))
        if hasattr(episode, "terminations"):
            inspect_nested("terminations", episode.terminations)
        if hasattr(episode, "truncations"):
            inspect_nested("truncations", episode.truncations)

    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    if returns:
        print(f"shown episode returns: {returns}")
        print(f"mean shown return: {np.mean(returns):.6f}")
    if lengths:
        print(f"shown episode lengths: {lengths}")
        print(f"mean shown length: {np.mean(lengths):.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()