#!/usr/bin/env python3
"""
test_online_transfer.py

Online fine-tuning starting from an offline OGBench HIQL checkpoint.

Two modes:
- online_hiql: keep acting with HIQL and update HIQL online using HGC-style relabeling.
- pex: "policy expansion" by training an online SAC policy on (obs, goal) while executing a
  convex combination of (frozen) HIQL and SAC actions with a ramped mixing coefficient.

Added here:
- optional mixed replay for online_hiql:
  sample part of each update batch from the original offline OGBench dataset
  and the rest from the growing online shifted replay buffer.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from envs.shifted_antmaze import make_shifted_antmaze_env
from gymnasium.wrappers import TimeLimit

import jax
import numpy as np


from pathlib import Path


def _short_steps(n: int) -> str:
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def _short_shift_name(shift_family: str | None, shift_level: str | None) -> str:
    if shift_family is None or shift_level is None:
        return "nominal"

    family_map = {
        "composite_shift": "comp",
        "floor_friction": "floor",
        "friction": "fric",
        "joint_damping": "damp",
        "actuator_gear": "gear",
    }
    fam = family_map.get(shift_family, shift_family.replace("_", ""))
    return f"{fam}-{shift_level}"


def make_default_save_dir(args) -> str:
    method = "hiql" if args.method == "online_hiql" else "pex"
    shift = _short_shift_name(args.shift_family, args.shift_level)
    steps = _short_steps(int(args.total_steps))

    parts = [method, shift, steps, f"s{args.seed}"]

    run_name = getattr(args, "run_name", None)
    if run_name:
        parts.append(run_name)

    return str((Path("results") / "_".join(parts)).resolve())

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


def build_hiql_agent_config(overrides: dict):
    from agents.hiql import get_config

    cfg = get_config()
    # State-based AntMaze in OGBench: encoder=None, frame_stack=None
    cfg.encoder = None
    cfg.frame_stack = None
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def make_agent(agent_name: str, env, seed: int, agent_cfg):
    from agents import agents as agent_registry

    ex_obs = np.asarray([env.observation_space.sample()], dtype=np.float32)
    ex_act = np.asarray([env.action_space.sample()], dtype=np.float32)

    agent_cls = agent_registry[agent_name]
    return agent_cls.create(seed, ex_obs, ex_act, agent_cfg)


@dataclass
class OnlineEpisodesIndex:
    """
    Tracks episode boundaries for a growing, non-circular buffer.
    """
    episode_end: np.ndarray  # shape (max_size,), int32 (filled for [0, size))
    size: int = 0
    cur_ep_start: int = 0

    def start_new_episode(self):
        self.cur_ep_start = self.size

    def push_and_maybe_close_episode(self, done: bool):
        self.size += 1
        if done:
            end = self.size - 1
            self.episode_end[self.cur_ep_start : end + 1] = end
            self.cur_ep_start = self.size


def _geom_offsets(discount: float, n: int) -> np.ndarray:
    # matches ogbench/impls/utils/datasets.py geometric sampling in spirit
    p = 1.0 - float(discount)
    p = np.clip(p, 1e-6, 1.0)
    return np.random.geometric(p=p, size=n).astype(np.int64)  # in [1, inf)


def sample_hgc_batch(buffer, idx: OnlineEpisodesIndex, batch_size: int, cfg) -> dict:
    """
    Online version of HGCDataset.sample() that works on a growing buffer.

    Produces the exact keys HIQL expects:
      observations, actions, next_observations, rewards, masks,
      value_goals, low_actor_goals, high_actor_goals, high_actor_targets
    """
    n = idx.size
    if n < 2:
        raise RuntimeError("Not enough online data yet to sample a batch.")

    # only sample indices whose episode_end is known (i.e., episode finished)
    valid = np.nonzero(idx.episode_end[:n] >= 0)[0]
    if len(valid) == 0:
        raise RuntimeError("No completed episodes in buffer yet.")

    idxs = valid[np.random.randint(len(valid), size=batch_size)]
    final_state_idxs = idx.episode_end[idxs]

    # Value goals: mix of current / future-in-episode / random.
    random_goal_idxs = valid[np.random.randint(len(valid), size=batch_size)]

    if cfg["value_geom_sample"]:
        offsets = _geom_offsets(cfg["discount"], batch_size)
        traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
    else:
        distances = np.random.rand(batch_size)
        lo = np.minimum(idxs + 1, final_state_idxs)
        hi = final_state_idxs
        traj_goal_idxs = np.round(lo * distances + hi * (1.0 - distances)).astype(np.int64)

    if float(cfg["value_p_curgoal"]) == 1.0:
        value_goal_idxs = idxs
    else:
        pick_traj = np.random.rand(batch_size) < (
            cfg["value_p_trajgoal"] / (1.0 - cfg["value_p_curgoal"])
        )
        value_goal_idxs = np.where(pick_traj, traj_goal_idxs, random_goal_idxs)
        pick_cur = np.random.rand(batch_size) < cfg["value_p_curgoal"]
        value_goal_idxs = np.where(pick_cur, idxs, value_goal_idxs)

    successes = (idxs == value_goal_idxs).astype(np.float32)
    masks = 1.0 - successes
    rewards = successes - (1.0 if cfg["gc_negative"] else 0.0)

    # Low-level goals are fixed subgoal_steps into the future (clamped to episode end).
    low_goal_idxs = np.minimum(idxs + int(cfg["subgoal_steps"]), final_state_idxs)

    # High-level actor goals/targets.
    if cfg["actor_geom_sample"]:
        offsets = _geom_offsets(cfg["discount"], batch_size)
        high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
    else:
        distances = np.random.rand(batch_size)
        lo = np.minimum(idxs + 1, final_state_idxs)
        hi = final_state_idxs
        high_traj_goal_idxs = np.round(lo * distances + hi * (1.0 - distances)).astype(np.int64)

    high_traj_target_idxs = np.minimum(
        idxs + int(cfg["subgoal_steps"]), high_traj_goal_idxs
    )

    high_random_goal_idxs = valid[np.random.randint(len(valid), size=batch_size)]
    high_random_target_idxs = np.minimum(
        idxs + int(cfg["subgoal_steps"]), final_state_idxs
    )

    pick_random = np.random.rand(batch_size) < cfg["actor_p_randomgoal"]
    high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
    high_target_idxs = np.where(
        pick_random, high_random_target_idxs, high_traj_target_idxs
    )

    batch = buffer.sample(batch_size, idxs)
    batch["rewards"] = rewards.astype(np.float32)
    batch["masks"] = masks.astype(np.float32)
    batch["value_goals"] = buffer["observations"][value_goal_idxs]
    batch["low_actor_goals"] = buffer["observations"][low_goal_idxs]
    batch["high_actor_goals"] = buffer["observations"][high_goal_idxs]
    batch["high_actor_targets"] = buffer["observations"][high_target_idxs]
    batch["valids"] = np.ones((batch_size,), dtype=np.float32)
    return batch


def _batch_to_numpy_dict(batch: dict) -> dict:
    return {k: np.asarray(v) for k, v in batch.items()}


def _concat_batches(batch_a: dict, batch_b: dict) -> dict:
    if batch_a is None:
        return _batch_to_numpy_dict(batch_b)
    if batch_b is None:
        return _batch_to_numpy_dict(batch_a)

    a = _batch_to_numpy_dict(batch_a)
    b = _batch_to_numpy_dict(batch_b)

    keys_a = set(a.keys())
    keys_b = set(b.keys())
    if keys_a != keys_b:
        raise KeyError(
            f"Batch key mismatch.\nA only: {sorted(keys_a - keys_b)}\nB only: {sorted(keys_b - keys_a)}"
        )

    out = {}
    for k in a.keys():
        out[k] = np.concatenate([a[k], b[k]], axis=0)
    return out


def sample_mixed_hiql_batch(
    offline_dataset,
    online_buffer,
    online_index: OnlineEpisodesIndex,
    batch_size: int,
    cfg,
    offline_fraction: float,
):
    """
    Mix an offline HIQL/HGC batch with an online relabeled HIQL batch.

    offline_fraction = fraction of the total batch sampled from the offline dataset.
    If online data is not ready yet, falls back to offline-only.
    """
    offline_fraction = float(np.clip(offline_fraction, 0.0, 1.0))

    if offline_dataset is None or offline_fraction <= 0.0:
        return sample_hgc_batch(online_buffer, online_index, batch_size, cfg)

    if offline_fraction >= 1.0:
        return _batch_to_numpy_dict(offline_dataset.sample(batch_size))

    offline_bs = int(round(batch_size * offline_fraction))
    offline_bs = min(max(offline_bs, 1), batch_size - 1)
    online_bs = batch_size - offline_bs

    offline_batch = _batch_to_numpy_dict(offline_dataset.sample(offline_bs))

    try:
        online_batch = sample_hgc_batch(online_buffer, online_index, online_bs, cfg)
    except RuntimeError:
        # Early in training, no completed online episodes yet.
        return _batch_to_numpy_dict(offline_dataset.sample(batch_size))

    return _concat_batches(offline_batch, online_batch)


def parse_task_ids(task_ids_str: str) -> list[int]:
    vals = [int(x.strip()) for x in task_ids_str.split(",") if x.strip()]
    if not vals:
        raise ValueError("online_task_ids must contain at least one task id.")
    for t in vals:
        if t < 1 or t > 5:
            raise ValueError(f"Task ids must be in [1,5], got {t}")
    return vals


def choose_next_task_id(
    online_task_ids: list[int],
    schedule: str,
    episode_idx: int,
    rng: np.random.RandomState,
    fixed_task_id: int | None = None,
) -> int:
    if schedule == "fixed":
        if fixed_task_id is None:
            raise ValueError("fixed_task_id must be provided when schedule='fixed'.")
        return fixed_task_id
    if schedule == "cycle":
        return online_task_ids[episode_idx % len(online_task_ids)]
    if schedule == "random":
        return int(rng.choice(online_task_ids))
    raise ValueError(f"Unknown task schedule: {schedule}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--env_name", type=str, default="antmaze-medium-navigate-v0")
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Single-task training/eval task id. Required only when not using --multitask_online.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, default="online_hiql", choices=["online_hiql", "pex"])
    parser.add_argument("--total_steps", type=int, default=20_000)
    parser.add_argument("--start_updates_after", type=int, default=1_000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--action_noise", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default=None, help="Where to save final fine-tuned weights.")

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional short suffix to distinguish runs, e.g. pilot, mix075, mt.",
    )

    # mixed replay knobs
    parser.add_argument(
        "--use_offline_replay",
        action="store_true",
        help="For online_hiql, mix the original offline OGBench dataset into each update batch.",
    )
    parser.add_argument(
        "--offline_batch_fraction",
        type=float,
        default=0.75,
        help="Fraction of each HIQL update batch sampled from the offline dataset when --use_offline_replay is enabled.",
    )

    parser.add_argument("--multitask_online", action="store_true")
    parser.add_argument(
        "--online_task_ids",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated task ids used for online collection when --multitask_online is enabled.",
    )
    parser.add_argument(
        "--task_schedule",
        type=str,
        default="fixed",
        choices=["fixed", "cycle", "random"],
        help="How to choose the next task for online collection.",
    )
    parser.add_argument(
        "--eval_all_tasks_during_train",
        action="store_true",
        help="If set, periodic eval during training runs on all online_task_ids and prints mean/per-task success.",
    )

    # PEX knobs (SAC expansion)
    parser.add_argument("--pex_mix_final", type=float, default=0.8)
    parser.add_argument("--pex_mix_warmup", type=int, default=5_000)
    parser.add_argument("--sac_batch_size", type=int, default=256)

    parser.add_argument("--source_xml", type=str, default=None)
    parser.add_argument("--shift_family", type=str, default=None)
    parser.add_argument("--shift_level", type=str, default=None)
    parser.add_argument("--generated_assets_dir", type=str, default="generated_assets")
    parser.add_argument("--max_episode_steps", type=int, default=500)
    args = parser.parse_args()

    online_task_ids = parse_task_ids(args.online_task_ids)

    if args.multitask_online:
        if args.task_schedule == "fixed":
            args.task_schedule = "cycle"
        fixed_task_id = online_task_ids[0]
    else:
        if args.task_id is None:
            raise ValueError("--task_id is required when not using --multitask_online.")
        online_task_ids = [args.task_id]
        args.task_schedule = "fixed"
        fixed_task_id = args.task_id

    task_rng = np.random.RandomState(args.seed)
    current_episode_idx = 0
    current_task_id = choose_next_task_id(
        online_task_ids=online_task_ids,
        schedule=args.task_schedule,
        episode_idx=current_episode_idx,
        rng=task_rng,
        fixed_task_id=fixed_task_id,
    )

    print(
        f"[tasks] multitask_online={args.multitask_online} "
        f"schedule={args.task_schedule} "
        f"online_task_ids={online_task_ids} "
        f"start_task={current_task_id}"
    )

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    ckpt_file = checkpoint_dir / f"params_{args.step}.pkl"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    setup_imports()

    import ogbench
    from utils.flax_utils import restore_agent, save_agent
    from utils.evaluation import evaluate as ogbench_evaluate
    from utils.datasets import ReplayBuffer, Dataset, HGCDataset

    from gymnasium.wrappers import TimeLimit
    from envs.shifted_antmaze import make_shifted_antmaze_env
    from envs.shifted_maze_factory import make_maze_env

    print(f"[env] Creating: {args.env_name}")
<<<<<<< HEAD

    # env = ogbench.make_env_and_datasets(args.env_name, env_only=True)

    shifted_env, shift_spec = make_shifted_antmaze_env(
        env_name=args.env_name,
        source_xml_path="generated_assets/antmaze-medium-navigate-v0_composite_shift_moderate.xml",
        generated_assets_dir="generated_assets",
        shift_family="friction",
        shift_level="moderate_low",
        render_mode="rgb_array",
        add_noise_to_goal=False,
    )

    env = TimeLimit(shifted_env, max_episode_steps=1000)
    print("[shift] using shifted env:", shift_spec)
=======
    if args.shift_family is None or args.shift_level is None:
        # nominal env
        maze_type = args.env_name.split("-")[1]
        env = make_maze_env(
            loco_env_type="ant",
            maze_env_type="maze",
            maze_type=maze_type,
            ob_type="states",
            render_mode="rgb_array",
            add_noise_to_goal=False,
        )
    else:
        if args.source_xml is None:
            raise ValueError("--source_xml is required when using a shifted environment.")

        env, spec = make_shifted_antmaze_env(
            env_name=args.env_name,
            source_xml_path=args.source_xml,
            generated_assets_dir=args.generated_assets_dir,
            shift_family=args.shift_family,
            shift_level=args.shift_level,
            render_mode="rgb_array",
            add_noise_to_goal=False,
        )
        print(f"[env] Using shifted XML: {spec.generated_xml_path}")

    env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
>>>>>>> 415467a (added offline buffer replay mixing)

    obs, info = env.reset(seed=args.seed, options=dict(task_id=current_task_id))
    goal = info.get("goal")

    # HIQL config: mirror the same knobs used in offline runs.
    hiql_cfg = build_hiql_agent_config(
        overrides=dict(
            batch_size=args.batch_size,
            subgoal_steps=25,
            expectile=0.7,
            high_alpha=3.0,
            low_alpha=3.0,
            actor_p_trajgoal=0.5,
            actor_p_randomgoal=0.5,
            actor_geom_sample=True,
            discount=0.99,
        )
    )
    hiql_agent = make_agent("hiql", env, args.seed, hiql_cfg)
    hiql_agent = restore_agent(hiql_agent, str(checkpoint_dir), args.step)
    print(f"[restore] HIQL restored from {checkpoint_dir} @ step {args.step}")

    # Optional offline dataset for mixed replay.
    offline_hiql_dataset = None
    if args.method == "online_hiql" and args.use_offline_replay:
        print(f"[offline] Loading OGBench dataset for {args.env_name}")
        _, train_dataset_dict, _ = ogbench.make_env_and_datasets(
            args.env_name,
            compact_dataset=True,
        )
        offline_n = int(train_dataset_dict["observations"].shape[0])
        base_dataset = Dataset.create(**train_dataset_dict)
        offline_hiql_dataset = HGCDataset(base_dataset, hiql_cfg)
        print(
            f"[offline] loaded {offline_n} transitions | "
            f"offline_batch_fraction={args.offline_batch_fraction:.2f}"
        )

    # Online buffers.
    max_buf = int(args.total_steps) + 10_000
    example_transition = dict(
        observations=np.asarray(obs, dtype=np.float32),
        actions=np.asarray(env.action_space.sample(), dtype=np.float32),
        rewards=np.asarray(0.0, dtype=np.float32),
        masks=np.asarray(1.0, dtype=np.float32),
        terminals=np.asarray(0.0, dtype=np.float32),
        next_observations=np.asarray(obs, dtype=np.float32),
    )
    rb = ReplayBuffer.create(example_transition, size=max_buf)
    ep_index = OnlineEpisodesIndex(episode_end=np.full((max_buf,), -1, dtype=np.int32))
    ep_index.start_new_episode()

    # PEX/SAC agent (trained online on concatenated (obs, goal) as "observation")
    sac_agent = None
    sac_rb = None
    if args.method == "pex":
        from agents.sac import get_config as sac_get_config
        from agents.sac import SACAgent

        if goal is None:
            raise RuntimeError("PEX mode requires env to provide info['goal'].")
        sac_cfg = sac_get_config()
        sac_cfg.batch_size = args.sac_batch_size

        ex_obs_sac = np.concatenate([np.asarray(obs, np.float32), np.asarray(goal, np.float32)], axis=-1)
        ex_obs_sac = ex_obs_sac[None, :]
        ex_act = np.asarray([env.action_space.sample()], dtype=np.float32)

        sac_agent = SACAgent.create(args.seed, ex_obs_sac, ex_act, sac_cfg)

        sac_example = dict(
            observations=ex_obs_sac[0].astype(np.float32),
            actions=ex_act[0].astype(np.float32),
            rewards=np.asarray(0.0, dtype=np.float32),
            masks=np.asarray(1.0, dtype=np.float32),
            next_observations=ex_obs_sac[0].astype(np.float32),
        )
        sac_rb = ReplayBuffer.create(sac_example, size=max_buf)

    def eval_hiql(step: int):
        if args.eval_all_tasks_during_train:
            task_successes = {}
            for task_id in online_task_ids:
                stats, _, _ = ogbench_evaluate(
                    hiql_agent,
                    env,
                    task_id=task_id,
                    config=hiql_cfg,
                    num_eval_episodes=args.eval_episodes,
                    num_video_episodes=0,
                )
                task_successes[task_id] = float(stats.get("success", 0.0))

            mean_succ = float(np.mean(list(task_successes.values())))
            detail = " ".join([f"task{tid}={succ:.3f}" for tid, succ in task_successes.items()])
            print(f"[eval] step={step} mean_success={mean_succ:.3f} {detail}")
        else:
            stats, _, _ = ogbench_evaluate(
                hiql_agent,
                env,
                task_id=args.task_id,
                config=hiql_cfg,
                num_eval_episodes=args.eval_episodes,
                num_video_episodes=0,
            )
            succ = float(stats.get("success", 0.0))
            print(f"[eval] step={step} task={args.task_id} success={succ:.3f}")
    total_env_steps = 0
    episode_return = 0.0
    episode_len = 0
    rng = jax.random.PRNGKey(args.seed)

    print(f"[run] method={args.method} total_steps={args.total_steps}")
    while total_env_steps < args.total_steps:
        if goal is None:
            raise RuntimeError("Environment did not provide info['goal']; cannot act with HIQL.")

        # Action selection.
        rng, act_key = jax.random.split(rng)
        base_action = np.asarray(
            hiql_agent.sample_actions(observations=obs, goals=goal, temperature=0.0, seed=act_key),
            dtype=np.float32,
        )
        action = base_action

        if args.method == "pex":
            mix = min(1.0, total_env_steps / max(1, int(args.pex_mix_warmup))) * float(args.pex_mix_final)
            sac_obs = np.concatenate([np.asarray(obs, np.float32), np.asarray(goal, np.float32)], axis=-1)
            rng, sac_key = jax.random.split(rng)
            sac_action = np.asarray(
                sac_agent.sample_actions(observations=sac_obs, goals=None, temperature=1.0, seed=sac_key),
                dtype=np.float32,
            )
            action = (1.0 - mix) * base_action + mix * sac_action

        # Exploration noise (both methods).
        if args.action_noise > 0:
            action = action + np.random.normal(0.0, args.action_noise, size=action.shape).astype(np.float32)
        action = np.clip(action, -1.0, 1.0)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        next_goal = info.get("goal", goal)

        episode_return += float(reward)
        episode_len += 1

        # Store transition in HIQL online buffer (for relabeling).
        rb.add_transition(
            dict(
                observations=np.asarray(obs, dtype=np.float32),
                actions=np.asarray(action, dtype=np.float32),
                rewards=np.asarray(float(reward), dtype=np.float32),
                masks=np.asarray(0.0 if done else 1.0, dtype=np.float32),
                terminals=np.asarray(1.0 if done else 0.0, dtype=np.float32),
                next_observations=np.asarray(next_obs, dtype=np.float32),
            )
        )
        ep_index.push_and_maybe_close_episode(done)

        if args.method == "pex":
            sac_obs = np.concatenate([np.asarray(obs, np.float32), np.asarray(goal, np.float32)], axis=-1)
            sac_next_obs = np.concatenate([np.asarray(next_obs, np.float32), np.asarray(next_goal, np.float32)], axis=-1)
            sac_rb.add_transition(
                dict(
                    observations=sac_obs.astype(np.float32),
                    actions=np.asarray(action, dtype=np.float32),
                    rewards=np.asarray(float(reward), dtype=np.float32),
                    masks=np.asarray(0.0 if done else 1.0, dtype=np.float32),
                    next_observations=sac_next_obs.astype(np.float32),
                )
            )

        total_env_steps += 1

        # Online updates.
        if total_env_steps >= args.start_updates_after:
            if args.method == "online_hiql":
                for _ in range(int(args.updates_per_step)):
                    if offline_hiql_dataset is not None:
                        batch = sample_mixed_hiql_batch(
                            offline_dataset=offline_hiql_dataset,
                            online_buffer=rb,
                            online_index=ep_index,
                            batch_size=args.batch_size,
                            cfg=hiql_cfg,
                            offline_fraction=args.offline_batch_fraction,
                        )
                    else:
                        batch = sample_hgc_batch(rb, ep_index, args.batch_size, hiql_cfg)

                    hiql_agent, _ = hiql_agent.update(batch)
            else:
                for _ in range(int(args.updates_per_step)):
                    b = sac_rb.sample(args.sac_batch_size)
                    sac_agent, _ = sac_agent.update(b)

        # Periodic eval.
        if args.eval_every and (total_env_steps % int(args.eval_every) == 0):
            eval_hiql(total_env_steps)

        if done:
            print(
                f"[episode] task={current_task_id} "
                f"steps={episode_len} return={episode_return:.3f}"
            )
            current_episode_idx += 1
            current_task_id = choose_next_task_id(
                online_task_ids=online_task_ids,
                schedule=args.task_schedule,
                episode_idx=current_episode_idx,
                rng=task_rng,
                fixed_task_id=args.task_id,
            )

            obs, info = env.reset(options=dict(task_id=current_task_id))
            goal = info.get("goal")
            episode_return = 0.0
            episode_len = 0
            continue

        obs = next_obs
        goal = next_goal

    if (not args.eval_every) or (total_env_steps % int(args.eval_every) != 0):
        eval_hiql(total_env_steps)

    # Save final weights.
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = make_default_save_dir(args)
    save_dir = str(Path(save_dir).resolve())
    os.makedirs(save_dir, exist_ok=True)
    final_step = int(args.step) + int(total_env_steps)

    hiql_out = os.path.join(save_dir, "hiql")
    os.makedirs(hiql_out, exist_ok=True)
    save_agent(hiql_agent, hiql_out, final_step)
    print(f"[save] HIQL -> {hiql_out}/params_{final_step}.pkl")

    if args.method == "pex":
        sac_out = os.path.join(save_dir, "sac")
        os.makedirs(sac_out, exist_ok=True)
        save_agent(sac_agent, sac_out, final_step)
        print(f"[save] SAC (pex) -> {sac_out}/params_{final_step}.pkl")

    env.close()


if __name__ == "__main__":
    main()