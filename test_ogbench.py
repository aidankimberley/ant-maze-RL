"""
train_hiql_antmaze.py

Trains HIQL on OGBench AntMaze. Can be run from anywhere — just point
OGBENCH_IMPLS at your local ogbench/impls/ directory:

    git clone https://github.com/seohongpark/ogbench.git
    cd ogbench/impls && pip install -r requirements.txt && cd -

    export OGBENCH_IMPLS=/path/to/ogbench/impls
    python train_hiql_antmaze.py

OGBENCH_IMPLS defaults to ../ogbench/impls relative to this file,
which works if your project sits next to the ogbench clone:

    your_project/
    ├── train_hiql_antmaze.py   ← this file
    ogbench/
    └── impls/

To change training settings, edit the CONFIG dict below.
"""

import os
import sys
import time
import random

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    # Environment / dataset
    "env_name": "antmaze-medium-navigate-v0",  # change to antmaze-medium-navigate-v0 for faster runs
    "seed": 0,

    # # Training
    # "train_steps": 1_000_000,
    # "batch_size": 1024,
    # "eval_interval": 50_000,    # how often to run evaluation rollouts
    # "eval_episodes": 50,        # episodes per eval
    # "save_interval": 250_000,   # how often to checkpoint

    "train_steps": 300_000,
    "batch_size": 512,
    "eval_interval": 100_000,
    "eval_episodes": 10,
    "save_interval": 500_000,


    "save_dir": "/content/drive/MyDrive/hiql_experiments",

    # HIQL hyperparameters (tuned for AntMaze navigate, from OGBench paper)
    "subgoal_steps": 25,        # k: how many steps ahead the high-level policy looks
    "expectile": 0.7,           # IQL expectile for value function training
    "high_alpha": 3.0,          # AWR temperature for high-level policy
    "low_alpha": 3.0,           # AWR temperature for low-level policy
    "actor_p_trajgoal": 0.5,    # fraction of batch using in-trajectory goals
    "actor_p_randomgoal": 0.5,  # fraction of batch using random goals
    "actor_geom_sample": True,  # geometrically sample future states (recommended)
    "discount": 0.99,

    # Logging
    "use_wandb": False,          # set True and fill in project below if you want W&B
    "wandb_project": "hiql-antmaze",
}
# ─────────────────────────────────────────────────────────────────────────────


def setup_imports():
    """
    Inject ogbench/impls onto sys.path so its agents/ and utils/ modules
    are importable from anywhere.

    Resolution order:
      1. OGBENCH_IMPLS environment variable  (most explicit)
      2. ../ogbench/impls relative to this file  (default layout)
    """
    # 1. Env var takes priority
    impls_dir = os.environ.get("OGBENCH_IMPLS", "")

    # 2. Fall back to ../ogbench/impls next to this file
    if not impls_dir:
        this_file = os.path.abspath(__file__)
        impls_dir = os.path.join(os.path.dirname(this_file), "ogbench", "impls")

    impls_dir = os.path.abspath(impls_dir)

    if not os.path.isdir(impls_dir):
        print(
            f"ERROR: ogbench/impls not found at: {impls_dir}\n\n"
            "Either:\n"
            "  a) Set the OGBENCH_IMPLS environment variable:\n"
            "       export OGBENCH_IMPLS=/path/to/ogbench/impls\n\n"
            "  b) Clone ogbench next to this project:\n"
            "       git clone https://github.com/seohongpark/ogbench.git ../ogbench\n"
        )
        sys.exit(1)

    if impls_dir not in sys.path:
        sys.path.insert(0, impls_dir)

    try:
        import agents  # noqa: F401
        import ogbench  # noqa: F401
    except ImportError as e:
        print(
            f"ERROR: Found impls dir at {impls_dir} but import failed: {e}\n"
            "Make sure you have installed dependencies:\n"
            "    pip install -r ogbench/impls/requirements.txt\n"
        )
        sys.exit(1)

    print(f"  [setup] ogbench/impls loaded from: {impls_dir}")


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import jax
        jax.config.update("jax_default_matmul_precision", "float32")
    except Exception:
        pass


def build_hiql_agent_config(user: dict):
    """
    Full HIQL ConfigDict from agents/hiql.py defaults, merged with user CONFIG.
    State-based AntMaze: encoder=None (not a string — only pixel encoders like impala_small live in encoder_modules).
    """
    from agents.hiql import get_config

    cfg = get_config()
    cfg.encoder = None
    cfg.frame_stack = None

    for user_key, cfg_key in (
        ("subgoal_steps", "subgoal_steps"),
        ("expectile", "expectile"),
        ("high_alpha", "high_alpha"),
        ("low_alpha", "low_alpha"),
        ("actor_p_trajgoal", "actor_p_trajgoal"),
        ("actor_p_randomgoal", "actor_p_randomgoal"),
        ("actor_geom_sample", "actor_geom_sample"),
        ("discount", "discount"),
        ("batch_size", "batch_size"),
    ):
        if user_key in user:
            cfg[cfg_key] = user[user_key]

    return cfg


def make_agent(env, seed: int, agent_cfg):
    """
    Instantiate HIQL like impls/main.py: batched example obs/actions of shape (1, dim).
    """
    from agents import agents as agent_registry
    import numpy as np

    ex_obs = np.asarray([env.observation_space.sample()], dtype=np.float32)
    ex_act = np.asarray([env.action_space.sample()], dtype=np.float32)

    agent_cls = agent_registry["hiql"]
    return agent_cls.create(seed, ex_obs, ex_act, agent_cfg)


def make_dataset(train_dataset: dict, agent_cfg):
    """Same pattern as impls/main.py: Dataset.create then HGCDataset(dataset, config)."""
    from utils.datasets import Dataset, HGCDataset

    base = Dataset.create(**train_dataset)
    return HGCDataset(dataset=base, config=agent_cfg)


def evaluate(agent, env, num_episodes: int, agent_cfg) -> dict:
    """Average success over OGBench evaluation tasks (task_id 1..5 for AntMaze)."""
    import numpy as np
    from utils.evaluation import evaluate as ogbench_evaluate

    successes = []
    for task_id in range(1, 6):
        eval_stats, _, _ = ogbench_evaluate(
            agent,
            env,
            task_id=task_id,
            config=agent_cfg,
            num_eval_episodes=num_episodes,
            num_video_episodes=0,
        )
        successes.append(float(eval_stats.get("success", 0.0)))

    return {
        "success": float(np.mean(successes)),
        "per_task_success": successes,
    }


def save_checkpoint(agent, step: int, save_dir: str):
    from utils.flax_utils import save_agent

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_{step}")
    os.makedirs(path, exist_ok=True)
    save_agent(agent, path, step)
    print(f"  [checkpoint] Saved to {path}")


def train(config: dict):
    import ogbench

    # ── Environment + dataset ─────────────────────────────────────────────────
    print(f"Loading env and dataset: {config['env_name']}...")
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        config["env_name"],
        compact_dataset=False,  # keep next_observations for TD targets
    )
    env.reset(seed=config["seed"])
    print(f"  obs shape:    {env.observation_space.shape}")
    print(f"  action shape: {env.action_space.shape}")
    print(f"  dataset size: {len(train_dataset['observations']):,} transitions")

    # ── Agent + dataset (shared HIQL ConfigDict, matches impls/main.py) ───────
    agent_cfg = build_hiql_agent_config(config)

    print("Creating HIQL agent...")
    set_seeds(config["seed"])
    agent = make_agent(env, config["seed"], agent_cfg)

    print("Wrapping dataset...")
    dataset = make_dataset(train_dataset, agent_cfg)

    # ── W&B (optional) ────────────────────────────────────────────────────────
    if config["use_wandb"]:
        import wandb
        wandb.init(
            project=config["wandb_project"],
            config=config,
            name=f"hiql_{config['env_name']}_seed{config['seed']}",
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training for {config['train_steps']:,} steps...\n")
    os.makedirs(config["save_dir"], exist_ok=True)

    t0 = time.time()
    for step in range(1, config["train_steps"] + 1):

        # Sample a batch and update the agent
        batch = dataset.sample(config["batch_size"])
        agent, update_info = agent.update(batch)

        # ── Logging ───────────────────────────────────────────────────────────
        if step % 5_000 == 0:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            remaining = (config["train_steps"] - step) / steps_per_sec

            print(
                f"  step {step:>8,} / {config['train_steps']:,}"
                f"  |  {steps_per_sec:.0f} steps/s"
                f"  |  ETA {remaining/60:.1f} min"
                f"  |  value_loss {update_info.get('value_loss', float('nan')):.4f}"
            )

        # ── Evaluation ────────────────────────────────────────────────────────
        if step % config["eval_interval"] == 0:
            print(f"\n  [eval] step {step:,} ...")
            eval_stats = evaluate(agent, env, config["eval_episodes"], agent_cfg)
            success_rate = eval_stats.get("success", 0.0)
            mean_return = eval_stats.get("episode.return", 0.0)
            print(f"  [eval] success_rate={success_rate:.3f}  mean_return={mean_return:.2f}\n")

            if config["use_wandb"]:
                import wandb
                wandb.log({"eval/success_rate": success_rate, "eval/mean_return": mean_return}, step=step)

        # ── Checkpointing ─────────────────────────────────────────────────────
        if step % config["save_interval"] == 0:
            save_checkpoint(agent, step, config["save_dir"])

    # ── Final eval + save ─────────────────────────────────────────────────────
    print("\nFinal evaluation...")
    eval_stats = evaluate(agent, env, config["eval_episodes"], agent_cfg)
    print(f"  success_rate={eval_stats.get('success', 0.0):.3f}")
    save_checkpoint(agent, config["train_steps"], config["save_dir"])

    if config["use_wandb"]:
        import wandb
        wandb.finish()

    print("\nDone.")
    return agent


if __name__ == "__main__":
    setup_imports()
    train(CONFIG)