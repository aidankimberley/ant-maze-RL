import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)


ENV_ID = "AntMaze_UMaze-v5"
MAX_EPISODE_STEPS = 100
NUM_RANDOM_STEPS = 5


def describe_space(name: str, space) -> None:
    print(f"\n{name}:")
    print(f"  type: {type(space)}")
    print(f"  repr: {space}")


def describe_array(name: str, arr) -> None:
    arr_np = np.asarray(arr)
    print(f"{name}:")
    print(f"  shape: {arr_np.shape}")
    print(f"  dtype: {arr_np.dtype}")
    print(f"  min:   {arr_np.min():.6f}")
    print(f"  max:   {arr_np.max():.6f}")
    flat_preview = arr_np.flatten()[:10]
    print(f"  first 10 values: {flat_preview}")


def main() -> None:
    env = gym.make(
        ENV_ID,
        render_mode=None,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    print("=" * 80)
    print("ANTMAZE ENV INSPECTION")
    print("=" * 80)
    print(f"Env ID: {ENV_ID}")

    describe_space("Observation space", env.observation_space)
    describe_space("Action space", env.action_space)

    if hasattr(env.action_space, "shape"):
        print("\nAction space details:")
        print(f"  shape: {env.action_space.shape}")
        print(f"  low:   {env.action_space.low}")
        print(f"  high:  {env.action_space.high}")

    obs, info = env.reset()

    print("\n" + "=" * 80)
    print("RESET OUTPUT")
    print("=" * 80)
    print(f"obs type:  {type(obs)}")
    print(f"info type: {type(info)}")
    print(f"info keys: {list(info.keys())}")

    if isinstance(obs, dict):
        print("\nObservation is a dict with keys:")
        for key in obs.keys():
            print(f"  - {key}")

        print("\nPer-key observation details:")
        for key, value in obs.items():
            describe_array(key, value)

        try:
            flat_env = gym.wrappers.FlattenObservation(
                gym.make(
                    ENV_ID,
                    render_mode=None,
                    max_episode_steps=MAX_EPISODE_STEPS,
                )
            )
            flat_obs, flat_info = flat_env.reset()
            print("\nFlattened observation details:")
            describe_array("flattened_obs", flat_obs)
            print(f"flattened info keys: {list(flat_info.keys())}")
            flat_env.close()
        except Exception as exc:
            print(f"\nCould not flatten observation automatically: {exc}")
    else:
        print("\nObservation is not a dict.")
        describe_array("obs", obs)

    print("\n" + "=" * 80)
    print("RANDOM STEP INSPECTION")
    print("=" * 80)

    for step in range(NUM_RANDOM_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {step}")
        describe_array("sampled_action", action)
        print(f"reward:      {reward}")
        print(f"terminated:  {terminated}")
        print(f"truncated:   {truncated}")
        print(f"info keys:   {list(info.keys())}")

        if "success" in info:
            print(f"success:     {info['success']}")

        if isinstance(obs, dict):
            for key, value in obs.items():
                arr_np = np.asarray(value)
                print(f"obs[{key}] shape: {arr_np.shape}")

        if terminated or truncated:
            print("\nEpisode ended early.")
            break

    env.close()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()