import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make("AntMaze_UMaze-v5", render_mode="human", max_episode_steps=1000)
env.reset()
for i in range(5000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: Reward = {reward}, Terminated = {terminated}, Truncated = {truncated}")   
    if terminated or truncated:
        break

env.close()