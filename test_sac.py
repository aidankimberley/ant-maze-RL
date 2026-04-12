import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sac_continuous_action

gym.register_envs(gymnasium_robotics)


def make_env():
    def thunk():
        env = gym.make("AntMaze_UMaze-v5", render_mode="human", max_episode_steps=100)
        env = gym.wrappers.FlattenObservation(env)  # Dict obs → Box(109,) for MLP Actor
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv([make_env()])
obs, _ = envs.reset()
actor = sac_continuous_action.Actor(envs)

# Smoke test: policy forward + one env step (untrained weights — just checks shapes / API)
with torch.no_grad():
    action, _, _ = actor.get_action(torch.as_tensor(obs, dtype=torch.float32))
obs, reward, term, trunc, _ = envs.step(action.numpy())
print("OK — obs", obs.shape, "action", action.shape, "reward", reward)

envs.close()
