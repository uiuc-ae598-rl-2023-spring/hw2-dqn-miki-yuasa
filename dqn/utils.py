from itertools import count

import numpy as np
import torch
from torch import Tensor

from dqn.dqn import DQN
from discreteaction_pendulum import Pendulum


def compute_averaged_episodic_reward(
    policy_net: DQN, env: Pendulum, num_episodes: int
) -> float:
    """Compute the averaged episodic reward of the policy network on the given environment.

    Args:
        policy_net (DQN): The policy network.
        env (gym.Env): The environment.
        num_episodes (int): The number of episodes.
        device (torch.device): The device.

    Returns:
        float: The averaged episodic reward.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    averaged_episodic_reward: float = 0
    for _ in range(num_episodes):
        state_np: np.ndarray = env.reset()
        state: Tensor = torch.tensor(
            state_np, dtype=torch.float32, device=device
        ).unsqueeze(0)
        episodic_reward: float = 0
        for _ in count():
            action = policy_net(state).max(1)[1].view(1, 1)
            observation, reward, done = env.step(action.item())
            episodic_reward += reward
            if done:
                break
            else:
                state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
        averaged_episodic_reward += episodic_reward
    averaged_episodic_reward /= num_episodes
    return averaged_episodic_reward
