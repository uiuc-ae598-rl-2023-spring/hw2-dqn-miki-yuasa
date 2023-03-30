import random
from typing import TypedDict
from matplotlib import pyplot as plt

import numpy as np
from numpy import ndarray
import torch

from discreteaction_pendulum import Pendulum
from dqn.dqn import DQN


class PolicyWrapper:
    def __init__(self, policy: DQN, device: torch.device):
        self.policy = policy
        self.device = device

    def __call__(self, state: ndarray) -> int:
        return (
            self.policy(
                torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
                    0
                )
            )
            .max(1)[1]
            .item()
        )


class DataDict(TypedDict):
    t: list[int]
    s: list[ndarray] |ndarray
    a: list[int]
    r: list[float]


def generate_trajectory(
    env: Pendulum,
    policy: DQN,
    traj_filename: str,
    video_filename: str,
    device: torch.device,
) -> None:
    wrapped_policy = PolicyWrapper(policy, device)
    env.video(wrapped_policy, filename=video_filename)

    # Initialize simulation
    s = env.reset()

    # Create dict to store data from simulation
    data: DataDict = {
        "t": [0],
        "s": [s],
        "a": [],
        "r": [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = random.randrange(env.num_actions)
        (s, r, done) = env.step(a)
        data["t"].append(data["t"][-1] + 1)
        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)

    # Parse data from simulation
    data["s"] = np.array(data["s"])
    theta = data["s"][:, 0]
    thetadot = data["s"][:, 1]
    tau = [env._a_to_u(a) for a in data["a"]]

    # Plot data and save to png file
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(data["t"], theta, label="theta")
    ax[0].plot(data["t"], thetadot, label="thetadot")
    ax[0].legend()
    ax[1].plot(data["t"][:-1], tau, label="tau")
    ax[1].legend()
    ax[2].plot(data["t"][:-1], data["r"], label="r")
    ax[2].legend()
    ax[2].set_xlabel("time step")
    plt.tight_layout()
    plt.savefig(traj_filename)
