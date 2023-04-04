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
    s: list[ndarray] | ndarray
    a: list[int]
    r: list[float]


def generate_trajectory(
    env: Pendulum,
    policy: DQN,
    traj_filename: str,
    video_filename: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        a = wrapped_policy(s)
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


def plot_policy_value(
    policy_net: DQN, env: Pendulum, policy_plot_path: str, value_plot_path: str
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyWrapper(policy_net, device)
    theta = np.linspace(-np.pi, np.pi, 100)
    theta_dot = np.linspace(-env.max_thetadot, env.max_thetadot, 100)
    x_axis, y_axis = np.meshgrid(theta, theta_dot)

    policy_array = np.zeros_like(x_axis)
    for i in range(len(theta)):
        for j in range(len(theta)):
            s = np.array((x_axis[i, j], y_axis[i, j]))
            policy_array[i, j] = policy(s)

    V_array = np.zeros_like(x_axis)

    for i in range(len(theta)):
        for j in range(len(theta)):
            s = np.array((x_axis[i, j], y_axis[i, j]))
            V_array[i, j] = torch.max(
                policy_net(torch.tensor(s, device=device).float())
            ).item()

    plt.figure()
    plt.pcolor(x_axis, y_axis, policy_array)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.colorbar()
    plt.savefig(policy_plot_path, dpi=600)

    plt.figure()
    plt.pcolor(x_axis, y_axis, V_array)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.colorbar()
    plt.savefig(value_plot_path, dpi=600)
