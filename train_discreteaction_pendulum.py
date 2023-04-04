# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer

import pickle
from matplotlib import pyplot as plt


from discreteaction_pendulum import Pendulum
from dqn.dqn import DQN
from dqn.train import train
from dqn.plot import generate_trajectory, plot_policy_value
from dqn.utils import compute_averaged_episodic_reward


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

num_episodes = 20000
memory_buffer_size: int = 10000
num_ablation_episodes: int = 1000

model_path: str = "models/dqn_w_rep_w_target.pt"
learning_curve_path: str = "figures/learning_curve_w_rep_w_target.png"
policy_plot_path: str = "figures/policy_w_rep_w_target.png"
value_plot_path: str = "figures/value_w_rep_w_target.png"

traj_filename: str = "figures/traj_w_rep_w_target.png"
video_filename: str = "figures/video_w_rep_w_target.gif"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.subplot.bottom"] = 0.15
plt.rcParams["figure.subplot.left"] = 0.15

env = Pendulum()

policy_net, policy_learning_curve = train(
    num_episodes,
    env,
    memory_buffer_size,
    GAMMA,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    BATCH_SIZE,
    TAU,
    LR,
)

plot_policy_value(policy_net, env, policy_plot_path, value_plot_path)
generate_trajectory(env, policy_net, traj_filename, video_filename)

replay_target_bool_pairs: list[tuple[bool, bool]] = [
    (True, False),
    (False, True),
    (False, False),
]

averaged_episodic_rewards: list[float] = [
    compute_averaged_episodic_reward(policy_net, env, num_ablation_episodes)
]
learning_curves: list[list[float]] = [policy_learning_curve]

for with_replay_memory, with_target_net in replay_target_bool_pairs:
    net, learning_curve = train(
        num_episodes,
        env,
        memory_buffer_size,
        GAMMA,
        EPS_START,
        EPS_END,
        EPS_DECAY,
        BATCH_SIZE,
        TAU,
        LR,
        with_replay_memory,
        with_target_net,
    )

    averaged_episodic_rewards.append(
        compute_averaged_episodic_reward(net, env, num_ablation_episodes)
    )
    learning_curves.append(learning_curve)

print(averaged_episodic_rewards)

fig, ax = plt.subplots()
labels: list[str] = [
    "with replay and target",
    "no target",
    "no replay",
    "no replay, no target",
]
for learning_curve, label in zip(learning_curves, labels):
    ax.plot(learning_curve, label=label)
ax.set_xlabel("Episodes [-]")
ax.set_ylabel("Episodic Reward [-]")
ax.legend()
plt.savefig("figures/learning_curve_ablation.png", dpi=600)
with open("figures/learning_curve_ablation.pkl", "wb") as f:
    pickle.dump([learning_curves, averaged_episodic_rewards], f)
