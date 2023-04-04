# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer

from matplotlib import pyplot as plt


from discreteaction_pendulum import Pendulum
from dqn.dqn import DQN
from dqn.train import train
from dqn.plot import generate_trajectory, plot_policy_value


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

num_episodes = 5
memory_buffer_size: int = 10000

model_path: str = "models/dqn_w_rep_w_target.pt"
learning_curve_path: str = "figures/learning_curve_w_rep_w_target.png"
policy_plot_path: str = "figures/policy_w_rep_w_target.png"
value_plot_path: str = "figures/value_w_rep_w_target.png"

traj_filename: str = "figures/traj_w_rep_w_target.png"
video_filename: str = "figures/video_w_rep_w_target.gif"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.subplot.bottom"] = 0.15

env = Pendulum()

policy_net: DQN = train(
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
    model_path,
    learning_curve_path,
)

plot_policy_value(policy_net, env, policy_plot_path, value_plot_path)
generate_trajectory(env, policy_net, traj_filename, video_filename)
