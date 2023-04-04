import torch
from discreteaction_pendulum import Pendulum

from dqn.dqn import DQN
from dqn.plot import generate_trajectory

env = Pendulum()

n_actions: int = env.num_actions
n_observations: int = len(env.reset())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load("models/dqn_w_rep_w_target.pt"))

traj_filename: str = "figures/traj_w_rep_w_target.png"
video_filename: str = "figures/video_w_rep_w_target.gif"
generate_trajectory(env, policy_net, traj_filename, video_filename)
