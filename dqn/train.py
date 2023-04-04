from itertools import count
import random, math
from matplotlib import pyplot as plt

from numpy import ndarray
import torch
from torch import Tensor, nn, optim

from discreteaction_pendulum import Pendulum
from dqn.dqn import DQN
from dqn.replay import ReplayMemory, Transition


def select_action(
    state: Tensor,
    steps_done: int,
    policy_net: DQN,
    env: Pendulum,
    device: torch.device,
    EPS_START: float,
    EPS_END: float,
    EPS_DECAY: float,
) -> Tensor:
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.choice(list(range(env.num_actions)))]],
            device=device,
            dtype=torch.long,
        )


def optimize_model(
    memory: ReplayMemory,
    policy_net: DQN,
    target_net: DQN,
    optimizer: torch.optim.AdamW,
    device: torch.device,
    BATCH_SIZE: int,
    GAMMA: float,
):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss: torch.Tensor = criterion(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train(
    num_episodes: int,
    env: Pendulum,
    memory_buffer: int,
    GAMMA: float,
    EPS_START: float,
    EPS_END: float,
    EPS_DECAY: float,
    BATCH_SIZE: int,
    TAU: float,
    LR: float,
    model_path: str,
    learning_curve_plot_path: str,
) -> DQN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_actions: int = env.num_actions
    n_observations: int = len(env.reset())

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(memory_buffer)

    episodic_rewards: list[float] = []
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        print(f"Episode {i_episode} of {num_episodes}")
        state_np: ndarray = env.reset()
        state: Tensor = torch.tensor(
            state_np, dtype=torch.float32, device=device
        ).unsqueeze(0)

        steps_done: int = 0
        episodic_reward: float = 0
        for t in count():
            action = select_action(
                state,
                steps_done,
                policy_net,
                env,
                device,
                EPS_START,
                EPS_END,
                EPS_DECAY,
            )
            observation, reward, done = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            episodic_reward += reward.item()

            if done:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            if next_state is not None and state.dim() == 1:
                pass
            # Store the transition in memory
            memory.push(Transition(state, action, reward, next_state))

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(
                memory, policy_net, target_net, optimizer, device, BATCH_SIZE, GAMMA
            )

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break
            else:
                pass

        episodic_rewards.append(episodic_reward)

    print("Complete")

    torch.save(policy_net.state_dict(), model_path)

    ax, fig = plt.subplots()
    plt.plot(episodic_rewards)
    plt.xlabel("Episode [-]")
    plt.ylabel("Episodic reward [-]")
    plt.savefig(learning_curve_plot_path)

    return policy_net
