from torch import nn
from torch import Tensor


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.tanh(self.layer1(x))
        x = nn.functional.tanh(self.layer2(x))
        x = self.layer3(x)
        return x
