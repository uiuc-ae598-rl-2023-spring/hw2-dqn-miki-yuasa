from collections import deque
import random
from typing import NamedTuple

from torch import Tensor


class Transition(NamedTuple):
    state: Tensor | tuple[Tensor, ...] | None
    action: Tensor | tuple[Tensor, ...]
    reward: float | tuple[Tensor, ...] | Tensor
    next_state: Tensor | tuple[Tensor, ...] | None


class ReplayMemory(object):
    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
