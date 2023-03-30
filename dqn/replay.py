from collections import deque
import random
from typing import NamedTuple


class Transition(NamedTuple):
    state: object
    action: object
    reward: float
    next_state: object


class ReplayMemory(object):
    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
