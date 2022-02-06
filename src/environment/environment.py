#Copyright (c) 2022 Be Considerate: Avoiding Negative Side Effects
#in Reinforcement Learning Authors

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from abc import ABC, abstractmethod
from typing import FrozenSet, Generic, Hashable, Optional, Tuple, TypeVar

ActionId = int
Observation = FrozenSet[str]


class State(ABC):
    uid: Hashable

    @abstractmethod
    def __str__(self) -> str:
        pass


S = TypeVar('S', bound=State)


class Environment(ABC, Generic[S]):
    num_actions: int = 0
    state: S

    def __init__(self, state: S):
        self.state = state

    @abstractmethod
    def apply_action(self, a: ActionId):
        pass

    @abstractmethod
    def cost(self, s0: S, a: ActionId, s1: S) -> float:
        pass

    @abstractmethod
    def observe(self, state: S) -> Observation:
        pass

    @abstractmethod
    def reset(self, state: Optional[S] = None):
        pass


class RewardFn(ABC, Generic[S]):
    environment: Environment

    def __init__(self, environment: Environment):
        self.environment = environment

    @abstractmethod
    def __call__(self, s0: S, a: ActionId, s1: S) -> Tuple[float, bool]:
        pass

    @abstractmethod
    def reset(self):
        pass
