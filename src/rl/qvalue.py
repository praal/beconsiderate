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

from random import Random
from typing import Dict, Hashable, List, Optional, Tuple

from environment import ActionId, State

from .rl import Policy


class Greedy(Policy):
    alpha: float
    default_q: Tuple[float, bool]
    gamma: float
    num_actions: int
    Q: Dict[Tuple[Hashable, ActionId], Tuple[float, bool]]
    rng: Random

    def __init__(self, alpha: float, gamma: float, default_q: float,
                 num_actions: int, rng: Random):
        self.alpha = alpha
        self.gamma = gamma
        self.default_q = default_q, False
        self.num_actions = num_actions
        self.rng = rng
        self.Q = {}

    def clear(self):
        self.Q = {}

    def get_Q(self):
        return self.Q.copy()

    def estimate(self, state: State) -> float:

        max_q = self.Q.get((state.uid, 0), self.default_q)
        for action in range(1, self.num_actions):
            q = self.Q.get((state.uid, action), self.default_q)
            if q > max_q:
                max_q = q
        return max_q[0]

    def get_best_action(self, state: State,
                        restrict: Optional[List[int]] = None) -> ActionId:
        if restrict is None:
            restrict = list(range(self.num_actions))
        max_q = self.Q.get((state.uid, restrict[0]), self.default_q)
        best_actions = [restrict[0]]
        for action in restrict[1:]:
            q = self.Q.get((state.uid, action), self.default_q)
            if q > max_q:  # or (self.evaluation and q[1] and not max_q[1]):
                max_q = q
                best_actions = [action]
            elif q == max_q:
                best_actions.append(action)
        return self.rng.choice(best_actions)

    def get_train_action(self, state: State,
                         restrict: Optional[List[int]] = None) -> ActionId:
        return self.get_best_action(state, restrict)

    def update(self, s0: State, a: ActionId, s1: State, r: float, end: bool):
        q = (1.0 - self.alpha) * self.Q.get((s0.uid, a), self.default_q)[0]
        if end:
            q += self.alpha * r
        else:
            q += self.alpha * (r + self.gamma * self.estimate(s1))

        self.Q[(s0.uid, a)] = q, True

    def reset(self, evaluation: bool):
        self.evaluation = evaluation

    def get_policy(self):
        ans = {}
        for state,_ in self.Q:
            max_q = self.Q.get((state, 0), self.default_q)
            max_action = 0
            for action in range(1, self.num_actions):
                q = self.Q.get((state, action), self.default_q)
                if q > max_q:
                    max_q = q
                    max_action = action
            ans[state] = max_action
        return ans

    def report(self) -> str:
        return "|Q| = {}".format(len(self.Q))


class EpsilonGreedy(Greedy):
    epsilon: float

    def __init__(self, alpha: float, gamma: float, epsilon: float,
                 default_q: float, num_actions: int, rng: Random):
        self.epsilon = epsilon
        super().__init__(alpha, gamma, default_q, num_actions, rng)

    def get_train_action(self, state: State,
                         restrict: Optional[List[int]] = None, restrict_locations = [[3, 1], [1, 3]]) -> ActionId:
        if self.rng.random() < self.epsilon:
            if restrict is None:
                if self.num_actions == 5:
                    restrict = list(range(self.num_actions - 1))
                    for i in range(len(restrict_locations)):
                        if state.x == restrict_locations[i][0] and state.y == restrict_locations[i][1]:
                            restrict = list(range(self.num_actions))
                else:
                    restrict = list(range(self.num_actions))
            return self.rng.choice(restrict)
        else:
            return self.get_best_action(state, restrict)
