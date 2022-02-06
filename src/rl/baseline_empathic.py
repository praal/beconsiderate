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
from environment.kitchen import OBJECTS1
from .rl import Policy

class BaseEmpathic(Policy):
    alpha: float
    default_q: Tuple[float, bool]
    gamma: float
    num_actions: int
    Q: Dict[Tuple[Hashable, ActionId], Tuple[float, bool]]
    rng: Random

    def __init__(self, alpha: float, gamma: float, epsilon: float, default_q: float,
                 num_actions: int, rng: Random, others_q, penalty: int, others_alpha, objects, problem_mood=1, others_dist = [1.0], others_init = [[1, 1]], our_alpha=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default_q = default_q, False
        self.num_actions = num_actions
        self.rng = rng
        self.Q = {}
        self.others_q = others_q
        self.penalty = penalty
        self.others_alpha = others_alpha
        self.our_alpha = our_alpha
        self.others_dist = others_dist
        self.objects = objects
        self.others_init = others_init
        self.problem_mood = problem_mood

    def clear(self):
        self.Q = {}

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
        if len(best_actions) > 1 and best_actions[0] == 4 and best_actions[1] == 5:
            return 4
        return self.rng.choice(best_actions)

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

    def get_train_action(self, state: State,
                         restrict: Optional[List[int]] = None, restrict_locations = [[3, 1], [1, 3]]) -> ActionId:
        if restrict is None:
            restrict = list(range(4))
            for i in range(len(restrict_locations)):
                if state.x == restrict_locations[i][0] and state.y == restrict_locations[i][1]:
                    restrict = list(range(self.num_actions))
        if self.rng.random() < self.epsilon:
            return self.rng.choice(restrict)
        else:
            return self.get_best_action(state, restrict)

    def get_max_q(self, uid, qval):
        max_q = qval.get((uid, 0), (-1000, False))
        for action in range(1, self.num_actions):
            q = qval.get((uid, action), (-1000, False))
            if q > max_q:
                max_q = q
        if max_q[0] == 0:
            print("zero!", uid)
        return max_q[0]

    def estimate_other(self, state, prev_state):
        ans = 0
        for other in range(len(self.others_q)):

            fact_list = [False] * len(self.objects)
            if self.problem_mood == 1 and state.facts[self.objects["fridge_open_now"]]:
                fact_list[4] = True
            if self.problem_mood == 2 and state.facts[self.objects["very_dirty"]]:
                fact_list[4] = True
            if self.problem_mood == 2 and state.facts[self.objects["dirty"]]:
                fact_list[3] = True
            if self.problem_mood == 3 and state.facts[self.objects["start_heat"]]:
                fact_list[4] = True
            if self.problem_mood == 4 and state.facts[self.objects["open"]]:
                fact_list[1] = True
            if self.problem_mood == 4 and state.facts[self.objects["up"]]:
                fact_list[4] = True

            facts = tuple(fact_list)
            uid = (self.others_init[other][0], self.others_init[other][1], facts)

            max_q = self.get_max_q(uid, self.others_q[other])

            ans += self.others_dist[other] * self.others_alpha[other] * max_q

        return ans

    def update(self, s0: State, a: ActionId, s1: State, r: float, end: bool):
        q = (1.0 - self.alpha) * self.Q.get((s0.uid, a), self.default_q)[0]
        if end:
            others = self.estimate_other(s1, s0)
            q += self.alpha * (self.our_alpha * r + self.gamma * others)
        else:
            q += self.alpha * (self.our_alpha * r + self.gamma * (self.estimate(s1)))

        self.Q[(s0.uid, a)] = q, True

    def reset(self, evaluation: bool):
        self.evaluation = evaluation

    def report(self) -> str:
        return "|Q| = {}".format(len(self.Q))

    def get_Q(self):
        return self.Q.copy()



