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

from typing import Iterable, Tuple, Union

from .craft import Craft, CraftState
from .kitchen import Kitchen, KitchenState
from .environment import ActionId, RewardFn


CraftKitchen = Union[Craft, Kitchen]
CraftKitchenState = Union[CraftState, KitchenState]


class ReachFacts(RewardFn):
    target: Tuple[int, ...]

    def __init__(self, environment: CraftKitchen, facts: Iterable[int], notfacts: Iterable[int], problem_mood=1):
        super().__init__(environment)
        self.target = tuple(facts)
        self.nottargets = tuple(notfacts)
        self.problem_mood = problem_mood

    def __call__(self, s0: CraftKitchenState, a: ActionId,
                 s1: CraftKitchenState) -> Tuple[float, bool]:
        cost = self.environment.cost(s0, a, s1)

        for fact in self.target:
            if not s1.facts[fact]:
                return -cost, False
        for fact in self.nottargets:
            if s1.facts[fact]:
                return -cost, False

        if isinstance(s0, CraftState):

            if self.problem_mood == 1:
                if s1.x == 1 and s1.y == 1:
                    return -cost, True
                elif s1.x == self.environment.width - 2 and s1.y == self.environment.height - 2:
                    return -cost, True
                return -cost, False
            else:
                return -cost, True
        else:

            if self.environment.problem_mood == 2:
                if s1.x == self.environment.default_x and s1.y == self.environment.default_y:
                    return -cost, True
                return -cost, False

            elif s1.x == self.environment.width - 2 and s1.y == self.environment.height - 2:
                if cost == 0.0:
                    return 1.0, True
                return -cost, True

        return -cost, False

    def reset(self):
        pass
