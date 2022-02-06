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

import logging
from random import Random
from typing import FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from .environment import ActionId, Environment, Observation, State

ACTIONS: List[Tuple[int, int]] = [
    (0, 1),   # down
    (0, -1),  # up
    (-1, 0),  # left
    (1, 0),   # right
    (0, 0),
    (0, 0),
]


OBJECTS1 = dict([(v, k) for k, v in enumerate(
    ["fridge", "oven", "food", "fridge_open_now", "fridge_open_prev"])])

OBJECTS2 = dict([(v, k) for k, v in enumerate(
    ["fridge", "oven", "food", "dirty", "very_dirty"])])

OBJECTS3 = dict([(v, k) for k, v in enumerate(
    ["fridge", "oven", "food", "start_heat", "hot"])])

OBJECTS4 = dict([(v, k) for k, v in enumerate(
    ["fridge",  "open_prev", "food", "open", "up" ])])


def update_facts(problem_mood, facts: Sequence[bool], objects: Observation, allergic = False, extra_move=False, baking = False, down=False, short=False):
    state = set([i for i, v in enumerate(facts) if v])

    if problem_mood == 1:
        for o in objects:
            if o in OBJECTS1:
                state.add(OBJECTS1[o])
            if o == "fridge":
                state.add(OBJECTS1["fridge_open_now"])
                if extra_move:
                    state.remove(OBJECTS1["fridge_open_now"])
        if OBJECTS1["oven"] in state and OBJECTS1["fridge"] in state and OBJECTS1["fridge_open_prev"] not in state:
            state.add(OBJECTS1["food"])
    if problem_mood == 2:
        for o in objects:
            if o in OBJECTS2:
                state.add(OBJECTS2[o])
            if o == "oven" and OBJECTS2["food"] not in state and OBJECTS2["oven"] in state and OBJECTS2["fridge"] in state:
                if OBJECTS2["very_dirty"] not in state and not (allergic and OBJECTS2["dirty"] in state):
                    state.add(OBJECTS2["food"])
                    state.add(OBJECTS2["very_dirty"])

        if extra_move:
            if OBJECTS2["very_dirty"] in state:
                state.remove(OBJECTS2["very_dirty"])
                state.add(OBJECTS2["dirty"])
            elif OBJECTS2["dirty"] in state:
                state.remove(OBJECTS2["dirty"])
    if problem_mood == 3:
        for o in objects:
            if o in OBJECTS3:
                state.add(OBJECTS3[o])
            if o == "oven" and OBJECTS3["fridge"] in state:
                if baking:
                    if OBJECTS3["hot"] in state:
                        state.add(OBJECTS3["food"])
                else:
                    state.add(OBJECTS3["food"])

            if o == "oven" and extra_move:
                if OBJECTS3["start_heat"] in state:
                    state.remove(OBJECTS3["start_heat"])
                    state.add(OBJECTS3["hot"])
                    if OBJECTS3["fridge"] in state:
                        state.add(OBJECTS3["food"])

                else:
                    state.add(OBJECTS3["start_heat"])

    if problem_mood == 4:
        for o in objects:
            if o == "fridge" and OBJECTS4["open"] not in state:
                if short and OBJECTS4["up"] in state:
                    if extra_move:
                        state.remove(OBJECTS4["up"])
                        state.add(OBJECTS4["open"])
                        state.add(OBJECTS4["fridge"])
                    continue
                state.add(OBJECTS4["open"])
                state.add(OBJECTS4["fridge"])

            elif o == "fridge" and OBJECTS4["open"] in state:
                if extra_move:
                    state.remove(OBJECTS4["open"])
                    state.add(OBJECTS4["up"])
                if down:
                    state.remove(OBJECTS4["up"])

            elif o == "oven" and OBJECTS4["fridge"] in state and OBJECTS4["open_prev"] not in state:
                state.add(OBJECTS4["food"])

    return state


class KitchenState(State):
    facts: Tuple[bool, ...]
    map_data: Tuple[Tuple[Observation, ...], ...]

    def __init__(self, x: int, y: int, facts: Set[int], default_x, default_y):
        self.x = x
        self.y = y
        self.default_x = default_x
        self.default_y = default_y
        fact_list = [False] * len(OBJECTS1)

        for fact in facts:
            fact_list[fact] = True
        self.facts = tuple(fact_list)

        self.uid = (self.x, self.y, self.facts)

    def __str__(self) -> str:
        return "({:2d}, {:2d}, {:2d}, {:2d}, {})".format(self.x, self.y, self.facts)

    @staticmethod
    def random(problem_mood, rng: Random,
               map_data: Sequence[Sequence[Observation]], default_x, default_y, randomness=True) -> 'KitchenState':

        while True:
            x = default_x
            y = default_y
            if "wall" not in map_data[y][x]:
                facts = update_facts(problem_mood, (), map_data[y][x])
                if randomness:
                    if problem_mood == 1:
                        rnd = rng.randrange(2)
                        if rnd == 1:#fridge open
                            facts.add(4)
                    elif problem_mood == 2:
                        rnd = rng.randrange(3)
                        if rnd == 1:#dirty
                            facts.add(3)
                        elif rnd == 2:#very_dirty
                            facts.add(4)
                    elif problem_mood == 3:
                        rnd = rng.randrange(2)
                        if rnd == 1:#hot
                            facts.add(4)
                    elif problem_mood == 4:
                        rnd = rng.randrange(3)
                        if rnd == 1:#open
                            facts.add(3)
                        if rnd == 2: #up
                            facts.add(4)
                return KitchenState(x, y, facts, default_x, default_y)


MAPPING: Mapping[str, FrozenSet[str]] = {
    'A': frozenset(),
    'X': frozenset(["wall"]),
    'a': frozenset(["wood"]),
    'f': frozenset(["fridge"]),
    'o': frozenset(["oven"]),
    ' ': frozenset(),
    }


def load_map(map_fn: str):
    with open(map_fn) as map_file:
        array = []
        for l in map_file:
            if len(l.rstrip()) == 0:
                continue

            row = []
            for cell in l.rstrip():
                row.append(MAPPING[cell])
            array.append(row)

    return array


class Kitchen(Environment[KitchenState]):
    map_data = [[]]
    num_actions = 4

    def __init__(self, map_fn: str, rng: Random, default_x, default_y, problem_mood, allergic = False, baking = False, short = False, noise = 0.0):
        self.map_data = load_map(map_fn)
        self.height = len(self.map_data)
        self.width = len(self.map_data[0])
        self.rng = rng
        self.key_locations = self.get_all_item()
        self.default_x = default_x
        self.default_y = default_y
        self.noise = noise
        self.problem_mood = problem_mood
        self.allergic = allergic
        self.baking = baking
        self.short = short

        super().__init__(KitchenState.random(problem_mood, self.rng, self.map_data, default_x, default_y))


    def get_all_item(self, item="key"):
        ans = []
        for y in range(self.height):
            for x in range(self.width):
                if item in self.map_data[y][x]:
                    ans.append([y, x])
        return ans

    def apply_action(self, a: ActionId):
        if self.rng.random() < self.noise:
            a = self.rng.randrange(self.num_actions)

        x, y = self.state.x + ACTIONS[a][0], self.state.y + ACTIONS[a][1]
        logging.debug("applying action %s:%s", a, ACTIONS[a])
        if x < 0 or y < 0 or x >= self.width or y >= self.height or \
                "wall" in self.map_data[y][x] :
            return
        objects = self.map_data[y][x]

        extra_move = False
        put_down = False
        if a >= 4:
            extra_move = True
            if a >= 5:
                put_down = True
        new_facts = update_facts(self.problem_mood, self.state.facts, objects, allergic=self.allergic, extra_move=extra_move, baking = self.baking, down=put_down, short=self.short)

        self.state = KitchenState(x, y, new_facts, self.default_x, self.default_y)
        logging.debug("success, current state is %s", self.state)

    def cost(self, s0: KitchenState, a: ActionId, s1: KitchenState) -> float:
        return 1.0

    def observe(self, state: KitchenState) -> Observation:
        return self.map_data[self.state.y][self.state.x]

    def reset(self, state: Optional[KitchenState] = None):
        if state is not None:
            self.state = state
        else:
            self.state = KitchenState.random(self.problem_mood, self.rng, self.map_data, self.default_x, self.default_y, randomness=True)
