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
]

key_IND = 0

OBJECTS1 = dict([(v, k) for k, v in enumerate(
    ["key", "wood", "hammer", "box", "extra", "extrawood"])])


OBJECTS2 = dict([(v, k) for k, v in enumerate(
    ["doll", "played"])])  # , "wall"])])

OBJECTS3 = dict([(v, k) for k, v in enumerate(["target", "fence"])])

OBJECTS4 = dict([(v, k) for k, v in enumerate(["key", "target"])])


def update_facts(problem_mood,facts: Sequence[bool], objects: Observation, is_key,tool_in_fac=False, wood_in_fac = False, put_extra=False):
    state = set([i for i, v in enumerate(facts) if v])
    key_change = 0
    if problem_mood == 1:

        if tool_in_fac:
            state.add(OBJECTS1["extra"])

        if wood_in_fac:
            state.add(OBJECTS1["extrawood"])
        for o in objects:

            if o == "key":
                if OBJECTS1["key"] in state:
                    state.remove(OBJECTS1["key"])
                    key_change = 1

                elif is_key:
                    state.add(OBJECTS1[o])
                    key_change = -1
            elif o in OBJECTS1:
                if o == "wood" and put_extra:
                    state.add(OBJECTS1["extrawood"])
                state.add(OBJECTS1[o])

        if "warehouse" in objects:
            if "hammer" in OBJECTS1:
                state.add((OBJECTS1["hammer"]))
            if put_extra:
                state.add(OBJECTS1["extra"])

        if "factory" in objects:
            if OBJECTS1["wood"] in state and OBJECTS1["key"] in state and OBJECTS1["hammer"] in state:
                state.add(OBJECTS1["box"])
            if OBJECTS1["wood"] in state and OBJECTS1["key"] in state and OBJECTS1["extra"] in state:
                state.add(OBJECTS1["box"])
            if OBJECTS1["extrawood"] in state and OBJECTS1["key"] in state and OBJECTS1["hammer"] in state:
                state.add(OBJECTS1["box"])
            if OBJECTS1["extrawood"] in state and OBJECTS1["key"] in state and OBJECTS1["extra"] in state:
                state.add(OBJECTS1["box"])

        return state, key_change

    elif problem_mood == 2:
        if "key" in objects and is_key:
            state.add(OBJECTS2["played"])
            state.add(OBJECTS2["doll"])
            key_change = -1

        if OBJECTS2["doll"] in state and put_extra:
            state.remove(OBJECTS2["doll"])
            key_change = 1
        return state, key_change
    elif problem_mood == 3:
        if "key" in objects and is_key:
            state.add(OBJECTS3["target"])
        if "fence" in objects and put_extra:
            state.add(OBJECTS3["fence"])
        return state, 0
    elif problem_mood == 4:
        if "key" in objects and is_key:
            state.add(OBJECTS4["key"])
            key_change = -1
        elif OBJECTS4["key"] in state and "key" in objects:
            state.remove(OBJECTS4["key"])
            key_change = 1
        elif "factory" in objects and OBJECTS4["key"] in state:
            state.add(OBJECTS4["target"])
        return state, key_change


class CraftState(State):
    facts: Tuple[bool, ...]
    map_data: Tuple[Tuple[Observation, ...], ...]


    def __init__(self, x: int, y: int, key_x, key_y,  facts: Set[int], default_x, default_y, problem_mood):
        self.x = x
        self.y = y
        self.default_x = default_x
        self.default_y = default_y
        self.problem_mood = problem_mood
        if self.problem_mood == 1:
            fact_list = [False] * len(OBJECTS1)
        else:
            fact_list = [False] * len(OBJECTS2)
        for fact in facts:
            fact_list[fact] = True
        self.facts = tuple(fact_list)

        self.key_x = key_x
        self.key_y = key_y
        self.uid = (self.x, self.y, key_x, key_y, self.facts)

    def __str__(self) -> str:
        return "({:2d}, {:2d}, {:2d}, {:2d}, {})".format(self.x, self.y, self.key_x, self.key_y, self.facts)

    @staticmethod
    def random(problem_mood, rng: Random,
               map_data: Sequence[Sequence[Observation]], key_locations: List[List[int]], default_x, default_y, tool_in_fact = False, wood_in_fact=False, fence=False) -> 'CraftState':

        while True:
            x = default_x
            y = default_y
            ind = rng.randrange(len(key_locations))
            key_y = key_locations[ind][0]
            key_x = key_locations[ind][1]
            if "wall" not in map_data[y][x] and "wall" not in map_data[key_y][key_x]:
                next_tool = False
                next_wood = False
                next_fence = False
                if tool_in_fact:
                    if rng.random() < 0.5:
                        next_tool = True
                if wood_in_fact:
                    if rng.random() < 0.5:
                        next_wood = True
                if fence:
                    if rng.random() < 0.5:
                        next_fence = True
                facts, _ = update_facts(problem_mood, (), map_data[y][x], 0, next_tool, next_wood)
                if next_fence:
                    facts.add(1)
                return CraftState(x, y, key_x, key_y, facts, default_x, default_y, problem_mood)


MAPPING: Mapping[str, FrozenSet[str]] = {
    'A': frozenset(),
    'X': frozenset(["wall"]),
    'a': frozenset(["wood"]),
    'b': frozenset(["toolshed"]),
    'c': frozenset(["workbench"]),
    'e': frozenset(["factory"]),
    'f': frozenset(["iron"]),
    'k': frozenset(["key"]),
    'g': frozenset(["gold"]),
    'w': frozenset(["warehouse"]),
    'D': frozenset(["door"]),
    'h': frozenset(["gem"]),
    'p': frozenset(["fence"]),
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


class Craft(Environment[CraftState]):
    map_data = [[]]
    num_actions = 4

    def __init__(self, map_fn: str, rng: Random, default_x, default_y, objects, problem_mood, tool_in_fact = False, wood_in_fact = False, fence = False, noise = 0.0):
        self.problem_mood = problem_mood
        self.map_data = load_map(map_fn)
        self.height = len(self.map_data)
        self.width = len(self.map_data[0])
        self.rng = rng
        self.key_locations = self.get_all_item()
        self.default_x = default_x
        self.default_y = default_y
        self.noise = noise
        self.tool_in_fact_default = tool_in_fact
        self.wood_in_fact_default = wood_in_fact
        self.objects = objects
        self.fence = fence
        super().__init__(CraftState.random(self.problem_mood, self.rng, self.map_data, self.key_locations, default_x, default_y, self.tool_in_fact_default, self.wood_in_fact_default, self.fence))

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
        if "fence" in self.map_data[y][x] and self.state.facts[self.objects["fence"]]:

            return
        objects = self.map_data[y][x]

        is_key = False
        if x == self.state.key_x and y == self.state.key_y:
            is_key = True

        put_extra = False
        if a == 4:
            put_extra = True
        new_facts, key_change = update_facts(self.problem_mood, self.state.facts, objects, is_key, False, False, put_extra)

        new_key_x = self.state.key_x
        new_key_y = self.state.key_y

        if key_change == 1:

            new_key_x = x
            new_key_y = y


        elif key_change == -1:

            new_key_x = -1
            new_key_y = -1

        self.state = CraftState(x, y, new_key_x, new_key_y, new_facts, self.default_x, self.default_y, self.problem_mood)

        logging.debug("success, current state is %s", self.state)

    def cost(self, s0: CraftState, a: ActionId, s1: CraftState) -> float:
        if "extrawood" in self.objects and not s0.facts[self.objects["extrawood"]] and s1.facts[self.objects["extrawood"]] and a == 4:
            return 12.0
        if "fence" in self.objects and not s0.facts[self.objects["fence"]] and s1.facts[self.objects["fence"]] and a == 4:
            return 50.0
        return 1.0

    def observe(self, state: CraftState) -> Observation:
        return self.map_data[self.state.y][self.state.x]

    def reset(self, state: Optional[CraftState] = None):
        if state is not None:
            self.state = state
        else:
            self.state = CraftState.random(self.problem_mood, self.rng, self.map_data, self.key_locations, self.default_x, self.default_y, self.tool_in_fact_default, self.wood_in_fact_default, self.fence)
