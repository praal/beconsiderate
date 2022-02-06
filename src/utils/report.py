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

from csv import writer
from sys import stdout
from typing import IO, Sequence

from environment import State


class Report:
    def __init__(self, logfile: IO, log_step: int, states: Sequence[State],
                 steps: int, trials: int):
        self.writer = writer(logfile, delimiter=' ')
        self.log_step = log_step
        self.states = states
        self.steps = steps
        self.trials = trials

    def evaluate(self, agent, step: int, force: bool = False):
        if not force and step % self.log_step != 0:
            return
        values = agent.evaluate(self.states, self.steps, self.trials,
                                name=str(step))

        if len(values) == 0:
            return

        mean = sum(values) / len(values)
        values.sort()

        self.writer.writerow([step, mean] + values)


class StdoutReport(Report):
    def __init__(self, log_step: int, states: Sequence[State], steps: int,
                 trials: int):
        super().__init__(stdout, log_step, states, steps, trials)


class SequenceReport(Report):
    start: int

    def __init__(self, logfile: IO, log_step: int, states: Sequence[State],
                 steps: int, trials: int):
        self.start = 0
        super().__init__(logfile, log_step, states, steps, trials)

    def evaluate(self, agent, step: int, force: bool = False):
        if force or step % self.log_step == 0:
            super().evaluate(agent, step + self.start, force=True)

    def increment(self, steps: int):
        self.start += steps + 1
