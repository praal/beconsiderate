import sys  # noqa
from os import path as p  # noqa


from rl.qvalue import EpsilonGreedy
from rl.rl import Agent
sys.path.append(p.abspath(p.join(p.dirname(sys.modules[__name__].__file__),
                                 "..")))  # noqa
import logging
from os import path
from random import Random
from time import time


from environment import ReachFacts
from environment.kitchen import Kitchen, KitchenState, OBJECTS3, update_facts
from utils.report import SequenceReport
from rl.baseline_empathic import BaseEmpathic


DEFAULT_Q = -1000.0

TOTAL_STEPS1 = 100000
TOTAL_STEPS2 = 300000

EPISODE_LENGTH = 1000
TEST_EPISODE_LENGTH = 50
LOG_STEP = 10000
TRIALS = 5
START_TASK = 0
END_TASK = 3
logging.basicConfig(level=logging.INFO)


def print_state(state, action):
    if action == 0:
        print("Agent Location:", state.x, state.y, "Action: Down", state.facts)
    elif action == 1:
        print("Agent Location:", state.x, state.y, "Action: Up", state.facts)
    elif action == 2:
        print("Agent Location:", state.x, state.y,  "Action: Left", state.facts)
    elif action == 3:
        print("Agent Location:", state.x, state.y,  "Action: Right", state.facts)
    elif action == 4:
        print("Agent Location:", state.x, state.y,  "Action: Heat", state.facts)
    else:
        print("Agent Location:", state.x, state.y)


def evaluate_agent(env, policy1, reward1, init):
    print("Evaluation:")
    state_rewards = []
    for initial_state1 in init:
        env.reset(initial_state1)
        reward1.reset()
        policy1.reset(evaluation=True)

        trial_reward: float = 0.0

        for step in range(TEST_EPISODE_LENGTH):
            s0 = env.state
            a = policy1.get_best_action(s0)
            env.apply_action(a)
            s1 = env.state
            print_state(s0, a)
            step_reward, finished = reward1(s0, a, s1)
            if not finished:
                trial_reward += step_reward
            logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)
            if finished:
                print_state(s1, -1)
                break

        state_rewards.append(trial_reward)
        return trial_reward


def evaluate_second_agent(env, policy1, reward1,env2, policy2, reward2):

    print("Acting agent enters")
    init = [1, 1]
    initial_state = KitchenState(init[0], init[1], set(), init[0], init[1])
    env.reset(initial_state)
    reward1.reset()
    policy1.reset(evaluation=True)
    trial_reward: float = 0.0
    for step in range(TEST_EPISODE_LENGTH):
        s0 = env.state
        a = policy1.get_best_action(s0)
        env.apply_action(a)
        s1 = env.state
        print_state(s0, a)
        step_reward, finished = reward1(s0, a, s1)
        if not finished:
            trial_reward += step_reward
        logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)
        if finished:
            print_state(s1, -1)
            break

    facts = set()
    if env.state.facts[OBJECTS3["start_heat"]]:
        facts.add(4)

    print("Agent 2 enters")
    env2.reset(KitchenState(init[0], init[1], facts, init[0], init[1]))

    trial_reward2 = 0
    for step in range(TEST_EPISODE_LENGTH):
        s0 = env2.state
        a = policy2.get_best_action(s0)
        env2.apply_action(a)
        s1 = env2.state
        print_state(s0, a)
        step_reward, finished = reward2(s0, a, s1)
        if not finished:
            trial_reward2 += step_reward
        logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)
        if finished:
            print_state(s1, -1)
            break
    return trial_reward, trial_reward2


def create_init(init_locations):
    ans = []
    for j in init_locations:
        ans.append(KitchenState(j[0], j[1], (), j[0], j[1]))

    return ans


def train(filename, seed):
    problem_mood = 3
    here = path.dirname(__file__)
    map_fn = path.join(here, "craft/kitchen.map")

    rng1 = Random(seed + 1)
    env1 = Kitchen(map_fn, rng1, 1, 1, problem_mood = problem_mood)
    init1 = create_init([[1,1]])

    rng4 = Random(seed + 4)
    env4 = Kitchen(map_fn, rng4, 1, 1, problem_mood = problem_mood)
    init4 = create_init([[1,1]])

    rng2 = Random(seed + 2)
    env2 = Kitchen( map_fn, rng2, 1, 1, problem_mood = problem_mood, baking=False)
    init2 = create_init([[1,1]])

    rng3 = Random(seed + 3)
    env3 = Kitchen(map_fn, rng3, 1, 1, problem_mood = problem_mood, baking=True)
    init3 = create_init([[1,1]])

    tasks = [[OBJECTS3["food"]]]
    not_task = []
    tasks = tasks[START_TASK:END_TASK+1]

    with open(filename, "w") as csvfile:

        print("ql: begin experiment")
        report2 = SequenceReport(csvfile, LOG_STEP, init2, EPISODE_LENGTH, TRIALS)
        report3 = SequenceReport(csvfile, LOG_STEP, init3, EPISODE_LENGTH, TRIALS)

        for j, goal in enumerate(tasks):
            print("ql: begin task {}".format(j + START_TASK))
            rng2.seed(seed + j)

            reward2 = ReachFacts(env2, goal, [])
            policy2 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.2,
                                    default_q=DEFAULT_Q, num_actions=4, rng=rng2)
            agent2 = Agent(env2, policy2, reward2, rng2)

            rng3.seed(seed + j)

            reward3 = ReachFacts(env3, goal, [])
            policy3 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.2,
                                    default_q=DEFAULT_Q, num_actions=5, rng=rng3)
            agent3 = Agent(env3, policy3, reward3, rng3)

            try:

                agent2.train(steps=TOTAL_STEPS1,
                             steps_per_episode=EPISODE_LENGTH, report=report2)
                agent3.train(steps=TOTAL_STEPS1,
                             steps_per_episode=EPISODE_LENGTH, report=report3)

                report1 = SequenceReport(csvfile, LOG_STEP, init1, EPISODE_LENGTH, TRIALS)
                reward1 = ReachFacts(env1, goal, not_task)

                policy1 = BaseEmpathic(alpha=1.0, gamma=1.0, epsilon=0.2,
                                       default_q=DEFAULT_Q, num_actions=5, rng=rng1, others_q=[policy3.get_Q()], others_init=[[1,1]], others_dist=[1.0], penalty=-2*EPISODE_LENGTH, others_alpha=[1.0], objects=OBJECTS3, problem_mood=problem_mood)
                agent1 = Agent(env1, policy1, reward1, rng1)

                agent1.train(steps=TOTAL_STEPS2,
                             steps_per_episode=EPISODE_LENGTH, report=report1)
                report4 = SequenceReport(csvfile, LOG_STEP, init4, EPISODE_LENGTH, TRIALS)
                reward4 = ReachFacts(env4, goal, not_task)
                policy4 = BaseEmpathic(alpha=1.0, gamma=1.0, epsilon=0.2,
                                       default_q=DEFAULT_Q, num_actions=5, rng=rng4, others_q=[policy3.get_Q()], others_init=[[1,1]], others_dist=[1.0], penalty=-2*EPISODE_LENGTH, others_alpha=[1.0], objects=OBJECTS3, problem_mood=problem_mood)
                agent4 = Agent(env4, policy4, reward4, rng4)

                agent4.train(steps=TOTAL_STEPS2,
                             steps_per_episode=EPISODE_LENGTH, report=report4)


                print("-----------------")
                print("The first agent in the env without considering others")
                base_step = evaluate_agent(env2, policy2, reward2, init2)
                print("Total Reward = ", base_step)
                print("-----------------")
                print("The second agent in the env without considering others")
                base_step2 = evaluate_agent(env3, policy3, reward3, init3)
                print("Total Reward = ", base_step2)
                print("-----------------")
                print("Baseline 1, Q-learning")
                step2 = evaluate_second_agent(env2, policy2, reward2, env3, policy3, reward3)
                print("Total Reward Compared to Base = ", -1 * (step2[0] - base_step), -1 * (step2[1] - base_step2 ))
                print("-----------------")
                print("Baseline 2, Krakovna et al")
                step1 = evaluate_second_agent(env1, policy1, reward1, env3, policy3, reward3)
                print("Total Reward Compared to Base= ", -1 * (step1[0] - base_step), -1 * (step1[1] - base_step2 ))
                print("-----------------")
                print("Our Approach")
                step3 = evaluate_second_agent(env4, policy4, reward4, env3, policy3, reward3)
                print("Total Reward Compared to Base= ", -1 * (step3[0] - base_step), -1 * (step3[1] - base_step2))
                print("-----------------")

            except KeyboardInterrupt:

                logging.warning("ql: interrupted task %s after %s seconds!",
                                j + START_TASK, end - start)


start = time()
train("./test.csv", 2019)
end = time()
print("Total Time:", end - start)

