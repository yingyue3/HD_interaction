#!/usr/bin/env python3

import itertools
import matplotlib
import numpy as np
import sys
import numpy as np
import random



from collections import defaultdict


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


class QAgent:
    def __init__(self, discount_factor=1.0, alpha=0.5, epsilon=0.1, load_model = None):
        self.Q = defaultdict(lambda: np.zeros(3))
        self.policy = make_epsilon_greedy_policy(self.Q, epsilon, 3)
        self.grid = np.array([
            [0, 1, -1, -1],  
            [0, -1, 1, -1],
            [0, -1, -1, 1]
        ])
        start_state = random.randint(0, 2)
        self.start_state = (start_state, 0)
        self.state = self.start_state
        self.discount_factor = discount_factor
        self.alpha = alpha
    
    def reset(self):
        start_state = random.randint(0, 2)
        self.start_state = (start_state, 0)
        self.state = self.start_state
        return

    def is_terminal(self, state):
        return self.grid[state] == 1 or self.grid[state] == -1

    def get_next_state(self, state, tagid):
        next_state = list(state)
        if tagid == 0:  # Move forward
            next_state[1] = max(3, state[1] + 1)
        elif tagid == 1:  # Move right
            next_state[1] = min(3, state[1] + 2)
        elif tagid == 2:  # Move left
            next_state[1] = min(3, state[1] + 3)
        return tuple(next_state)

    def step(self, tagid):
        next_state = self.get_next_state(self.state, tagid)
        reward = self.grid[next_state]
        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done
    
    def episode(self, tagid):
        state = self.reset()
        action_probs = self.policy(state)
        print(action_probs)
        print(self.start_state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done = self.step(tagid)

        best_next_action = np.argmax(self.Q[next_state])    
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta

if __name__ == "__main__":
    Duckiebot = QAgent()
    for i in range(5):
        tagid = random.randint(0,2)
        Duckiebot.episode(tagid)
    print(Duckiebot.Q[0][0])
    pass
