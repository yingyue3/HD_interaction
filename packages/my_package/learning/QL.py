import numpy as np
import random

# Define the gridworld environment
class DuckietownEnv:
    def __init__(self, nS, nA, P):
        self.grid = np.array([
            [0, 0, 0],  # Wall with reward -1
            [0, 0, 0],
            [0, 0, 0]  # Start at (3, 0)
        ])
        self.start_state = (2, 1)
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self.state

    def is_terminal(self, state):
        return self.grid[state] == 1 or self.grid[state] == -1

    def get_next_state(self, state, action):
        next_state = list(state)
        if action == 0:  # Move forward
            next_state[0] = max(0, state[0] - 1)
        elif action == 1:  # Move right
            next_state[1] = min(3, state[1] + 1)
        elif action == 2:  # Move left
            next_state[0] = min(3, state[1] - 1)
        return tuple(next_state)

    def step(self, action):
        next_state = self.get_next_state(self.state, action)
        reward = self.grid[next_state]
        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done