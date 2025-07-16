#!/usr/bin/env python3
import numpy as np
import random
import pickle


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
        best_action = np.flatnonzero(Q[observation] == np.max(Q[observation]))
        for action in best_action:
            A[action] += (1.0 - epsilon)/len(best_action)
        return A
    return policy_fn


class QAgent:
    def __init__(self, nA = 3, discount_factor=1.0, alpha=0.5, epsilon=0.1, episode = 25, model_path = ""):
        if model_path == "":
            self.Q = self.reset_Q(3,4)
            self.policy = make_epsilon_greedy_policy(self.Q, epsilon, nA)
            self.prev_episodes = 0
            self.discount_factor = discount_factor
        else:
            self.load_model(model_path)
            self.policy = make_epsilon_greedy_policy(self.Q, self.discount_factor, nA)

        self.grid = np.array([
            [0, 1, -1, -1],  
            [0, -1, 1, -1],
            [0, -1, -1, 1]
        ])
        self.alpha = alpha
        self.episodes = episode

        start_state = random.randint(0, 2)
        self.start_state = (start_state, 0)
        self.state = self.start_state
        self.action = None
      
    def load_model(self, path):
      with open(path, 'rb') as f:
        data = pickle.load(f)
      self.Q = data["q_table"]
      self.prev_episodes = data["episode"]
      self.discount_factor = data["epsilon"]
      print(self.Q)
    
    def save_model(self, path):
      checkpoint = {
        'episode': 25,
        'epsilon': self.discount_factor,
        'q_table': self.Q  # Or model state_dict if using neural networks
      }
      with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


    def reset_Q(self, n, m):
        Q = defaultdict(list)
        for i in range(n):
            for j in range(m):
                Q[(i, j)] = np.zeros(3) 
        return Q
            

    def reset(self):
        start_state = random.randint(0, 2)
        self.start_state = (start_state, 0)
        self.state = self.start_state
        return self.start_state

    def is_terminal(self, state):
        return self.grid[state] == 1 or self.grid[state] == -1

    def tagid_to_state(self, tagid, state = (0,0)):
        next_state = list(state)
        if tagid == 0:  # Move forward
            next_state[1] = min(3, state[1] + 1)
        elif tagid == 1:  # Move right
            next_state[1] = min(3, state[1] + 2)
        elif tagid == 2:  # Move left
            next_state[1] = min(3, state[1] + 3)
        else:
            next_state = [0, 0]
        return tuple(next_state)

    def step(self, tagid):
        next_state = self.tagid_to_state(tagid, self.state)
        reward = self.grid[next_state]
        self.state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done
    
    def select_action(self):
        action_probs = self.policy(self.state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action       
    
    def update(self, action, tagid):
        state = self.state
        next_state, reward, done = self.step(tagid)
        best_next_action = np.argmax(self.Q[next_state])    
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        td_delta = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_delta
        return reward


if __name__ == "__main__":
    tagid = 3 # this should based on what is reported from the environment
    action = None

    # Sample usage
    # Invoke
    Duckiebot = QAgent()

    ###### When detect the intersection tagid
    for i in range(30):
        state = Duckiebot.tagid_to_state(2000)

        Duckiebot.reset()
        # print(f"starting state: {Duckiebot.start_state}")
        action = Duckiebot.select_action()
        tagid = random.randint(0, 2)
        state = Duckiebot.tagid_to_state(tagid)

        reward = Duckiebot.update(action, tagid)
        # print(f"action: {action}")


    # Save the checkpoint using pickle or other serialization method
    # Duckiebot.save_model('checkpoint.pkl')
    print(Duckiebot.Q)
