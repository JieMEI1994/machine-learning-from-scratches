import numpy as np
import pandas as pd

class MontaCarloAgent:
    def __init__(self, actions,
                 learning_rate = 0.01, 
                 discount_rate = 0.9, 
                 greedy_rate = 0.9):
        # actions is a set of action
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount_rate
        self.epsilon = greedy_rate
        self.q_table = pd.DataFrame(columns=self.actions)
        
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series(
                [0]*len(self.actions),
                index=self.q_table.columns,
                name=state,))

    def pick_action(self, observation):
        self.check_state_exist(observation)
        # e-greedy selection
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.ix[observation, :]
            # some actions have same value
            # randomly permute(置换) a sequence, or return a permuted range
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, next_state):
        self.check_state_exist(next_state)
        q_predict = self.q_table.ix[state, action]
        if next_state != 'terminal':
            q_target = reward + self.gamma * self.q_table.ix[next_state, :].max()
        else:
            q_target = reward
        error = q_target - q_predict
        self.q_table.ix[state, action] += self.alpha * error