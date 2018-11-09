import numpy as np

# Copied from src/causality/exercises.py
class BasicPolicy:
    def __init__(self, pi):
        self.pi = pi
    def get_action(self):
        return np.random.choice(2, 1, p=[1-self.pi, self.pi])
    def get_n_actions(self):
        return 2

class StandardModel:
    ## Now the mean is a vector
    def __init__(self, mean):
        self.mean = mean
        #print("my mean: ", self.mean)
    def get_response(self, action):
        #print("action: ", action, self.mean[action])
        return np.random.normal(self.mean[action], 1)
    def Evaluate(self, policy, n_samples):
        hat_U = 0
        for t in range(n_samples):
            a = policy.get_action()
            y = self.get_response(a)
            hat_U += y
        return hat_U/n_samples
