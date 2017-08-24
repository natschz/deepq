import deepqnet
import gym
from sklearn import metrics
import numpy as np


class History():
    def __init__(self, env, history_length, input_length):
        self.env = env
        self.history_length = history_length
        self.input_length = input_length
        self.reset()
        self.reward = 0

    def reset(self):
        s = self.env.reset()
        state = [s] * (self.history_length - 1)
        self.state = np.append(s, state)
        self.reward = 0
        return self.state

    def step(self, action):
        s, r, terminal, info = self.env.step(action)
        self.state = np.append(self.state[self.input_length:], s)
        self.reward += r

        return self.state, r, terminal, info

    def render(self):
        self.env.render()


history_length = 5
input_length = 2

net = deepqnet.DeepQNet([250], input_length * history_length, 2, batch_size=50, gamma=0.9, learning_rate=1e-3,
                        min_observations=500, max_observations=15000, epsilon_decay=0.01, min_epsilon=0.05,
                        epsilon=0.05, max_steps=1500)
env = gym.make('MountainCar-v0')
env = History(env, history_length, input_length)

# env = sgw.SGW(5, 100)
net.fit(env, render=True)
