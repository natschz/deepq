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

        return self.state, r * 10, terminal, info

    def render(self):
        self.env.render()


a_learning_rate = 1e-3


def learning_rate(episode):
    global a_learning_rate
    if episode > 4000 and a_learning_rate > 1e-5:
        a_learning_rate -= 1e-6
    return a_learning_rate


history_length = 5

learning_rate = 1e-3

net = deepqnet.DeepQNet([500], 4 * history_length, 2, batch_size=50, gamma=0.9, learning_rate=learning_rate,
                        min_observations=500, max_observations=1500, epsilon_decay=0.005, min_epsilon=0.005,
                        max_steps=1000, training_dropout=1)
env = gym.make('CartPole-v0')
env = History(env, history_length, 4)

# env = sgw.SGW(5, 100)
net.fit(env, render=False)
