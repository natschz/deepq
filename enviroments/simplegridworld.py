import numpy as np
from sklearn import metrics
import os


class SGW:
    def __init__(self, grid_size=5, max_steps=250):
        self.GRIDSIZE = grid_size
        self.MAXSTEPS = max_steps

    def reset(self):
        self.reward = 0
        self.step_count = 0
        self.state = np.array([0.0] * (self.GRIDSIZE * self.GRIDSIZE), float).reshape(self.GRIDSIZE * self.GRIDSIZE)
        self.state[0] = 1

        rsh_state = self.state  # self.state.reshape(self.GRIDSIZE, self.GRIDSIZE)
        self.s = np.append(rsh_state, (rsh_state, rsh_state, rsh_state))

        return self.s

    def render(self):
        print("reward", self.reward)
        print("step_count", self.step_count)

    def step(self, action):
        pos_idx = self.state.argmax(axis=0)
        pos = pos_idx + 1
        game_ended = False

        self.state[pos_idx] = 0
        if action == 0 and pos % self.GRIDSIZE != 1:  # left
            self.state[pos_idx - 1] = 1
        elif action == 1 and pos > self.GRIDSIZE:  # up
            self.state[pos_idx - self.GRIDSIZE] = 1
        elif action == 2 and pos % self.GRIDSIZE != 0:  # right
            self.state[pos_idx + 1] = 1
        elif action == 3 and pos < (self.GRIDSIZE * self.GRIDSIZE - self.GRIDSIZE + 1):  # down
            self.state[pos_idx + self.GRIDSIZE] = 1
        else:  # noop
            self.state[pos_idx] = 1
            pass

        self.step_count += 1
        if self.step_count >= self.MAXSTEPS:
            self.reward = -100
            game_ended = True
        elif self.state[self.GRIDSIZE * self.GRIDSIZE - 1] == 1:  # reward & finished
            self.reward = 100
            game_ended = True
        else:
            self.reward = -1

        self.s = np.append(self.s[self.GRIDSIZE * self.GRIDSIZE:], self.state)

        return self.s, self.reward, game_ended, ''
