import classifiers.deepqnet as dqn
import enviroments.simplegridworld as sgw

net = dqn.DeepQNet([50], 100, 5, batch_size=1, min_epsilon=0.001)

env = sgw.SGW(5, 100)
net.fit(env, render=True)
