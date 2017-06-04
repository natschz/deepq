# Deep Q
Because a few people requested to see the deep q network i created which solves a simple grid world, I finaly decided to make it public.

So help improve and please don't just copy paste for any assignments!

## General

I created this two years ago and didn't run it since then and also rearranged the folder structure, and yea it's pretty sure broken to a certain point (please someone fix it! ;) ). 
If you run it and fix some stuff I strongly encourage you to push your changes, so that other people like you can profit from it! Also general improvements are very welcome and feel free to add some enviroments and samples that would be really nice, for example for cartpole or mountain hill (get them to work - I didn't)!

I'll try to explain the most important stuff below as good as I can remember. Also feel free to help improve!

## Enviroments

There are enviroments, which are used to train the deep q net. A enviroment for example could for example be a grid world or cartpole. Essentially it defines how the world work, the rules. 

Each of this enviroments must define two functions:

* **reset(self)** Which is used to reset the enviroment to it's default state.
* **step(self, action)** Performs a action in the enviroment. **action** is a integer which is given by the deep q classifier (the action the deep q network choose).
* **render(self)** This function should render the progess in the enviroment, for example print the current state in the console.

## Deep Q Classifier

The heart of this all is the deep q classifier, it is a scikit style classifier and mostly documented.

Here is a short example for further examples see the samples.

```python
import deepqnet
import simplegridworld as sgw

net = deepqnet.DeepQNet([50], 100, 5, batch_size=1, min_epsilon=0.001) # initialze the deep q classifier

env = sgw.SGW(5, 100) # initialize the gridworld
# trains the netowrk
# render=True would print out the enviroments state by calling simplegridworld's render function
net.fit(env, render=False)
```