import numpy as np
from collections import deque
from sklearn import metrics
import random
import tensorflow as tf


class DeepQNet:
    """ Sklearn style deep-q estimator.

    Parameters:
    model_fn: ...
    layers: Number of layers and nodes per layer as list. (example: layers=[100, 50])
    input_size: Number of input nodes. (layers as well as input_size maybe replaced by model_fn)
    n_actions: Number of possible actions. (Similar to n_classes)
    batch_size: Mini batch size.
    episodes: Number of episodes of training.
    max_steps: Maximum number of stes in a episode befor termination.
    optimizer: Tensorflow training optimizer. (tf.train.AdamOptimizer,
    tf.train.GradientDescentOptimizer)
    learning_rate: Learning rate.
    gamma: How much are future rewards taken into account.
    epsilon: Weight between random and predicted actions. (A value of one
    produces complete random values)
    min_epsilon: Minimum epsilon befor decay stops.
    epsilon_decay: Specifies how much the epsilon value decays over time.
    min_observation: Minimum observations befor training starts. (Must be
    greater than batch_size or else
    it will be set to batch_size)
    max_observations: Maximum observations stored befor the earlier observations get
    removed. (Must be greater than batch_size or else it will be set to batch_size)
    training_dropout: The dropout use while training 1.0 means no dropout. (to turn it on eg. 0.5)
    continue_training: When true training will be continued on every call of fit.
    verbose: Controls the the output while training:
        0: Silent no output
        1: Display output
    """

    def __init__(self,
                 # model_fn,
                 layers,
                 input_size,
                 n_actions,
                 batch_size=30,
                 episodes=100000,
                 max_steps=10000,
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=1e-3,
                 gamma=0.9,
                 epsilon=1,
                 min_epsilon=0.05,
                 epsilon_decay=0.01,
                 min_observations=500,
                 max_observations=10000,
                 training_dropout=1.0,
                 verbose=1,
                 continue_training=True):
        # self.model_fn = model_fn
        self.layers = layers
        self.input_size = input_size
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.episodes = episodes
        self.max_steps = max_steps
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_observations = min_observations
        self.max_observations = max_observations
        self.training_dropout = training_dropout
        self.verbose = verbose
        self.continue_training = continue_training

        if batch_size > min_observations:
            min_observations = batch_size
        if batch_size > max_observations:
            max_observations = batch_size

        self.sess = None
        self.initialize_new_session()

    def initialize_new_session(self):
        if self.sess is not None:
            self.sess.close()
        self.sess = tf.Session()
        self.create_fc_model(self.input_size, self.layers)
        self.create_train()

        self.sess.run(tf.global_variables_initializer())

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

    def create_fc_layer(self, in_layer, shape, layer_name, activation=tf.nn.relu, keep_prob=0.0):
        """Creates a fully connected layer.
        Args:
            x: The previous/input layer.
            in_size: Number of nodes in the input layer.
            units: Number of nodes the layer should contain.
            layer_name: Name for the layer. (should be unique)
            activation(optional): A Tensorflow activation function. (example: tf.nn.relu)

        Returns:
            The fully connected layer as TensorFlow object.
        """
        fc_layer = None
        with tf.variable_scope(layer_name):
            W = tf.get_variable("weights", shape,
                                initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable("biases", [shape[1]],
                                initializer=tf.constant_initializer(0.1))
            fc_layer = tf.matmul(in_layer, W) + b
            if activation is not None:
                fc_layer = activation(fc_layer)
        return fc_layer

    def create_fc_model(self, input_size, layers):
        """Creates a fully connected inference model.

            Note: The layers argument should take a form like:
            "layers=[100, 50]" this will result in a network with 100 nodes in the
            first layer and 50 in the second.

        Args:
            input_size: Number of input nodes/feature size.
            layers: Number of layers and number of nodes per layer as python list.
        """

        self.x = tf.placeholder('float', shape=[None, input_size])  # <-- change!
        self.y = tf.placeholder('float', shape=[None])
        self.a = tf.placeholder('float', shape=[None, self.n_actions])
        self.lr = tf.placeholder('float')
        self.keep_prob = tf.placeholder('float')

        last_layer = self.x
        last_layer_size = input_size

        for layer_index, layer in enumerate(layers):
            last_layer = self.create_fc_layer(last_layer, [last_layer_size, layer], 'layer' + str(layer_index))
            layer_layer = tf.nn.dropout(last_layer, self.keep_prob)
            last_layer_size = layer

        self.prediction = self.create_fc_layer(last_layer, [last_layer_size, self.n_actions], 'readout_layer',
                                               activation=None)

    def create_train(self):
        with tf.name_scope('training'):
            action = tf.reduce_sum(tf.multiply(self.prediction, self.a), reduction_indices=1)
            self.loss = tf.reduce_mean(tf.square(self.y - action))
            self.train = self.optimizer(self.lr).minimize(self.loss)

    def predict(self, state):
        """Predicts the q values of all the actions for a given state.

        Args:
            state: A state of the enviroment.

        Returns:
            A array containing all the q values of the actions.

        """
        return self.prediction.eval(session=self.sess, feed_dict={self.x: state, self.keep_prob: 1.0})

    def fit(self, env, render=False):
        """Trains the the model with a given model.

        Note: For now this should be a openai gym enviroment, passed as
        argument. Later on this maybe should be a wrapper/interface and
        also n_actions and such arguments in __init__ could be removed
        and the envirment passt in __init__.

        Args:
            interact_fn: A function that takes a action as argument and returns
            the resulting state, a reward and terminal(boolean)

        """
        observations = deque()

        if self.continue_training is False:
            self.sess.run(tf.global_variables_initializer())

        for episode in range(self.episodes):
            self.episode = episode
            state = env.reset()
            reward = 0
            terminal = False
            step = 0
            for step in range(self.max_steps):
                action = [0] * self.n_actions
                if render:
                    env.render()
                # with probability of epsilon select a random action
                if self.epsilon < random.random() and len(observations) > self.min_observations:
                    action_pred = self.predict([state])
                    action[np.argmax(action_pred)] = 1
                else:  # otherwise select the action with the highest Q value
                    rand = random.randint(0, self.n_actions - 1)
                    action[rand] = 1
                # Execute the action and get a observation, reward and terminal.
                old_state = state  # set the old state
                action_argmax = np.argmax(action)
                state, reward, terminal, info = env.step(action_argmax)
                # Store the transition
                observations.append((old_state, action, reward, state, terminal))

                if len(observations) > self.max_observations:
                    observations.popleft()

                # training
                if len(observations) >= self.min_observations:
                    batch = random.sample(observations, self.batch_size)
                    self.train_batch(batch)

                if terminal:
                    break

            if self.epsilon >= self.min_epsilon and len(observations) > self.min_observations:
                self.epsilon -= self.epsilon_decay
            if self.verbose > 0:
                print('episode', episode, 'ended at step', step, ' epsilon', self.epsilon)

    def train_batch(self, batch, verbose=False):
        """ Trains the model with a the given batch.

        Note: The batch must be a list ob tuples with the following elements:
            0 ... original state
            1 ... actions
            2 ... rewards
            3 ... new state
            4 ... is terminal state

        Args:
            :param batch: Training batch
            :param verbose: If set to true there will be verbose output.
        """

        state = [d[0] for d in batch]
        action = [d[1] for d in batch]
        reward = [d[2] for d in batch]
        new_state = [d[3] for d in batch]
        terminal = [d[4] for d in batch]
        predicted = self.predict(new_state)

        q_value = []

        for idx in range(0, len(state)):
            if terminal[idx]:  # is terminal state
                q_value.append(reward[idx] - 10)  # reward only
            else:
                # calculate q value
                q_value.append(reward[idx] + self.gamma * np.max(predicted[idx]))

            if verbose:
                print(state[idx])
                print(action[idx])
                print(reward[idx])
                print(predicted[idx])
                print(q_value[idx])

        lrate = self.learning_rate
        if type(self.learning_rate) is not float:
            lrate = self.learning_rate(self.episode)
        # perform a gradient descent step
        self.train.run(session=self.sess, feed_dict={self.x: state, self.y: q_value, self.a: action, self.lr: lrate,
                                                     self.keep_prob: self.training_dropout})
