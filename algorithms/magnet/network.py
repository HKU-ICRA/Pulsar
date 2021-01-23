import os
import sys
import numpy as np
import tensorflow as tf

from graph_construct import GraphConstruction
from action_execution import ActionExecution


class CategoricalPd:
    """
        Args:
            logits: a tensor of logits outputted from a neural network
            x: the sampled argmax action index
    """
    def mode(self, logits):
        return tf.argmax(logits, axis=-1)

    def mean(self, logits):
        return tf.nn.softmax(logits)

    def neglogp(self, logits, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        x = tf.one_hot(x, logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=x)

    def kl(self, logits, other_logits):
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        a1 = other_logits - tf.reduce_max(other_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=-1)

    def entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)

    def sample(self, logits):
        u = tf.random.uniform(tf.shape(logits), dtype=logits.dtype)
        return tf.argmax(logits - tf.math.log(-tf.math.log(u)), axis=-1)



class normc_initializer(tf.keras.initializers.Initializer):
    def __init__(self, std=1.0, axis=0):
        self.std = std
        self.axis = axis
    def __call__(self, shape, dtype=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= self.std / np.sqrt(np.square(out).sum(axis=self.axis, keepdims=True))
        return tf.constant(out)


class Network(tf.keras.Model):

    def __init__(self, batch_size=1):
        super(Network, self).__init__()
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        self.pd = CategoricalPd()
        with tf.name_scope("magnet"):
            self.nn1 = GraphConstruction()
            self.nn2 = ActionExecution()
        with tf.name_scope("policy"):
            self.mlp1_p = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_p = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_p = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.p = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(0.01))
        with tf.name_scope("qvalue"):
            self.mlp1_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.q = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(0.1))

    def input_to_ppo(self, prev_state, curr_state, graph):
        graph = self.nn1(state_1=prev_state, state_2=curr_state, labels=np.asmatrix(graph.flatten()))
        graph = np.asmatrix(graph)
        input_to_ppo = np.concatenate([curr_state, graph], axis=1)
        print(curr_state.shape, graph.shape)
        return input_to_ppo

    def get_initial_states(self):
        """
            Returns:
                prev_state, curr_state, initial graph
        """
        return None, None, np.random.rand(4, 120).astype("float32") + 0.0001

    def state_to_matrix_with_action(self, obs, action):

    def act(self, obs, prev_state, curr_state, graph, pr_action):
        if pr_action is not None:
            curr_state = self.state_to_matrix_with_action(obs, action=pr_action)
        if prev_state is not None:
            input_curr_state = curr_state
            input_prev_state = prev_state
            input_to_ppo = self.input_to_ppo(input_prev_state, input_curr_state)
            curr_state_matrix = curr_state
            prev_state_matrix = prev_state
            input_to_ppo = self.input_to_ppo(prev_state_matrix, curr_state_matrix, graph)
        return action
    
    def pretrain_transformer(self, batch_size=5000, epochs=100, early_stopping=20,
                             save_best_only=True, random_state=392, test_size=0.2, shuffle=True):
        state_merged = np.load(train_data_state)
        prev_state_merged = np.load(train_data_state)
        labels_merged = np.load(train_data_labels)
        rewards_merged = np.load(train_data_reward)
        for _ in range(100):
            state_merged = np.random.shuffle(state_merged)
            prev_state_merged = np.random.shuffle(prev_state_merged)
            labels_merged = np.random.shuffle(labels_merged)
            self.nn1.train(state_merged, prev_state_merged, labels_merged)

    def get_neglogp(self, logits, action):
        return self.pd.neglogp(logits, action)

    @tf.function
    def call(self, obs, taken_action=None):
        with tf.name_scope("policy"):
            p = self.mlp1_p(obs)
            p = self.mlp2_p(p)
            p = self.mlp3_p(p)
            p = self.p(p)
            probs = self.pd.mean(p)
            action = self.pd.sample(p)
            neglogp = self.get_neglogp(p, action)
            entropy = self.pd.entropy(p)
        with tf.name_scope("qvalue"):
            q = self.mlp1_q(obs)
            q = self.mlp2_q(q)
            q = self.mlp3_q(q)
            q = self.q(q)[:, 0]
        if taken_action != None:
            taken_action = taken_action[0]
            taken_action_neglogp = self.get_neglogp(p, taken_action)
            return action, neglogp, entropy, q, probs, p, taken_action_neglogp
        return action, neglogp, entropy, q, probs, p
