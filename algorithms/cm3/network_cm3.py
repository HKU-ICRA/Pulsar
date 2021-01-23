import os
import sys
import numpy as np
import tensorflow as tf


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
        p = logits - tf.math.log(-tf.math.log(u))
        return tf.argmax(p, axis=-1), p



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
        with tf.name_scope("policy_s1"):
            n_h1_actor = 64
            n_h2_actor = 64
            self.mlp1_actor_s1 = tf.keras.layers.Dense(units=n_h1_actor, activation=tf.nn.relu, kernel_initializer=normc_initializer(0.01))
            self.W_branch1_h2_actor = tf.Variable(initial_value=tf.keras.initializers.TruncatedNormal(0, 0.01)([n_h1_actor, n_h2_actor]))
        with tf.name_scope("policy_s2"):
            n_h1_others = 64
            self.mlp1_actor_s2 = tf.keras.layers.Dense(units=n_h1_others, activation=tf.nn.relu, kernel_initializer=normc_initializer(0.01))
            self.W_branch1_h2_others = tf.Variable(initial_value=tf.keras.initializers.TruncatedNormal(0, 0.01)([n_h1_others, n_h2_actor]))
            self.B_actor = tf.Variable(initial_value=tf.keras.initializers.GlorotUniform()([n_h2_actor]))
            self.out_p = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(0.01))
        with tf.name_scope("global_critic_s1"):
            n_h1_1_gq = 64
            n_h2_gq = 64
            self.mlp1_gq_s1 = tf.keras.layers.Dense(units=n_h1_1_gq, activation=tf.nn.relu, kernel_initializer=normc_initializer(0.01))
            self.W_branch1_h2_gq = tf.Variable(initial_value=tf.keras.initializers.TruncatedNormal(0, 0.01)([n_h1_1_gq, n_h2_gq]))
            self.out_gq = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(0.01), use_bias=False)
        with tf.name_scope("credit_critic_s2"):
            n_h1_1_cq = 64
            n_h1_2_cq = 128
            n_h2_cq = 64
            self.mlp1_cq_s1 = tf.keras.layers.Dense(units=n_h1_1_gq, activation=tf.nn.relu, kernel_initializer=normc_initializer(0.01))
            self.W_branch1_h2_cq = tf.Variable(initial_value=tf.keras.initializers.TruncatedNormal(0, 0.01)([n_h1_1_gq, n_h2_gq]))
            self.mlp1_cq_s2 = tf.keras.layers.Dense(units=n_h1_2_cq, activation=tf.nn.relu, kernel_initializer=normc_initializer(0.01))
            self.W_branch2_h2_cq = tf.Variable(initial_value=tf.keras.initializers.TruncatedNormal(0, 0.01)([n_h1_2_cq, n_h2_cq]))
            self.out_cq = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(0.01), use_bias=False)

    def get_neglogp(self, logits, action):
        return self.pd.neglogp(logits, action)

    def reset_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5)

    def initialize_credit_weights(self):
        self.mlp1_cq_s1.set_weights(self.mlp1_gq_s1.get_weights())
        self.W_branch1_h2_cq.assign(self.W_branch1_h2_gq.read_value())
        self.out_cq.set_weights(self.out_gq.get_weights())

    @tf.function
    def get_global_critic(self, obs):
        branch1 = self.mlp1_gq_s1(obs)
        list_mult = []
        list_mult.append( tf.matmul(branch1, self.W_branch1_h2_gq) )
        h2 = tf.nn.relu(tf.add_n(list_mult))
        out = self.out_gq(h2)
        return out

    @tf.function
    def get_credit_critic(self, obs, other_obs=None, other_action=None, stage=2):
        branch1 = self.mlp1_cq_s1(obs)
        list_mult = []
        list_mult.append( tf.matmul(branch1, self.W_branch1_h2_cq) )
        if stage > 1:
            other_action_1H = tf.one_hot(other_action, 5, dtype=tf.float32)
            other_inputs = tf.concat([other_obs, other_action_1H], axis=-1)
            others = self.mlp1_cq_s2(other_inputs)
            list_mult.append(tf.matmul(others, self.W_branch2_h2_cq))
        h2 = tf.nn.relu(tf.add_n(list_mult))
        out = self.out_cq(h2)
        return out

    @tf.function
    def call(self, obs, other_obs=None, stage=1, taken_action=None):
        branch_self  = self.mlp1_actor_s1(obs)
        list_mult = []
        list_mult.append( tf.matmul(branch_self, self.W_branch1_h2_actor) )
        if stage > 1:
            others = self.mlp1_actor_s2(other_obs)
            list_mult.append(tf.matmul(others, self.W_branch1_h2_others))
        h2 = tf.nn.relu(tf.nn.bias_add(tf.add_n(list_mult), self.B_actor))
        p = self.out_p(h2)
        action, probs = self.pd.sample(p)
        probs = self.pd.mean(probs)
        neglogp = self.get_neglogp(p, action)
        entropy = self.pd.entropy(p)            
        if taken_action != None:
            taken_action = taken_action[0]
            taken_action_neglogp = self.get_neglogp(p, taken_action)
            return action, neglogp, entropy, probs, p, taken_action_neglogp
        return action, neglogp, entropy, probs, p
