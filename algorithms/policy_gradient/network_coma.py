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
        sample_p = logits - tf.math.log(-tf.math.log(u))
        return tf.argmax(sample_p, axis=-1), sample_p



class normc_initializer(tf.keras.initializers.Initializer):
    def __init__(self, std=1.0, axis=0):
        self.std = std
        self.axis = axis
    def __call__(self, shape, dtype=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= self.std / np.sqrt(np.square(out).sum(axis=self.axis, keepdims=True))
        return tf.constant(out)


class Network(tf.keras.Model):

    def __init__(self, batch_size=1, n_agents=2):
        super(Network, self).__init__()
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.coptimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        self.poptimizer = tf.keras.optimizers.Adam(learning_rate=2e-3, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        self.pd = CategoricalPd()
        with tf.name_scope("policy"):
            self.mlp1_p = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_p = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_p = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.p = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(0.01))
        with tf.name_scope("qvalue"):
            # Normal q-value
            self.mlp1_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.q = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(1.0))
            # Target q-value
            self.mlp1_tq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp2_tq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp3_tq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.tq = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(1.0), trainable=False)

    def get_neglogp(self, logits, action):
        return self.pd.neglogp(logits, action)

    def polyak_qnet(self):
        polyak = 0.995
        self.mlp1_tq.set_weights([polyak * tq1 + (1 - polyak) * q1 for tq1, q1 in zip(self.mlp1_tq.get_weights(), self.mlp1_q.get_weights())])
        self.mlp2_tq.set_weights([polyak * tq2 + (1 - polyak) * q2 for tq2, q2 in zip(self.mlp2_tq.get_weights(), self.mlp2_q.get_weights())])
        self.mlp3_tq.set_weights([polyak * tq3 + (1 - polyak) * q3 for tq3, q3 in zip(self.mlp3_tq.get_weights(), self.mlp3_q.get_weights())])
        self.tq.set_weights([polyak * tq + (1 - polyak) * q for tq, q in zip(self.tq.get_weights(), self.q.get_weights())])

    def get_q_value(self, obs, actions):
        all_actions = []
        for ai in range(self.n_agents - 1):
            action_1H = tf.one_hot(actions[:, ai], 5, dtype=tf.float32)
            all_actions.append(action_1H)
        all_actions = tf.concat(all_actions, axis=-1)
        q_input = tf.concat([obs, all_actions], axis=-1)    
        q = self.mlp1_q(q_input)
        q = self.mlp2_q(q)
        q = self.mlp3_q(q)
        q = self.q(q)
        return q
    
    def get_tq_value(self, obs, actions):
        all_actions = []
        for ai in range(self.n_agents - 1):
            action_1H = tf.one_hot(actions[:, ai], 5, dtype=tf.float32)
            all_actions.append(action_1H)
        all_actions = tf.concat(all_actions, axis=-1)
        tq_input = tf.concat([obs, all_actions], axis=-1)    
        tq = self.mlp1_tq(tq_input)
        tq = self.mlp2_tq(tq)
        tq = self.mlp3_tq(tq)
        tq = self.tq(tq)
        return tq        

    @tf.function
    def call(self, obs, taken_action=None):
        with tf.name_scope("policy"):
            p = self.mlp1_p(obs)
            p = self.mlp2_p(p)
            p = self.mlp3_p(p)
            p = self.p(p)
            action, sample_p = self.pd.sample(p)
            neglogp = self.get_neglogp(p, action)
            probs = self.pd.mean(sample_p)#[tf.exp(-self.get_neglogp(p, np.array([i for _ in range(p.shape[0])]))) for i in range(5)]
            #probs = tf.transpose(probs, [1, 0])
            entropy = self.pd.entropy(p)
        if taken_action != None:
            taken_action = taken_action[0]
            taken_action_neglogp = self.get_neglogp(p, taken_action)
            return action, neglogp, entropy, probs, p, taken_action_neglogp
        return action, neglogp, entropy, probs, p
