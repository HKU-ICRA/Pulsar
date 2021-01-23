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
        return tf.argmax(logits - tf.math.log(-tf.math.log(u)), axis=-1)



class normc_initializer(tf.keras.initializers.Initializer):
    def __init__(self, std=1.0, axis=0):
        self.std = std
        self.axis = axis
    def __call__(self, shape, dtype=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= self.std / np.sqrt(np.square(out).sum(axis=self.axis, keepdims=True))
        return tf.constant(out)


class QMixer(tf.keras.layers.Layer):

    def __init__(self, n_agents, trainable=True):
        super(QMixer, self).__init__(trainable=trainable)
        self.n_agents = n_agents
        self.n_embed = n_embed = 64
        self.hyper1w_1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.hyper1w_2 = tf.keras.layers.Dense(units=n_embed * self.n_agents)
        self.hyper2w_1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.hyper2w_2 = tf.keras.layers.Dense(units=n_embed)
        self.hyper1b = tf.keras.layers.Dense(units=n_embed)  # bias for hidden layer
        self.v_1 = tf.keras.layers.Dense(units=n_embed, activation=tf.nn.relu) # V(s) instead of bias for last layers
        self.v_2 = tf.keras.layers.Dense(units=1)
    
    def call(self, agent_qs, obs):
        bs = obs.shape[0]
        agent_qs = tf.reshape(agent_qs, [bs, 1, self.n_agents])
        # First layer
        w1 = self.hyper1w_1(obs)
        w1 = self.hyper1w_2(w1)
        w1 = tf.abs(w1)
        b1 = self.hyper1b(obs)
        w1 = tf.reshape(w1, [bs, self.n_agents, self.n_embed])
        b1 = tf.reshape(b1, [bs, 1, self.n_embed])
        hidden = tf.nn.elu(tf.matmul(agent_qs, w1) + b1)
        # Second layer
        w2 = self.hyper2w_1(obs)
        w2 = self.hyper2w_2(w2)
        w2 = tf.abs(w2)
        w2 = tf.reshape(w2, [bs, self.n_embed, 1])
        # State dependent bias
        v = self.v_1(obs)
        v = self.v_2(v)
        v = tf.reshape(v, [bs, 1, 1])
        # QTOT
        y = tf.matmul(hidden, w2)
        q_tot = y + v
        return q_tot[:, 0, 0]


class Network(tf.keras.Model):

    def __init__(self, batch_size=1, n_agents=2):
        super(Network, self).__init__()
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        self.pd = CategoricalPd()
        self.eps = tf.Variable(initial_value=0.0, name="eps", trainable=False)
        self.update_eps = tf.Variable(initial_value=0.0, name="update_eps", trainable=False)
        with tf.name_scope("qvalue"):
            self.qmixer = QMixer(n_agents)
            self.mlp1_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.q = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(1.0))
        with tf.name_scope("target_qvalue"):
            self.tqmixer = QMixer(n_agents, trainable=False)
            self.mlp1_tq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp2_tq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp3_tq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.tq = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(1.0), trainable=False)

    def update_target(self):
        self.tqmixer.set_weights(self.qmixer.get_weights())
        self.mlp1_tq.set_weights(self.mlp1_q.get_weights())
        self.mlp2_tq.set_weights(self.mlp2_q.get_weights())
        self.mlp3_tq.set_weights(self.mlp3_q.get_weights())
        self.tq.set_weights(self.q.get_weights())

    def polyak_qnet(self):
        polyak = 0.995
        self.mlp1_tq.set_weights([polyak * tq1 + (1 - polyak) * q1 for tq1, q1 in zip(self.mlp1_tq.get_weights(), self.mlp1_q.get_weights())])
        self.mlp2_tq.set_weights([polyak * tq2 + (1 - polyak) * q2 for tq2, q2 in zip(self.mlp2_tq.get_weights(), self.mlp2_q.get_weights())])
        self.mlp3_tq.set_weights([polyak * tq3 + (1 - polyak) * q3 for tq3, q3 in zip(self.mlp3_tq.get_weights(), self.mlp3_q.get_weights())])
        self.tq.set_weights([polyak * tq + (1 - polyak) * q for tq, q in zip(self.tq.get_weights(), self.q.get_weights())])

    def get_q_value(self, obs, last_act):
        last_act = tf.one_hot(last_act, 5, dtype=tf.float32)
        q_input = tf.concat([obs, last_act], axis=-1)
        q = self.mlp1_q(q_input)
        q = self.mlp2_q(q)
        q = self.mlp3_q(q)
        q = self.q(q)
        return q
    
    def get_tq_value(self, obs, last_act):
        last_act = tf.one_hot(last_act, 5, dtype=tf.float32)
        tq_input = tf.concat([obs, last_act], axis=-1)
        tq = self.mlp1_tq(tq_input)
        tq = self.mlp2_tq(tq)
        tq = self.mlp3_tq(tq)
        tq = self.tq(tq)
        return tq
    
    def get_neglogp(self, logits, action):
        return self.pd.neglogp(logits, action)
    
    def call(self, obs, last_act, stochastic=True):
        q = self.get_q_value(obs, last_act)
        deterministic_actions = tf.argmax(q, axis=1)
        bs = obs.shape[0]
        random_actions = tf.random.uniform(tf.stack([bs]), minval=0, maxval=5, dtype=tf.int64)
        chose_random = tf.random.uniform(tf.stack([bs]), minval=0, maxval=1, dtype=tf.float32) < self.eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
        output_actions = tf.cond(stochastic, lambda: stochastic_actions, lambda: deterministic_actions)
        eps_to_assign = tf.cond(self.update_eps >= 0, lambda: self.update_eps, lambda: self.eps)
        self.eps.assign(eps_to_assign)
        return output_actions
