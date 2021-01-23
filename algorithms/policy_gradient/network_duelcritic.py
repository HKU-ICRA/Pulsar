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


class Network(tf.keras.Model):

    def __init__(self, batch_size=1):
        super(Network, self).__init__()
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        self.pd = CategoricalPd()
        with tf.name_scope("policy"):
            self.mlp1_p = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_p = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_p = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.p = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(0.01))
        with tf.name_scope("target_policy"):
            self.mlp1_tp = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp2_tp = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp3_tp = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.tp = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(0.01), trainable=False)
        with tf.name_scope("advantage"):
            self.mlp1_a = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_a = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_a = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.a = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(1.0))
        with tf.name_scope("target_advantage"):
            self.mlp1_ta = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp2_ta = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp3_ta = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.ta = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(1.0), trainable=False)
        with tf.name_scope("value"):
            self.mlp1_v = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_v = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_v = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.v = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(1.0))
        with tf.name_scope("target_value"):
            self.mlp1_tv = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp2_tv = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp3_tv = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.tv = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(1.0), trainable=False)

    def get_neglogp(self, logits, action):
        return self.pd.neglogp(logits, action)

    def polyak_anet(self):
        polyak = 0.995
        self.mlp1_ta.set_weights([polyak * ta1 + (1 - polyak) * a1 for ta1, a1 in zip(self.mlp1_ta.get_weights(), self.mlp1_a.get_weights())])
        self.mlp2_ta.set_weights([polyak * ta2 + (1 - polyak) * a2 for ta2, a2 in zip(self.mlp2_ta.get_weights(), self.mlp2_a.get_weights())])
        self.mlp3_ta.set_weights([polyak * ta3 + (1 - polyak) * a3 for ta3, a3 in zip(self.mlp3_ta.get_weights(), self.mlp3_a.get_weights())])
        self.ta.set_weights([polyak * ta + (1 - polyak) * a for ta, a in zip(self.ta.get_weights(), self.a.get_weights())])
    
    def polyak_vnet(self):
        polyak = 0.995
        self.mlp1_tv.set_weights([polyak * tv1 + (1 - polyak) * v1 for tv1, v1 in zip(self.mlp1_tv.get_weights(), self.mlp1_v.get_weights())])
        self.mlp2_tv.set_weights([polyak * tv2 + (1 - polyak) * v2 for tv2, v2 in zip(self.mlp2_tv.get_weights(), self.mlp2_v.get_weights())])
        self.mlp3_tv.set_weights([polyak * tv3 + (1 - polyak) * v3 for tv3, v3 in zip(self.mlp3_tv.get_weights(), self.mlp3_v.get_weights())])
        self.tv.set_weights([polyak * tv + (1 - polyak) * v for tv, v in zip(self.tv.get_weights(), self.v.get_weights())])

    def polyak_pnet(self):
        polyak = 0.995
        self.mlp1_tp.set_weights([polyak * tp1 + (1 - polyak) * p1 for tp1, p1 in zip(self.mlp1_tp.get_weights(), self.mlp1_p.get_weights())])
        self.mlp2_tp.set_weights([polyak * tp2 + (1 - polyak) * p2 for tp2, p2 in zip(self.mlp2_tp.get_weights(), self.mlp2_p.get_weights())])
        self.mlp3_tp.set_weights([polyak * tp3 + (1 - polyak) * p3 for tp3, p3 in zip(self.mlp3_tp.get_weights(), self.mlp3_p.get_weights())])
        self.tp.set_weights([polyak * tp + (1 - polyak) * p for tp, p in zip(self.tp.get_weights(), self.p.get_weights())])

    @tf.function
    def get_tp(self, obs):
        tp = self.mlp1_tp(obs)
        tp = self.mlp2_tp(tp)
        tp = self.mlp3_tp(tp)
        tp = self.tp(tp)
        t_action = self.pd.sample(tp)
        return t_action

    @tf.function
    def get_a_value(self, obs, action):
        action_1H = tf.one_hot(action, 5, dtype=tf.float32)
        a_input = tf.concat([obs, action_1H], axis=-1)    
        a = self.mlp1_a(a_input)
        a = self.mlp2_a(a)
        a = self.mlp3_a(a)
        a = self.a(a)[:, 0]
        return a
    
    @tf.function
    def get_ta_value(self, obs, action):
        action_1H = tf.one_hot(action, 5, dtype=tf.float32)
        ta_input = tf.concat([obs, action_1H], axis=-1)    
        ta = self.mlp1_ta(ta_input)
        ta = self.mlp2_ta(ta)
        ta = self.mlp3_ta(ta)
        ta = self.ta(ta)[:, 0]
        return ta
    
    @tf.function
    def get_value(self, obs):
        v = self.mlp1_v(obs)
        v = self.mlp2_v(v)
        v = self.mlp3_v(v)
        v = self.v(v)[:, 0]
        return v
    
    @tf.function
    def get_tvalue(self, obs):
        tv = self.mlp1_tv(obs)
        tv = self.mlp2_tv(tv)
        tv = self.mlp3_tv(tv)
        tv = self.tv(tv)[:, 0]
        return tv 

    @tf.function
    def get_q_value(self, obs, action):
        v = self.get_value(obs)
        a = self.get_a_value(obs, action)
        advantages = []
        for ai in range(5):
            advantages.append(self.get_a_value(obs, np.array([ai for _ in range(obs.shape[0])])))
        advantages = tf.transpose(advantages, [1, 0])
        return v + (a - tf.reduce_mean(advantages, axis=1))
    
    @tf.function
    def get_tq_value(self, obs, action):
        tv = self.get_tvalue(obs)
        ta = self.get_ta_value(obs, action)
        tadvantages = []
        for ai in range(5):
            tadvantages.append(self.get_ta_value(obs, np.array([ai for _ in range(obs.shape[0])])))
        tadvantages = tf.transpose(tadvantages, [1, 0])
        return tv + (ta - tf.reduce_mean(tadvantages, axis=1))

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
        if taken_action != None:
            taken_action = taken_action[0]
            taken_action_neglogp = self.get_neglogp(p, taken_action)
            return action, neglogp, entropy, probs, p, taken_action_neglogp
        return action, neglogp, entropy, probs, p
