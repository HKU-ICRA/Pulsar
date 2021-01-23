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
        up = logits - tf.math.log(-tf.math.log(u))
        return tf.argmax(up, axis=-1), up



class normc_initializer(tf.keras.initializers.Initializer):
    def __init__(self, std=1.0, axis=0):
        self.std = std
        self.axis = axis
    def __call__(self, shape, dtype=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= self.std / np.sqrt(np.square(out).sum(axis=self.axis, keepdims=True))
        return tf.constant(out)


class FeedForwardNet(tf.keras.layers.Layer):

    def __init__(self, layer_sizes, activation=None, output_nonlinearity=None, name="FeedForwardNet"):
        super(FeedForwardNet, self).__init__(name=name)
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.output_nonlinearity = output_nonlinearity
        self.denses = [tf.keras.layers.Dense(layer_size) for layer_size in layer_sizes]
        self.latent_noise = tf.keras.layers.GaussianNoise(1.0)
    
    def call(self, inputs):
        outputs = inputs
        for idx in range(len(self.layer_sizes)):
            outputs = self.denses[idx](outputs)
            if idx == 0:
                outputs = self.latent_noise(outputs)
            if idx == len(self.layer_sizes) - 1 and self.activation != None:
                outputs = self.activation(outputs)
        if self.output_nonlinearity != None:
            outputs = self.output_nonlinearity(outputs)
        return outputs


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
        with tf.name_scope("qvalue"):
            self.mlp1_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_q = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.q = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(1.0))
        with tf.name_scope("joint_qvalue"):
            self.mlp1_jq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_jq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_jq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.jq = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(1.0))
        with tf.name_scope("target_joint_qvalue"):
            self.mlp1_tjq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp2_tjq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.mlp3_tjq = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01), trainable=False)
            self.tjq = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(1.0), trainable=False)
        with tf.name_scope("value"):
            self.mlp1_v = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_v = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_v = tf.keras.layers.Dense(units=128, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.v = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(1.0))
        with tf.name_scope("opponent_conditional_policy"):
            self.ocp = FeedForwardNet([128, 128, 5], tf.nn.relu)

    def polyak_qnet(self):
        polyak = 0.995
        self.mlp1_tjq.set_weights([polyak * tq1 + (1 - polyak) * q1 for tq1, q1 in zip(self.mlp1_tjq.get_weights(), self.mlp1_jq.get_weights())])
        self.mlp2_tjq.set_weights([polyak * tq2 + (1 - polyak) * q2 for tq2, q2 in zip(self.mlp2_tjq.get_weights(), self.mlp2_jq.get_weights())])
        self.mlp3_tjq.set_weights([polyak * tq3 + (1 - polyak) * q3 for tq3, q3 in zip(self.mlp3_tjq.get_weights(), self.mlp3_jq.get_weights())])
        self.tjq.set_weights([polyak * tq + (1 - polyak) * q for tq, q in zip(self.tjq.get_weights(), self.jq.get_weights())])

    def get_neglogp(self, logits, action):
        return self.pd.neglogp(logits, action)

    def get_ocp(self, obs, action, n_action_samples=1):
        action_samples, action_probs = [], []
        for idx in range(n_action_samples):
            action_1H = tf.one_hot(action, 5, dtype=tf.float32)
            ocp_input = tf.concat([obs, action_1H], axis=-1)
            ocp = self.ocp(ocp_input)
            ocp_act, ocp_prob = self.pd.sample(ocp)
            action_samples.append(ocp_act)
            ocp_prob = self.pd.mean(ocp_prob)
            action_probs.append(ocp_prob)
        action_samples = tf.stack(action_samples, axis=1)
        action_probs = tf.stack(action_probs, axis=1)
        return action_probs

    @tf.function
    def get_q_value(self, obs, action):
        action_1H = tf.one_hot(action, 5, dtype=tf.float32)
        q_input = tf.concat([obs, action_1H], axis=-1)    
        q = self.mlp1_q(q_input)
        q = self.mlp2_q(q)
        q = self.mlp3_q(q)
        q = self.q(q)[:, 0]
        return q

    @tf.function
    def get_jq_value(self, obs, action, other_action):
        action_1H = tf.one_hot(action, 5, dtype=tf.float32)
        jq_input = tf.concat([obs, action_1H, other_action], axis=-1)    
        jq = self.mlp1_jq(jq_input)
        jq = self.mlp2_jq(jq)
        jq = self.mlp3_jq(jq)
        jq = self.jq(jq)[:, 0]
        return jq
    
    @tf.function
    def get_tjq_value(self, obs, action, other_action):
        action_1H = tf.one_hot(action, 5, dtype=tf.float32)
        tjq_input = tf.concat([obs, action_1H, other_action], axis=-1)    
        tjq = self.mlp1_tjq(tjq_input)
        tjq = self.mlp2_tjq(tjq)
        tjq = self.mlp3_tjq(tjq)
        tjq = self.tjq(tjq)[:, 0]
        return tjq
    
    @tf.function
    def get_value(self, obs):
        v = self.mlp1_v(obs)
        v = self.mlp2_v(v)
        v = self.mlp3_v(v)
        v = self.v(v)[:, 0]
        return v

    @tf.function
    def call(self, obs, taken_action=None):
        with tf.name_scope("policy"):
            p = self.mlp1_p(obs)
            p = self.mlp2_p(p)
            p = self.mlp3_p(p)
            p = self.p(p)
            probs = self.pd.mean(p)
            action, action_prob = self.pd.sample(p)
            neglogp = self.get_neglogp(p, action)
            entropy = self.pd.entropy(p)
        if taken_action != None:
            taken_action = taken_action[0]
            taken_action_neglogp = self.get_neglogp(p, taken_action)
            return action, neglogp, entropy, probs, p, taken_action_neglogp
        return action, neglogp, entropy, probs, p
