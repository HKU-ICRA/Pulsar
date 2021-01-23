import os
import sys
import numpy as np
import tensorflow as tf

from scipy.linalg import circulant

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


class Qkv_embed(tf.keras.layers.Layer):
    '''
        Compute queries, keys, and values
        Args:
            inp (tf) -- tensor w/ shape (bs, T, NE, features)
            heads (int) -- number of attention heads
            n_embd (int) -- dimension of queries, keys, and values will be n_embd / heads
            layer_norm (bool) -- normalize embedding prior to computing qkv
            qk_w (float) -- Initialization scale for keys and queries. Actual scale will be
                sqrt(qk_w / #input features)
            v_w (float) -- Initialization scale for values. Actual scale will be sqrt(v_w / #input features)
            reuse (bool) -- tf reuse
    '''
    def __init__(self, heads, n_embd, layer_norm=False, qk_w=1.0, v_w=0.01):
        super(Qkv_embed, self).__init__()
        self.heads = heads
        self.n_embd = n_embd
        self.layer_norm = layer_norm
        if layer_norm:
            self.layer_norm_1 = tf.keras.layers.LayerNormalization(axis=3)
        qk_scale = np.sqrt(qk_w / n_embd)
        self.qk = tf.keras.layers.Dense(n_embd * 2, kernel_initializer=tf.random_normal_initializer(stddev=qk_scale), name="qk_embed")  # bs x T x n_embd*2
        v_scale = np.sqrt(v_w / n_embd)
        self.value = tf.keras.layers.Dense(n_embd, kernel_initializer=tf.random_normal_initializer(stddev=v_scale), name="v_embed")  # bs x T x n_embd
        
    def call(self, inputs):
        bs = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        NE = tf.shape(inputs)[2]
        features = tf.shape(inputs)[3]
        outputs = inputs
        if self.layer_norm:
            outputs = self.layer_norm_1(outputs)
        # qk shape (bs x T x NE x h x n_embd/h)
        qk = self.qk(outputs)
        qk = tf.reshape(qk, (bs, T, NE, self.heads, self.n_embd // self.heads, 2))
        # (bs, T, NE, heads, features)
        query, key = [tf.squeeze(x, -1) for x in tf.split(qk, 2, -1)]
        value = self.value(outputs)  # bs x T x n_embd
        value = tf.reshape(value, (bs, T, NE, self.heads, self.n_embd // self.heads))
        query = tf.transpose(query, (0, 1, 3, 2, 4), name="transpose_query")  # (bs, T, heads, NE, n_embd / heads)
        key = tf.transpose(key, (0, 1, 3, 4, 2), name="transpose_key")  # (bs, T, heads, n_embd / heads, NE)
        value = tf.transpose(value, (0, 1, 3, 2, 4),name="transpose_value")  # (bs, T, heads, NE, n_embd / heads)
        return query, key, value


class SelfAttention(tf.keras.layers.Layer):
    '''
        Self attention over entities.
        Notation:
            T  - Time
            NE - Number entities
        Args:
            inp (tf) -- tensor w/ shape (bs, T, NE, features)
            mask (tf) -- binary tensor with shape (bs, T, NE). For each batch x time,
                            nner matrix represents entity i's ability to see entity j
            heads (int) -- number of attention heads
            n_embd (int) -- dimension of queries, keys, and values will be n_embd / heads
            layer_norm (bool) -- normalize embedding prior to computing qkv
            qk_w, v_w (float) -- scale for gaussian init for keys/queries and values
                Std will be sqrt(scale/n_embd)
            scope (string) -- tf scope
            reuse (bool) -- tf reuse
    '''
    def __init__(self, heads, n_embd, layer_norm=False, qk_w=1.0, v_w=0.01):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.n_embd = n_embd
        self.qkv_embed = Qkv_embed(heads, n_embd, layer_norm=layer_norm, qk_w=qk_w, v_w=v_w)
        self.sigma = tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer=normc_initializer(0.01))

    def stable_masked_softmax(self, logits, mask):
        '''
            Args:
                logits (tf): tensor with shape (bs, T, heads, NE, NE)
                mask (tf): tensor with shape(bs, T, 1, NE)
        '''
        #  Subtract a big number from the masked logits so they don't interfere with computing the max value
        if mask is not None:
            mask = tf.expand_dims(mask, 2)
            logits -= (1.0 - mask) * 1e10

        #  Subtract the max logit from everything so we don't overflow
        logits -= tf.reduce_max(logits, axis=-1, keepdims=True)
        unnormalized_p = tf.exp(logits)

        #  Mask the unnormalized probibilities and then normalize and remask
        if mask is not None:
            unnormalized_p *= mask
        normalized_p = unnormalized_p / (tf.reduce_sum(unnormalized_p, axis=-1, keepdims=True) + 1e-10)
        if mask is not None:
            normalized_p *= mask
        return normalized_p

    def call(self, inputs, mask):
        bs = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        NE = tf.shape(inputs)[2]
        features = tf.shape(inputs)[3]
        # Put mask in format correct for logit matrix
        entity_mask = None
        if mask is not None:
            # NOTE: Mask and input should have the same first 3 dimensions
            entity_mask = mask
            mask = tf.expand_dims(mask, -2)  # (BS, T, 1, NE)
        query, key, value = self.qkv_embed(inputs)
        logits = tf.matmul(query, key, name="matmul_qk_parallel")  # (bs, T, heads, NE, NE)
        logits /= np.sqrt(self.n_embd / self.heads)
        softmax = self.stable_masked_softmax(logits, mask)
        att_sum = tf.matmul(softmax, value, name="matmul_softmax_value")  # (bs, T, heads, NE, features)
        outputs = tf.transpose(att_sum, (0, 1, 3, 2, 4))  # (bs, T, n_output_entities, heads, features)
        n_output_entities = tf.shape(outputs)[2]
        outputs = tf.reshape(outputs, (bs, T, n_output_entities, self.n_embd))  # (bs, T, n_output_entities, n_embd)
        outputs = self.sigma(outputs)
        return outputs


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
        with tf.name_scope("obs_encoder"):
            self.obs_enc1 = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
        with tf.name_scope("conv"):
            self.conv1 = SelfAttention(8, 64)
            self.conv2 = SelfAttention(8, 64)
        with tf.name_scope("policy"):
            self.mlp1_p = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_p = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_p = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.p = tf.keras.layers.Dense(units=5, kernel_initializer=normc_initializer(0.01))
        with tf.name_scope("value"):
            self.mlp1_v = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp2_v = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.mlp3_v = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh, kernel_initializer=normc_initializer(0.01))
            self.v = tf.keras.layers.Dense(units=1, kernel_initializer=normc_initializer(1.0))

    def get_neglogp(self, logits, action):
        return self.pd.neglogp(logits, action)

    def get_C(self, agent_idx, batch_size):
        if agent_idx == 0:
            vec = [1, 0, 0, 0]
        elif agent_idx == 1:
            vec = [0, 0, 0, 1]
        elif agent_idx == 2:
            vec = [0, 0, 1, 0]
        elif agent_idx == 3:
            vec = [0, 1, 0, 0]
        return np.array([circulant(vec) for _ in range(batch_size)], dtype=np.float32)

    @tf.function
    def call(self, obs, agent_idx, taken_action=None):
        # Value
        F = self.obs_enc1(obs)
        C = self.get_C(agent_idx, obs.shape[0])
        CF = tf.matmul(C, F)
        CF = tf.expand_dims(CF, axis=1)
        c1 = self.conv1(CF, None)
        c2 = self.conv2(c1, None)
        c1 = tf.squeeze(c1, axis=1)
        c2 = tf.squeeze(c2, axis=1)
        v_input = tf.concat([F, c1, c2], axis=-1)
        v_input = tf.reshape(v_input, [v_input.shape[0], -1])
        v = self.mlp1_v(v_input)
        v = self.mlp2_v(v)
        v = self.mlp3_v(v)
        v = self.v(v)[:, 0]
        # Policy
        p = self.mlp1_p(obs[:, agent_idx])
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
            return action, neglogp, entropy, v, probs, p, taken_action_neglogp
        return action, neglogp, entropy, v, probs, p


def test():
    net = Network()
    obs = np.array([[[0, 0, 0] for _ in range(4)]], dtype=np.float32)
    net(obs, 0)
