import numpy as np
import tensorflow as tf


def entity_avg_pooling_masked(x, mask):
    '''
        Masks and pools x along the second to last dimension. Arguments have dimensions:
            x:    batch x time x n_entities x n_features
            mask: batch x time x n_entities
    '''
    mask = tf.expand_dims(mask, -1)
    masked = x * mask
    summed = tf.reduce_sum(masked, -2)
    denom = tf.reduce_sum(mask, -2) + 1e-5
    return summed / denom
    

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
        return outputs


class ResidualSelfAttention(tf.keras.Model):
    '''
        Residual self attention block for entities.
        Notation:
            T  - Time
            NE - Number entities
        Args:
            inp (tf): (BS, T, NE, f)
            mask (tf): (BS, T, NE)
            heads (int) -- number of attention heads
            n_embd (int) -- dimension of queries, keys, and values will be n_embd / heads
            layer_norm (bool) -- normalize embedding prior to computing qkv
            n_mlp (int) -- number of mlp layers. If there are more than 1 mlp layers, we'll add a residual
                connection from after the first mlp to after the last mlp.
            qk_w, v_w, post_w, mlp_w1, mlp_w2 (float) -- scale for gaussian init for keys/queries, values, mlp
                post self attention, second mlp, and third mlp, respectively. Std will be sqrt(scale/n_embd)
    '''
    def __init__(self, heads, n_embd, layer_norm=False, post_sa_layer_norm=False,
                 n_mlp=1, qk_w=0.125, v_w=0.125, post_w=0.125, mlp_w1=0.125, mlp_w2=0.125):
        super(ResidualSelfAttention, self).__init__()
        self.n_mlp = n_mlp
        self.selfAttention = SelfAttention(heads, n_embd, layer_norm=layer_norm, qk_w=qk_w, v_w=v_w)
        post_scale = np.sqrt(post_w / n_embd)
        self.post_selfAttention_mlp = tf.keras.layers.Dense(n_embd, kernel_initializer=tf.random_normal_initializer(stddev=post_scale), name="mlp1")
        self.post_sa_layer_norm = post_sa_layer_norm
        if post_sa_layer_norm:
            self.post_sa_layer_norm_1 = tf.keras.layers.LayerNormalization(axis=3)
        if n_mlp > 1:
            mlp2_scale = np.sqrt(mlp_w1 / n_embd)
            self.mlp_1 = tf.keras.layers.Dense(n_embd, kernel_initializer=tf.random_normal_initializer(stddev=mlp2_scale), name="mlp2")
        if n_mlp > 2:
            mlp3_scale = np.sqrt(mlp_w2 / n_embd)
            self.mlp_2 = tf.keras.layers.Dense(n_embd, kernel_initializer=tf.random_normal_initializer(stddev=mlp3_scale), name="mlp3")
    
    def call(self, inputs, mask):
        outputs = self.selfAttention(inputs, mask)
        post_a_mlp = self.post_selfAttention_mlp(outputs)
        outputs = inputs + post_a_mlp
        if self.post_sa_layer_norm:
            outputs = self.post_sa_layer_norm_1(outputs)
        if self.n_mlp > 1:
            mlp = outputs
            mlp = self.mlp_1(mlp)
        if self.n_mlp > 2:
            mlp = self.mlp_2(mlp)
        if self.n_mlp > 1:
            outputs = outputs + mlp
        return outputs
