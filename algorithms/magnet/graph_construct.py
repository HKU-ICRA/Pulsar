import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, units, zero_pad=True, scale=True):
        super(positional_encoding, self).__init__()
        self.units = units
        self.zero_pad = zero_pad
        self.scale = scale
    
    def call(self, inputs):
        N, T = inputs.get_shape().as_list()
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / self.units) for i in range(self.units)]
            for pos in range(T)])
        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)
        if self.zero_pad:
            lookup_table = tf.concat([tf.zeros(shape=[1, num_units]), lookup_table[1:, :]], axis=0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
        if self.scale:
            outputs = outputs * num_units ** 0.5
        return outputs


class MultiheadAttention(tf.keras.layers.Layer):

    def __init__(self, units=None, num_heads=8, dropout_rate=0, training=True, causality=False):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.training = training
        self.causality = causality
        self.Q = tf.keras.layers.Dense(units=units, activation=tf.nn.relu)
        self.K = tf.keras.layers.Dense(units=units, activation=tf.nn.relu)
        self.v = tf.keras.layers.Dense(units=units, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
    
    def call(self, queries, keys):
        # Linear projections
        q = self.Q(queries)
        k = self.K(keys)
        v = self.V(keys)
        # Split and concat
        Q_ = tf.concat(tf.split(q, self.num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(k, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(v, self.num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
        # Causality = Future blinding
        if self.causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [self.num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)
        # Dropouts
        outputs = self.dropout(outputs)
        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)  # (N, T_q, C)
        # Residual connection
        outputs += queries
        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)
        return outputs


class Normalize(tf.keras.layers.Layer):

    def __init__(self, input_shape, epsilon=1e-8):
        super(Normalize, self).__init__()
        self.epsilon = epsilon
        self.beta = tf.Variable(tf.zeros(input_shape))
        self.gamma = tf.Variable(tf.ones(input_shape))
    
    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self.epsilon) ** (.5))
        normalized = tf.dtypes.cast(normalized, tf.float32)
        outputs = self.gamma * normalized + self.beta
        return outputs


class FeedForward(tf.keras.layers.Layer):

    def __init__(self, input_shape, multi_units=[2048, 512]):
        super(FeedForward, self).__init__()
        self.conv1_1 = tf.keras.layers.Conv1D(filters=multi_units[0], kernel_size=1, activation=tf.nn.relu, use_bias=True)
        self.conv1_2 = tf.keras.layers.Conv1D(filters=multi_units[1], kernel_size=1, activation=None, use_bias=True)
        self.normalize = Normalize(input_shape=input_shape)
    
    def call(self, inputs):
        outputs = self.conv1_1(inputs)
        outputs = self.conv1_2(outputs)
        outputs = outputs + inputs
        outputs = self.normalize(outputs)
        return outputs


class GraphConstruction(tf.keras.layers.Layer):

    def __init__(self):
        super(GraphConstruction, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        # State 1
        self.conv1_state1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)
        self.conv1_pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2_state1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)
        self.conv2_pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        # State 2
        self.conv1_state2 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)
        self.conv1_pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2_state2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)
        self.conv2_pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        # model
        self.fcn = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        # Self Attention
        self.positional_encoding = PositionalEncoding(units=512, zero_pad=False, scale=False)
        self.encoder_dropout = tf.keras.layers.Dropout(rate=0.1)
        # Encoder blocks
        self.enc_block_attentions, self.enc_block_feedforwards = [], []
        for i in range(6):
            self.enc_block_attentions.append(MultiheadAttention(units=512, num_heads=8, dropout_rate=0.1, training=True, causality=False))
            self.enc_block_feedforwards.append(FeedForward(input_shape, multi_units=[4 * 512, 512]))
        # Decoder
        self.dec_positional_encoding = PositionalEncoding(units=512, zero_pad=False, scale=False)
        self.decoder_dropout = tf.keras.layers.Dropout(rate=0.1)
        # Decoder blocks
        self.dec_block_attentions, self.dec_vanblock_attentions, self.dec_block_feedforwards = [], [], []
        for i in range(6):
            self.dec_block_attentions.append(MultiheadAttention(units=512, num_heads=8, dropout_rate=0.1, training=True, causality=True))
            self.dec_vanblock_attentions.append(MultiheadAttention(units=512, num_heads=8, dropout_rate=0.1, training=True, causality=False))
            self.dec_block_feedforwards.append(FeedForward(input_shape, multi_units=[4 * 512, 512]))
        # Final linear projection
        self.final_dense = tf.keras.layers.Dense(units=1024)
        self.final_dropout = tf.keras.layers.Dropout(rate=0.4)
        self.logits = tf.keras.layers.Dense(units=480)

    def train(self, state_1, state_2, labels):
        logits = self(state_1, state_2, labels, True)
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        self.optimizer.minimize(loss, self.trainable_variables)

    def call(self, state_1, state_2, labels, training=True):
        # State 1
        output_state1 = self.conv1_state1(state_1)
        output_state1 = self.conv1_pool1(output_state1)
        output_state1 = self.conv2_state1(output_state1)
        output_state1 = self.conv2_pool1(output_state1)
        output_state1 = tf.reshape(output_state1, [-1, ...])
        # State 2
        output_state2 = self.conv1_state2(state_2)
        output_state2 = self.conv1_pool2(output_state2)
        output_state2 = self.conv2_state2(output_state2)
        output_state2 = self.conv2_pool2(output_state2)
        output_state2 = tf.reshape(output_state2, [-1, ...])
        # Model
        whole_model = tf.concat([output_state2, output_state1], axis=1)
        whole_model = self.fcn(whole_model)
        # Self attention
        decoder_inputs = tf.concat([tf.ones_like(labels[:, :1]) * 2, labels[:, :-1]], axis=-1)
        # Encoder
        enc = self.positional_encoding(dense)
        enc = self.encoder_dropout(enc, True)
        # Encoder blocks
        for i in range(6):
            enc = self.enc_block_attentions[i](queries=enc, keys=enc)
            enc = self.enc_block_feedforwards[i](enc)
        # Decoder
        dec = self.dec_positional_encoding(decoder_inputs)
        dec = self.decoder_dropout(dec, True)
        # Decoder blocks
        for i in range(6):
            dec = self.dec_block_attentions[i](queries=dec, keys=dec)
            dec = self.dec_vanblock_attentions[i](queries=dec, keys=dec)
            dec = self.dec_block_feedforwards[i](dec)
        # Final linear projection
        dec = self.final_dense(dec)
        dec = tf.reshape(dec, [1, -1])
        dec = self.final_dropout(dec, training)
        logits = self.logits(dec)   # shape of matrix -- 120 * 4
        return logits
