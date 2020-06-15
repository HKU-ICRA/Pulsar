import numpy as np
import tensorflow as tf
import sys


class ortho_init(tf.keras.initializers.Initializer):

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, shape, dtype=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (self.scale * q[:shape[0], :shape[1]]).astype(np.float32)


class Lstm(tf.keras.layers.Layer):
    
    def __init__(self, batch_size, bptt_len, n_in, hidden_units, layer_norm=False):
        super(Lstm, self).__init__()
        self.batch_size = batch_size
        self.bptt_len = bptt_len
        self.hidden_units = hidden_units
        self.layer_norm = layer_norm
        self.wx = tf.Variable(name="wx", initial_value=ortho_init(1.0)([n_in, hidden_units*4]), trainable=True)
        self.wh = tf.Variable(name="wh", initial_value=ortho_init(1.0)([hidden_units, hidden_units*4]), trainable=True)
        self.b = tf.Variable(name="b", initial_value=tf.constant_initializer(0.0)([hidden_units*4]), trainable=True)
        if layer_norm:
            # Gain and bias of layer norm
            self.gain_x = tf.Variable(name="gx", initial_value=tf.constant_initializer(1.0)([hidden_units*4]))
            self.bias_x = tf.Variable(name="bx", initial_value=tf.constant_initializer(0.0)([hidden_units*4]))

            self.gain_h = tf.Variable(name="gh", initial_value=tf.constant_initializer(1.0)([hidden_units*4]))
            self.bias_h = tf.Variable(name="bh", initial_value=tf.constant_initializer(0.0)([hidden_units*4]))

            self.gain_c = tf.Variable(name="gc", initial_value=tf.constant_initializer(1.0)([hidden_units]))
            self.bias_c = tf.Variable(name="bc", initial_value=tf.constant_initializer(0.0)([hidden_units]))

    def ln(self, inputs, gain, bias, epsilon=1e-5, axes=None):
        """
         Apply layer normalisation.
        :param input_tensor: (TensorFlow Tensor) The input tensor for the Layer normalization
        :param gain: (TensorFlow Tensor) The scale tensor for the Layer normalization
        :param bias: (TensorFlow Tensor) The bias tensor for the Layer normalization
        :param epsilon: (float) The epsilon value for floating point calculations
        :param axes: (tuple, list or int) The axes to apply the mean and variance calculation
        :return: (TensorFlow Tensor) a normalizing layer
        """
        if axes is None:
            axes = [1]
        mean, variance = tf.nn.moments(inputs, axes=axes, keepdims=True)
        outputs = (inputs - mean) / tf.sqrt(variance + epsilon)
        outputs = outputs * gain + bias
        return outputs

    def get_initial_state(self):
        shape = [self.batch_size*self.bptt_len, 2*self.hidden_units]
        initial_state = np.zeros(shape, dtype=np.float32)
        return initial_state

    def call(self, x, initial_state, mask):
        outputs, new_state = self.lstm_fp(x, mask, initial_state)
        return outputs, new_state

    def lstm_fp(self, xs, ms, s):
        s = tf.reshape(s, [self.batch_size, self.bptt_len, 2*self.hidden_units])[:, 0]
        c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
        for idx, (x, m) in enumerate(zip(xs, ms)):
            c = c*(1-m)
            h = h*(1-m)
            if self.layer_norm:
                z = self.ln(tf.matmul(x, self.wx), self.gain_x, self.bias_x) + self.ln(tf.matmul(h, self.wh), self.gain_h, self.bias_h) + self.b
            else:
                z = tf.matmul(x, self.wx) + tf.matmul(h, self.wh) + self.b
            i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            u = tf.tanh(u)
            c = f*c + i*u
            if self.layer_norm:
                h = o*tf.tanh(self.ln(c, self.gain_c, self.bias_c))
            else:
                h = o*tf.tanh(c)
            xs[idx] = h
        s = tf.concat(axis=1, values=[c, h])
        return xs, s


class DeepLstm(tf.keras.layers.Layer):

    def __init__(self, batch_size, bptt_len, n_in, hidden_units=384, num_lstm=3, layer_norm=False, name=None):
        super(DeepLstm, self).__init__(name=name)
        self.batch_size = batch_size
        self.bptt_len = bptt_len
        self.num_lstm = num_lstm
        self.lstms = [Lstm(batch_size, bptt_len, n_in, hidden_units, layer_norm=layer_norm)]
        for _ in range(num_lstm-1):
            self.lstms.append(Lstm(batch_size, bptt_len, hidden_units, hidden_units))
    
    def batch_to_seq(self, h, nbatch, nsteps, flat=False):
        if flat:
            h = tf.reshape(h, [nbatch, nsteps])
        else:
            h = tf.reshape(h, [nbatch, nsteps, -1])
        return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

    def seq_to_batch(self, h, flat=False):
        shape = h[0].get_shape().as_list()
        if not flat:
            assert(len(shape) > 1)
            nh = h[0].get_shape()[-1]
            return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
        else:
            return tf.reshape(tf.stack(values=h, axis=1), [-1])

    def get_initial_states(self):
        initial_states = np.asarray([lstm.get_initial_state() for lstm in self.lstms])
        initial_states = initial_states.swapaxes(0, 1)
        return initial_states

    def call(self, x, initial_states, mask):
        initial_states = tf.transpose(initial_states, perm=[1, 0, 2])
        new_states = []
        x = self.batch_to_seq(x, self.batch_size, self.bptt_len)
        mask = self.batch_to_seq(mask, self.batch_size, self.bptt_len)
        for idx, lstm in enumerate(self.lstms):
            x, new_state = lstm(x, initial_states[idx], mask)
            new_states.append(new_state)
        x = self.seq_to_batch(x)
        new_states = tf.transpose(new_states, perm=[1, 0, 2])
        return x, new_states
