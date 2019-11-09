import tensorflow as tf


class Glu(tf.keras.layers.Layer):

    def __init__(self, gate_size, output_size):
        super(Glu, self).__init__()
        self.gate_layer = tf.keras.layers.Dense(gate_size)
        self.linear_layer = tf.keras.layers.Dense(output_size)
    
    def call(self, x, context):
        gate = tf.math.sigmoid(self.gate_layer(context))
        output = gate * x
        output = self.linear_layer(output)
        return output
