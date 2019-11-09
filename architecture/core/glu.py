import tensorflow as tf


class Glu(tf.keras.layers.Layer):

    def __init__(self, input_size, output_size):
        super(Glu, self).__init__()
        self.gate_layer = tf.keras.layers.Dense(input_size, name="gate_layer")
        self.linear_layer = tf.keras.layers.Dense(output_size, name="linear_layer")
    
    def call(self, x, context):
        gate = tf.math.sigmoid(self.gate_layer(context))
        output = gate * x
        output = self.linear_layer(output)
        return output
