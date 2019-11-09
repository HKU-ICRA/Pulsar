import tensorflow as tf

class DeepMlp(tf.keras.layers.Layer):

    def __init__(self, num_units, num_layers):
        super(DeepMlp, self).__init__()
        assert num_layers > 1, 'num_layers has to be greater than 1'
        self.linear_layers = [tf.keras.layers.Dense(num_units, name="linear_layer_" + str(i)) for i in range(num_layers - 1)]
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization(axis=0, scale=False, fused=None) for i in range(num_layers - 1)]
        self.output_layer = tf.keras.layers.Dense(num_units, name="output_layer")
    
    def call(self, x, training):
        output = x
        for linear_layer, batch_norm_layer in zip(self.linear_layers, self.batch_norm_layers):
            output = linear_layer(output)
            output = batch_norm_layer(output, training)
            output = tf.nn.relu(output)
        output = self.output_layer(output)
        output = tf.nn.relu(output)
        return output
