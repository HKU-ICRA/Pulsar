import tensorflow as tf

class DeepMlp(tf.keras.layers.Layer):

    def __init__(self, num_units, num_layers, activation, w_reg=None, b_reg=None, name=None):
        super(DeepMlp, self).__init__(name=name)
        assert num_layers > 1, 'num_layers has to be greater than 1'
        self.linear_layers = [tf.keras.layers.Dense(num_units, kernel_regularizer=w_reg, bias_regularizer=b_reg,
                                                    name="linear_layer_" + str(i)) for i in range(num_layers - 1)]
        self.batch_norm_layers = [tf.keras.layers.LayerNormalization() for i in range(num_layers - 1)]
        self.output_layer = tf.keras.layers.Dense(num_units, kernel_regularizer=w_reg, bias_regularizer=b_reg,
                                                  name="output_layer")
        self.activation = activation
    
    def call(self, x, training=True):
        output = x
        for linear_layer, batch_norm_layer in zip(self.linear_layers, self.batch_norm_layers):
            output = linear_layer(output)
            output = batch_norm_layer(output)
            output = self.activation(output)
        output = self.output_layer(output)
        output = self.activation(output)
        return output
