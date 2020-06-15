import tensorflow as tf

class Resnet(tf.keras.layers.Layer):

    def __init__(self, n_blocks, n_units, activation, w_reg=None, b_reg=None, name=None):
        super(Resnet, self).__init__(name=name)
        self.batch_norms = []
        self.activation = activation
        self.linear_layers = []
        for idx in range(n_blocks):
            self.batch_norms.append(tf.keras.layers.LayerNormalization())
            self.linear_layers.append(tf.keras.layers.Dense(n_units, kernel_regularizer=w_reg, bias_regularizer=b_reg))
    
    def call(self, inputs):
        outputs = inputs
        for batch_norm, linear in zip(self.batch_norms, self.linear_layers):
            shortcut = outputs
            outputs = batch_norm(outputs)
            outputs = self.activation(outputs)
            outputs = linear(outputs)
            outputs += shortcut
        return outputs
