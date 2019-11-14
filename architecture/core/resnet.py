import tensorflow as tf

class Resnet(tf.keras.layers.Layer):

    def __init__(self, n_blocks, n_units):
        super(Resnet, self).__init__()
        self.batch_norms = []
        self.relus = []
        self.linear_layers = []
        for idx in range(n_blocks):
            self.batch_norms.append(tf.keras.layers.LayerNormalization())
            self.relus.append(tf.keras.layers.ReLU())
            self.linear_layers.append(tf.keras.layers.Dense(n_units))
    
    def call(self, inputs):
        outputs = inputs
        for batch_norm, relu, linear in zip(self.batch_norms, self.relus, self.linear_layers):
            shortcut = outputs
            outputs = batch_norm(outputs)
            outputs = relu(outputs)
            outputs = linear(outputs)
            outputs += shortcut
        return outputs
