import tensorflow as tf

class Embedding_layer(tf.keras.layers.Layer):

    def __init__(self, num_units=64):
        super(Embedding_layer, self).__init__()
        self.linear_layer = tf.keras.layers.Dense(num_units, activation=tf.nn.relu, name="linear_layer")
    
    def call(self, x):
        """Apply a linear layer to embed inputs.
        Args:
            x: A tensor with shape [batch_size, features]
        Returns:
            A tensor with shape [batch_size, num_units]
        """
        output = self.linear_layer(x)
        return output
