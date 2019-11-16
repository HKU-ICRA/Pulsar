import tensorflow as tf

from architecture.entity_encoder.attention_layer import SelfAttention


class Transformer_layer(tf.keras.layers.Layer):

    def __init__(self, hidden_size=256, num_heads=2, attention_dropout=0.5, hidden_layer_size=1024, train=True):
        super(Transformer_layer, self).__init__()
        self.self_attention = SelfAttention(hidden_size, num_heads, attention_dropout, train)
        self.hidden_layer = tf.keras.layers.Dense(hidden_layer_size, use_bias=True, name="hidden_layer")
        self.output_layer = tf.keras.layers.Dense(hidden_size, use_bias=True, name="output_layer")    # Output layer's size == hidden_size as output needs to feed back into another Transformer layer

    def call(self, x, bias):
        """Apply one transformer layer to input.
        Args:
            x: A tensor with shape [batch_size, length, hidden_size]
            bias: attention bias that will be added to the result of the dot product.
        Returns:
            A tensor with shape [batch_size, length, hidden_size]
        """
        output = self.self_attention(x, bias)
        output = self.hidden_layer(output)
        output = self.output_layer(output)
        return output


class Transformer(tf.keras.layers.Layer):

    def __init__(self, num_trans_layers=3, hidden_size=256, num_heads=2, attention_dropout=0.5, hidden_layer_size=1024, train=True):
        super(Transformer, self).__init__()
        self.transformer_layers = []
        for i in range(num_trans_layers):
            self.transformer_layers.append(Transformer_layer(hidden_size, num_heads, attention_dropout, hidden_layer_size, train))
        self.conv1d = tf.keras.layers.Conv2D(hidden_size, 1, name="conv1d")
        self.linear_layer = tf.keras.layers.Dense(hidden_size, use_bias=True, name="linear_layer")
    
    def call(self, x, bias):
        """Applies entity encoding to input.
        Args:
            x: A tensor with shape [batch_size, length, hidden_size]
            bias: attention bias that will be added to the result of the dot product.
        Returns:
            A tensor with shape [batch_size, length, hidden_size]
        """
        output = x
        for transform_layer in self.transformer_layers:
            output = transform_layer(output, bias)
        output = tf.nn.relu(output)
        output = self.conv1d(output)
        output = tf.nn.relu(output)
        entity_embeddings = output
        embedded_entity = tf.math.reduce_mean(output, axis=2)
        embedded_entity =  self.linear_layer(embedded_entity)
        embedded_entity = tf.nn.relu(embedded_entity)
        return entity_embeddings, embedded_entity
