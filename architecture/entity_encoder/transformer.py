import tensorflow as tf

from architecture.entity_encoder.attention_layer import ResidualSelfAttention, entity_avg_pooling_masked


class Transformer(tf.keras.layers.Layer):

    def __init__(self, num_attent_layers=1, n_embd=256, n_heads=2, n_mlp=1, name=None):
        super(Transformer, self).__init__(name)
        self.residualSelfAttentions = [ResidualSelfAttention(heads=n_heads, n_embd=n_embd, n_mlp=n_mlp, layer_norm=True, post_sa_layer_norm=False) for _ in range(num_attent_layers)]
    
    def call(self, x, mask):
        """Applies entity encoding to input.
        Args:
            x: A tensor with shape [batch_size, length, hidden_size]
            bias: attention bias that will be added to the result of the dot product.
        Returns:
            A tensor with shape [batch_size, length, hidden_size]
        """
        outputs = x
        for residualSelfAttention in self.residualSelfAttentions:
            outputs = residualSelfAttention(outputs, mask)
        embedded_entity = entity_avg_pooling_masked(outputs, mask)
        return embedded_entity
