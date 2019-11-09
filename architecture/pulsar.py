import numpy as np
import tensorflow as tf

from core.deeplstm import DeepLstm
from entity_encoder.model_utils import get_padding_bias
from entity_encoder.transformer import Transformer
from scalar_encoder.embedding_layer import Embedding_layer

class Pulsar(tf.keras.Model):

    def __init__(self):
        super(Pulsar, self).__init__()
        # Scalar encoder
        self.match_time_encoder = Embedding_layer(num_units=64)
        # Entity encoder
        self.transformer = Transformer(num_trans_layers=3, hidden_size=256, num_heads=2,
                                       attention_dropout=0.5, hidden_layer_size=1024,
                                       train=True)
        # Core
        self.deeplstm = DeepLstm(hidden_units=384, num_lstm=3)
    
    def call(self, scalar_features, entities, entity_masks, state=None):
        """
        Foward-pass neural network Pulsar.

        Args:
            scalar_features: dict of each scalar features. dict should include
                'match_time' : seconds. Required shape = [batch, 1]

        Returns:
            new_state: the new deep lstm state
        """
        # Scalar encoder
        encoded_match_time = self.match_time_encoder(scalar_features['match_time'])
        embedded_scalar = tf.concat([encoded_match_time], axis=-1)
        scalar_context = tf.concat([encoded_match_time], axis=-1)
        # Entity encoder
        bias = get_padding_bias(entity_masks)
        entity_embeddings, embedded_entity = self.transformer(entities, bias)
        # Core
        core_input = tf.concat([embedded_entity, embedded_scalar], axis=-1)
        if state == None:
            state = self.deeplstm.get_initial_state(core_input)
        core_output, new_state = self.deeplstm(core_input, state)
