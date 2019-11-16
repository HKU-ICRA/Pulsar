import os
import numpy as np
import tensorflow as tf

from architecture.core.deeplstm import DeepLstm
from architecture.core.deepmlp import DeepMlp
from architecture.core.glu import Glu
from architecture.core.resnet import Resnet
from architecture.entity_encoder.model_utils import get_padding_bias
from architecture.entity_encoder.transformer import Transformer, Transformer_layer
from architecture.scalar_encoder.embedding_layer import Embedding_layer


def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer


class Pulsar(tf.keras.Model):

    def __init__(self, training):
        super(Pulsar, self).__init__()
        self.training = training
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, beta_1=0, beta_2=0.99, epsilon=1e-5)
        with tf.name_scope("scalar_encoder"):
            self.match_time_encoder = Embedding_layer(num_units=64)
        with tf.name_scope("entity_encoder"):
            self.transformer = Transformer(num_trans_layers=3, hidden_size=256, num_heads=2,
                                           attention_dropout=0.5, hidden_layer_size=1024,
                                           train=True)
        with tf.name_scope("core"):
            self.deeplstm = DeepLstm(hidden_units=384, num_lstm=3)
        with tf.name_scope("embedding_1"):
            mlp_1_units = 256
            self.deepmlp_1 = DeepMlp(num_units=mlp_1_units, num_layers=5)
            self.glu_1 = Glu(input_size=mlp_1_units, output_size=512)
        with tf.name_scope("embedding_2"):
            self.lstm_projection = tf.keras.layers.Dense(mlp_1_units, activation=tf.nn.relu, name="lstm_projection")
            self.deepmlp_2 = DeepMlp(num_units=256, num_layers=3)
            self.attention = Transformer_layer(hidden_size=256, num_heads=2, attention_dropout=0.5,
                                               hidden_layer_size=1024, train=training)
        with tf.name_scope("output"):
            self.mean_xy = tf.keras.layers.Dense(2,
                                              kernel_initializer=normc_initializer(0.01),
                                              activation=None,
                                              name="mean_xy")
            zeros_initializer_xy = tf.zeros_initializer()
            self.logstd_xy = tf.Variable(name="logstd_xy", initial_value=zeros_initializer_xy([1, 2]))
            self.mean_yaw = tf.keras.layers.Dense(1,
                                              kernel_initializer=normc_initializer(0.01),
                                              activation=None,
                                              name="mean_yaw")
            zeros_initializer_yaw = tf.zeros_initializer()
            self.logstd_yaw = tf.Variable(name="logstd_yaw", initial_value=zeros_initializer_yaw([1, 1]))
        with tf.name_scope("value"):
            self.value_encoder = tf.keras.layers.Dense(256, activation=tf.nn.relu, name="value_encoder")
            self.resnet = Resnet(n_blocks=10, n_units=256)
            self.value = tf.keras.layers.Dense(1, name="value")

    def load(self, model_path):
        pass
    
    def neglogp_xy(self, mean_xy, x):
        std_xy = tf.exp(self.logstd_xy)
        return 0.5 * tf.reduce_sum(tf.square((x - mean_xy) / std_xy), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.dtypes.cast(tf.shape(x)[-1], dtype=tf.float32) \
               + tf.reduce_sum(self.logstd_xy, axis=-1)
    
    def neglogp_yaw(self, mean_yaw, x):
        std_yaw = tf.exp(self.logstd_yaw)
        return 0.5 * tf.reduce_sum(tf.square((x - mean_yaw) / std_yaw), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.dtypes.cast(tf.shape(x)[-1], dtype=tf.float32) \
               + tf.reduce_sum(self.logstd_yaw, axis=-1)

    def call_build(self):
        """
        IMPORTANT: This function has to be editted so that the below input features
        have the same shape as the actual inputs, otherwise the weights would not
        be restored properly.
        """
        scalar_features = {'match_time': np.array([[120]], dtype=np.float32)}
        scalar_features['bptt_match_time'] = np.expand_dims(scalar_features['match_time'], axis=1)
        entities = np.array([[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]], dtype=np.float32)
        entities = np.repeat(np.expand_dims(entities, axis=1), repeats=1, axis=1)
        entity_masks = np.array([[1, 1, 1, 1, 1]], dtype=np.float32)
        baseline = np.array([[1 for _ in range(18)]], dtype=np.float32)
        self(scalar_features, entities, entity_masks, baseline)

    def call(self, scalar_features, entities, entity_masks, baseline, state=None):
        """
        Foward-pass neural network Pulsar.

        Args:
            scalar_features: dict of each scalar features. dict should include
                'match_time' : seconds. Required shape = [batch, 1]
            entities: array of entities. Required shape = [batch_size, n_entities, feature_size]
            entity_masks: mask for entities. Required shape = [batch_size, n_entities]
            state: previous lstm state. None for initial state.

        Returns:
            new_state: the new deep lstm state
        """
        with tf.name_scope("scalar_encoder"):
            encoded_match_time = self.match_time_encoder(scalar_features['match_time'])
            bptt_encoded_match_time = self.match_time_encoder(scalar_features['bptt_match_time'])
            embedded_scalar = tf.concat([bptt_encoded_match_time], axis=-1)
            scalar_context = tf.concat([encoded_match_time], axis=-1)
        with tf.name_scope("entity_encoder"):
            bias = get_padding_bias(entity_masks)
            bias = np.repeat(np.expand_dims(bias, axis=1), repeats=entities.shape[1], axis=1)
            entity_embeddings, embedded_entity = self.transformer(entities, bias)
            entity_embeddings = entity_embeddings[:, -1, :, :]
        with tf.name_scope("core"):
            core_input = tf.concat([embedded_entity, embedded_scalar], axis=-1)
            if state == None:
                state = self.deeplstm.get_initial_state(core_input)
            core_output, new_state = self.deeplstm(core_input, state)
        with tf.name_scope("embedding_1"):
            embedding_1 = self.deepmlp_1(core_output, self.training)
            action_xyvel_layer = self.glu_1(embedding_1, scalar_context)
        with tf.name_scope("embedding_2"):
            core_projection = self.lstm_projection(core_output)
            auto_regressive_embedding = core_projection + embedding_1
            embedding_2 = self.deepmlp_2(auto_regressive_embedding, self.training)
            embedding_2 = tf.expand_dims(embedding_2, axis=1)
            attention_input = tf.concat([embedding_2, entity_embeddings], axis=1)
            attention_input = tf.expand_dims(attention_input, axis=1)
            action_yaw_layer = self.attention(attention_input, 0)
            action_yaw_layer = tf.squeeze(action_yaw_layer, axis=1)
            action_yaw_layer = tf.reduce_sum(action_yaw_layer, axis=1)
        with tf.name_scope('xyvel'):
            std_xy = tf.exp(self.logstd_xy)
            mean_xy = self.mean_xy(action_xyvel_layer)
            sampled_xyvel = mean_xy + std_xy * tf.random.normal(tf.shape(mean_xy))
            sampled_xyvel_neglogp = self.neglogp_xy(mean_xy, sampled_xyvel)
            entropy_xy = tf.reduce_sum(self.logstd_xy + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        with tf.name_scope('yaw'):
            std_yaw = tf.exp(self.logstd_yaw)
            mean_yaw = self.mean_yaw(action_yaw_layer)
            sampled_yaw = mean_yaw + std_yaw * tf.random.normal(tf.shape(mean_yaw))
            sampled_yaw_neglogp = self.neglogp_yaw(mean_yaw, sampled_yaw)
            entropy_yaw = tf.reduce_sum(self.logstd_yaw + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        with tf.name_scope('value'):
            flattened_baseline = tf.reshape(baseline, shape=[tf.shape(core_output)[0], -1])
            value_input = tf.concat([core_output, flattened_baseline], axis=1)
            value_encode = self.value_encoder(value_input)
            value_output = self.resnet(value_encode)
            value = self.value(value_output)

        actions = {'xyvel': sampled_xyvel, 'yaw': sampled_yaw}
        neglogp = {'xyvel': sampled_xyvel_neglogp, 'yaw': sampled_yaw_neglogp}
        entropy = {'xyvel': entropy_xy, 'yaw': entropy_yaw}
        mean = {'xyvel': mean_xy, 'yaw': mean_yaw}

        return actions, neglogp, entropy, mean, value, new_state
