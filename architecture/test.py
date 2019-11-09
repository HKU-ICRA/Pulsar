import numpy as np
import tensorflow as tf

from core.deeplstm import DeepLstm
from core.deepmlp import DeepMlp
from entity_encoder.model_utils import get_padding_bias
from entity_encoder.transformer import Transformer
from scalar_encoder.embedding_layer import Embedding_layer
from pulsar import Pulsar


def test_deeplstm():
    x = np.array([[1, 2], [3, 4], [3, 5]], dtype=np.float32)
    deeplstm = DeepLstm()
    output, new_state = deeplstm(x, deeplstm.get_initial_state(x))
    print("Output:", np.array(output).shape, "\n", "new_state:", np.array(new_state).shape)


def test_deepmlp():
    x = np.array([[1], [2]], dtype=np.float32)
    deepmlp = DeepMlp(num_units=128, num_layers=3)
    output = deepmlp(x, True)
    print("Output:", output.shape)


def test_entity_encoder():
    entity_encoder = Transformer()
    x = np.array([[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [1, 1, 0]]], dtype=np.float32)
    mask = np.array([[0, 1], [1, 0]], dtype=np.float32)
    bias = get_padding_bias(mask)
    entity_embeddings, embedded_entity = entity_encoder(x, bias)
    print(entity_embeddings.shape, embedded_entity.shape)


def test_scalar_encoder():
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)

    x_embed_layer = Embedding_layer(64)
    x_embed = x_embed_layer(x)
    print("Output:", x_embed.shape)


def test_pulsar():
    pulsar = Pulsar(True)
    scalar_features = {'match_time': np.array([[120], [110]])}
    entities = np.array([[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [1, 1, 0]]], dtype=np.float32)
    entity_masks = np.array([[0, 1], [1, 0]], dtype=np.float32)
    action_xyvel_layer, action_yaw_layer = pulsar(scalar_features, entities, entity_masks)
    print(action_xyvel_layer.shape, action_yaw_layer.shape)


#test_scalar_encoder()
#test_entity_encoder()
#test_deeplstm()
#test_deepmlp()
test_pulsar()
