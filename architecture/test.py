import numpy as np
import tensorflow as tf

from core.deeplstm import DeepLstm
from entity_encoder.model_utils import get_padding_bias
from entity_encoder.transformer import Transformer
from scalar_encoder.embedding_layer import Embedding_layer


def test_core():
    x = np.array([[[1, 2]], [[3, 4]]], dtype=np.float32)
    init_state = [None, None]
    deeplstm = DeepLstm()
    output, new_state = deeplstm(x, deeplstm.get_initial_state(x))
    print("Output:", output, "\n", "new_state:", new_state)


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


#test_scalar_encoder()
#test_entity_encoder()
#test_core()
