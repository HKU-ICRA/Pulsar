import numpy as np
import tensorflow as tf

from architecture.core.deeplstm import DeepLstm
from architecture.core.deepmlp import DeepMlp
from architecture.entity_encoder.model_utils import get_padding_bias
from architecture.entity_encoder.transformer import Transformer
from architecture.scalar_encoder.embedding_layer import Embedding_layer
from architecture.pulsar import Pulsar


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
    scalar_features = {'match_time': np.array([[120], [110], [100]])}
    scalar_features['bptt_match_time'] = np.repeat(np.expand_dims(np.array([[120], [110], [100]], dtype=np.float32), axis=1), repeats=10, axis=1)
    entities = np.array([[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [1, 1, 0]], [[0, 0, 1], [1, 1, 0]]], dtype=np.float32)
    entities = np.repeat(np.expand_dims(entities, axis=1), repeats=10, axis=1)
    entity_masks = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.float32)
    baseline = np.array([[0,1,1], [0,0,1], [0,1,0]], dtype=np.float32)
    print("scalar features:", scalar_features['match_time'].shape)
    print("entities:", entities.shape)
    print("entity masks:", entity_masks.shape)
    print("baseline:", baseline.shape)
    actions, neglogp, entropy, mean, value, new_state = pulsar(scalar_features, entities, entity_masks, baseline)
    #actions, neglogp, entropy, mean, value, new_state = pulsar(scalar_features, entities, entity_masks, baseline, new_state)
    for k, v in actions.items():
        print(k + ":", actions[k].shape, neglogp[k].shape, entropy[k].shape)
    print('value:', value.shape)
    

#test_scalar_encoder()
#test_entity_encoder()
#test_deeplstm()
#test_deepmlp()
test_pulsar()
