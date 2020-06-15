import numpy as np
import tensorflow as tf
import time

from architecture.core.deeplstm import DeepLstm
from architecture.core.deepmlp import DeepMlp
from architecture.entity_encoder.entity_formatter import Entity_formatter
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


def test_pular_time():
    entity_encoder = Entity_encoder()
    obs = {'observation_self': np.array([
                                            [ 6.96633793e+02,  4.82145933e+03,  1.05850126e+01,
                                                2.23475785e+00,  1.07207089e+02,  1.07705289e+02,
                                                1.97496876e+01, -2.00294798e-02
                                            ],
                                            [ 7.92263580e+03,  7.47302116e+02,  1.05891407e+01,
                                                -7.83228433e-01,  3.54040821e+01,  9.56652674e+01,
                                                1.98216989e+01, -3.17819067e-03
                                            ]
                                        ]),
            'agent_qpos_qvel': np.array([
                                            [
                                                [ 7.92263580e+03,  7.47302116e+02,  1.05891407e+01,
                                                    -7.83228433e-01,  3.54040821e+01,  9.56652674e+01,
                                                    1.98216989e+01, -3.17819067e-03
                                                ]
                                            ],

                                            [
                                                [ 6.96633793e+02,  4.82145933e+03,  1.05850126e+01,
                                                    2.23475785e+00,  1.07207089e+02,  1.07705289e+02,
                                                    1.97496876e+01, -2.00294798e-02
                                                ]
                                            ]
                                        ]),
            'F1': np.array([[0.], [0.], [1.], [0.]]),
            'F2': np.array([[0.], [0.], [1.], [0.]]),
            'F3': np.array([[1.], [0.], [0.], [0.]]),
            'F4': np.array([[0.], [0.], [0.], [1.]]),
            'F5': np.array([[0.], [0.], [1.], [0.]]),
            'F6': np.array([[0.], [0.], [1.], [0.]]),
            'Agent:buff': np.array([[[0], [0], [0], [0]], [[0], [0], [0], [0]]]),
            'colli_dmg': np.array([[ 0], [10]]),
            'nprojectiles': np.array([[50], [50]]),
            'proj_dmg': np.array([['0', 'n'], ['0', 'n']], dtype='<U11'),
            'agent_teams': np.array([['red'], ['blue']]),
            'agents_health': np.array([[2000.], [1700.]])
            }
    scalar_features = {'match_time': np.array([[120] for _ in range(pulsar.batch_size * pulsar.nsteps)], dtype=np.float32)}
    scalar_features['team'] = np.array([[0] for _ in range(pulsar.batch_size * pulsar.nsteps)], dtype=np.float32)
    mask, entities = entity_encoder.concat_encoded_entity_obs(2, 0, obs)
    print(mask.shape)
    entities = np.concatenate([entities for _ in range(pulsar.batch_size * pulsar.nsteps)], axis=0)
    baseline = entity_encoder.get_baseline(2, obs)
    baseline = np.concatenate([baseline for _ in range(pulsar.batch_size * pulsar.nsteps)], axis=0)
    state = pulsar.get_initial_states()
    mask  = np.array([False for _ in range(pulsar.batch_size * pulsar.nsteps)], dtype=np.float32)
    #pulsar.call_build()
    start_time = time.time()
    print(entities.shape)
    actions, neglogp, entropy, value, states, prev_state = pulsar(scalar_features, entities,
                                                                                   baseline, state,
                                                                                   mask)
    print("--- %s seconds ---" % (time.time() - start_time))
    

def test_pulsar():
    pulsar = Pulsar(2, 1, False)
    pulsar.call_build(True)
    pulsar2 = Pulsar(2, 1, False)
    print(tf.keras.optimizers.serialize(pulsar.optimizer))
    return


#test_scalar_encoder()
#test_entity_encoder()
#test_deeplstm()
#test_deepmlp()
test_pulsar()
