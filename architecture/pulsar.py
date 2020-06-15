import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import numpy as np
import tensorflow as tf

from architecture.core.glu import Glu
from architecture.core.resnet import Resnet
from architecture.core.deepmlp import DeepMlp
from architecture.core.deeplstm import DeepLstm
from architecture.entity_encoder.transformer import Transformer
from architecture.entity_encoder.model_utils import get_padding_bias
from architecture.entity_encoder.entity_formatter import Entity_formatter
from architecture.scalar_encoder.embedding_layer import Embedding_layer
from architecture.running_mean.running_mean import RunningMeanStd
from architecture.distributions.categorical import CategoricalPd


class normc_initializer(tf.keras.initializers.Initializer):
    def __init__(self, std=1.0, axis=0):
        self.std = std
        self.axis = axis
    def __call__(self, shape, dtype=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= self.std / np.sqrt(np.square(out).sum(axis=self.axis, keepdims=True))
        return tf.constant(out)


class Pulsar(tf.keras.Model):

    def __init__(self, batch_size=1, nsteps=1, training=True):
        super(Pulsar, self).__init__()
        self.batch_size = batch_size
        self.nsteps = nsteps
        self.training = training
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-6, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        self.entity_formatter = Entity_formatter()
        self.categoricalPd = CategoricalPd()
        with tf.name_scope("scalar_encoder"):
            n_scalar_embd = 64
            self.scalar_encoders = {
                "match_time": Embedding_layer(num_units=n_scalar_embd, activation=tf.nn.tanh, name="scalar_encoder_match_time"),
                "n_opponents": Embedding_layer(num_units=n_scalar_embd, activation=tf.nn.tanh, name="scalar_encoder_n_opponents")
            }
        with tf.name_scope("entity_encoder"):
            n_entity_embd = 64
            self.entity_encoders = {
                "my_qpos": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_my_qpos"),
                "my_qvel": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_my_qvel"),
                "local_qvel": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_my_qvel"),
                "teammate_qpos": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_teammate_qpos"),
                "opponent_qpos": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_opponent_qpos"),
                "my_hp": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_my_hp"),
                "teammate_hp": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_teammate_hp"),
                "opponent_hp": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_opponent_hp"),
                "my_projs": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_my_projs"),
                "teammate_projs": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_teammate_projs"),
                "opponent_projs": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_opponent_projs"),
                "my_armors": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_my_armors"),
                "teammate_armors": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_teammate_armors"),
                "my_hp_deduct": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_my_hp_deduct"),
                "my_hp_deduct_res": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_my_hp_deduct_res"),
                "zone": Embedding_layer(num_units=n_entity_embd, activation=tf.nn.tanh, name=f"entity_encoder_zone")
            }
        with tf.name_scope("transformer"):
            num_attent_layers = 1
            n_heads = 2
            n_mlp = 1
            self.transformer = Transformer(num_attent_layers=num_attent_layers, n_heads=n_heads,
                                           n_embd=n_entity_embd, n_mlp=n_mlp, name="transformer")
        with tf.name_scope("core"):
            n_lstm_input_units = 1024
            hidden_units = 1024
            num_lstm = 1
            self.coreInput_encoder = Embedding_layer(num_units=n_lstm_input_units, activation=tf.nn.tanh, name="coreInput_encoder")
            self.deeplstm = DeepLstm(self.batch_size, self.nsteps, n_lstm_input_units,
                                     hidden_units=hidden_units, num_lstm=num_lstm, layer_norm=True, name="deeplstm")
        with tf.name_scope("XYYaw_vel"):
            mlp_XYYAW_units = 256
            self.deepmlp_XYYAW = DeepMlp(num_units=mlp_XYYAW_units, num_layers=5, activation=tf.nn.tanh, name="deepmlp_XYYAW")
            #self.glu_XYYAW = Glu(input_size=mlp_XYYAW_units, output_size=mlp_XYYAW_units, name="glu_XYYAW")
            self.logits_x = tf.keras.layers.Dense(21,
                                              kernel_initializer=normc_initializer(0.01),
                                              name="logits_x")
            self.logits_y = tf.keras.layers.Dense(21,
                                              kernel_initializer=normc_initializer(0.01),
                                              name="logits_y")
            self.logits_yaw = tf.keras.layers.Dense(21,
                                              kernel_initializer=normc_initializer(0.01),
                                              name="logits_yaw")
        with tf.name_scope("opponent"):
            mlp_opponent_units = 256
            self.deepmlp_opponent = DeepMlp(num_units=mlp_opponent_units, num_layers=4, activation=tf.nn.tanh, w_reg=tf.keras.regularizers.l2, name="deepmlp_opponent")
            #self.glu_opponent = Glu(input_size=mlp_opponent_units, output_size=mlp_opponent_units, name="glu_opponent")
            self.logits_opponent = tf.keras.layers.Dense(3,
                                              kernel_initializer=normc_initializer(0.01),
                                              name="logits_opponent")
        with tf.name_scope("armor"):
            mlp_armor_units = 256
            self.deepmlp_armor = DeepMlp(num_units=mlp_armor_units, num_layers=4, activation=tf.nn.tanh, w_reg=tf.keras.regularizers.l2, name="deepmlp_armor")
            #self.glu_armor = Glu(input_size=mlp_armor_units, output_size=mlp_armor_units, name="glu_armor")
            self.logits_armor = tf.keras.layers.Dense(4,
                                              kernel_initializer=normc_initializer(0.01),
                                              name="logits_armor")
        with tf.name_scope("value"):
            self.value_encoder = Embedding_layer(num_units=256, activation=tf.nn.tanh, name="value_encoder")
            self.resnet = Resnet(n_blocks=8, n_units=256, activation=tf.nn.tanh, w_reg=tf.keras.regularizers.l2, name="resnet")
            self.value = tf.keras.layers.Dense(1, name="value", activation=None, kernel_initializer=normc_initializer(1.0))
    
    @tf.function
    def encode_all_scalars(self, scalars):
        embedded_scalars = []
        embedded_scalars.append(self.scalar_encoders['match_time'](scalars['match_time']))
        embedded_scalars.append(self.scalar_encoders['n_opponents'](scalars['n_opponents']))
        scalar_context = tf.concat(embedded_scalars, axis=-1)
        return scalar_context

    @tf.function
    def encode_all_entities(self, entities):
        embedded_entities = []
        embedded_entities.append(self.entity_encoders['my_qpos'](entities['my_qpos']))
        embedded_entities.append(self.entity_encoders['my_qvel'](entities['my_qvel']))
        embedded_entities.append(self.entity_encoders['local_qvel'](entities['local_qvel']))
        embedded_entities.append(self.entity_encoders['teammate_qpos'](entities['teammate_qpos']))
        embedded_entities.append(self.entity_encoders['opponent_qpos'](entities['opponent1_qpos']))
        embedded_entities.append(self.entity_encoders['opponent_qpos'](entities['opponent2_qpos']))
        embedded_entities.append(self.entity_encoders['my_hp'](entities['my_hp']))
        embedded_entities.append(self.entity_encoders['teammate_hp'](entities['teammate_hp']))
        embedded_entities.append(self.entity_encoders['opponent_hp'](entities['opponent1_hp']))
        embedded_entities.append(self.entity_encoders['opponent_hp'](entities['opponent2_hp']))
        embedded_entities.append(self.entity_encoders['my_projs'](entities['my_projs']))
        embedded_entities.append(self.entity_encoders['teammate_projs'](entities['teammate_projs']))
        embedded_entities.append(self.entity_encoders['opponent_projs'](entities['opponent1_projs']))
        embedded_entities.append(self.entity_encoders['opponent_projs'](entities['opponent2_projs']))
        embedded_entities.append(self.entity_encoders['my_armors'](entities['my_armors']))
        embedded_entities.append(self.entity_encoders['teammate_armors'](entities['teammate_armors']))
        embedded_entities.append(self.entity_encoders['my_hp_deduct'](entities['my_hp_deduct']))
        embedded_entities.append(self.entity_encoders['my_hp_deduct_res'](entities['my_hp_deduct_res']))
        embedded_entities.append(self.entity_encoders['zone'](entities['zone_1']))
        embedded_entities.append(self.entity_encoders['zone'](entities['zone_2']))
        embedded_entities.append(self.entity_encoders['zone'](entities['zone_3']))
        embedded_entities.append(self.entity_encoders['zone'](entities['zone_4']))
        embedded_entities.append(self.entity_encoders['zone'](entities['zone_5']))
        embedded_entities.append(self.entity_encoders['zone'](entities['zone_6']))
        embedded_entities = tf.concat(embedded_entities, axis=1)
        return embedded_entities

    @tf.function
    def call(self, scalar_features, entity_masks, entities, baseline, state, mask, taken_action=None, taken_logits=None, deterministic=False):
        """
        Foward-pass neural network Pulsar.

        Args:
            scalar_features: dict of each scalar features. dict should include
                'match_time' : seconds. Required shape = [batch, 1]
            entities: array of entities. Required shape = [batch_size, 1, n_entities, feature_size]
            entity_masks: mask for entities. Required shape = [batch_size, n_entities]
            state: previous lstm state. None for initial state.

        Returns:
            new_state: the new deep lstm state
        """
        with tf.name_scope("scalar_encoder"):
            scalar_context = self.encode_all_scalars(scalar_features)
        with tf.name_scope("entity_encoder"):
            embedded_entities = self.encode_all_entities(entities)
            embedded_entities  = tf.reshape(embedded_entities, [self.batch_size, self.nsteps] + embedded_entities.shape[1:])
            entity_masks = tf.reshape(entity_masks, [self.batch_size, self.nsteps, embedded_entities.shape[2]])
            embedded_entity = self.transformer(embedded_entities, entity_masks)
        with tf.name_scope("core"):
            embedded_entity = tf.reshape(embedded_entity, [self.batch_size * self.nsteps] + embedded_entity.shape[2:])
            embedded_scalar = tf.reshape(scalar_context, [self.batch_size * self.nsteps] + scalar_context.shape[1:])
            core_input = tf.concat([embedded_entity, embedded_scalar], axis=-1)
            embedded_coreInput = self.coreInput_encoder(core_input)
            core_output, new_states = self.deeplstm(embedded_coreInput, state, mask)
        with tf.name_scope("XYYaw_vel"):
            action_xyyaw_layer = self.deepmlp_XYYAW(core_output)
            #action_xyyaw_layer = self.glu_XYYAW(action_xyyaw_layer, scalar_context)
            logit_x = self.logits_x(action_xyyaw_layer)
            logit_y = self.logits_y(action_xyyaw_layer)
            logit_yaw = self.logits_yaw(action_xyyaw_layer)
            sampled_x = self.categoricalPd.sample(logit_x)
            sampled_y = self.categoricalPd.sample(logit_y)
            sampled_yaw = self.categoricalPd.sample(logit_yaw)
        with tf.name_scope("opponent"):
            action_opponent_layer = self.deepmlp_opponent(core_output)
            #action_opponent_layer = self.glu_opponent(action_opponent_layer, scalar_context)
            logit_opponent = self.logits_opponent(action_opponent_layer)
            sampled_opponent = self.categoricalPd.sample(logit_opponent)
        with tf.name_scope("armor"):
            action_armor_layer = self.deepmlp_armor(core_output)
            #action_armor_layer = self.glu_armor(action_armor_layer, scalar_context)
            logit_armor = self.logits_armor(action_armor_layer)
            sampled_armor = self.categoricalPd.sample(logit_armor)
        with tf.name_scope('value'):
            flattened_baseline = tf.reshape(baseline, shape=[tf.shape(core_output)[0], -1])
            value_input = tf.concat([core_output, flattened_baseline], axis=1)
            embedded_value = self.value_encoder(value_input)
            value_output = self.resnet(embedded_value)
            value = self.value(value_output)[:, 0]   # flatten value otherwise it might broadcast
        if not deterministic:
            actions = {'x': sampled_x, 'y': sampled_y, 'yaw': sampled_yaw, 'opponent': sampled_opponent, 'armor': sampled_armor}
        else:
            actions = {'x': self.categoricalPd.mean(logit_x), 'y': self.categoricalPd.mean(logit_y),
                       'yaw': self.categoricalPd.mean(logit_yaw), 'opponent': self.categoricalPd.mean(logit_opponent),
                       'armor': self.categoricalPd.mean(logit_armor)}
        logits = {'x': logit_x, 'y': logit_y, 'yaw': logit_yaw, 'opponent': logit_opponent, 'armor': logit_armor}
        neglogp = self.categoricalPd.neglogp(logit_x, sampled_x) + self.categoricalPd.neglogp(logit_y, sampled_y) + self.categoricalPd.neglogp(logit_yaw, sampled_yaw) + self.categoricalPd.neglogp(logit_opponent, sampled_opponent) + self.categoricalPd.neglogp(logit_armor, sampled_armor)
        entropy = self.categoricalPd.entropy(logit_x) + self.categoricalPd.entropy(logit_y) + self.categoricalPd.entropy(logit_yaw) + self.categoricalPd.entropy(logit_opponent) + self.categoricalPd.entropy(logit_armor)
        if taken_action != None:
            taken_action_neglogp = self.categoricalPd.neglogp(logit_x, taken_action['x']) + self.categoricalPd.neglogp(logit_y, taken_action['y']) + self.categoricalPd.neglogp(logit_yaw, taken_action['yaw']) + self.categoricalPd.neglogp(logit_opponent, taken_action['opponent']) + self.categoricalPd.neglogp(logit_armor, taken_action['armor'])
            all_logits = tf.concat([logit_x, logit_y, logit_yaw, logit_opponent, logit_armor], axis=-1)
            all_taken_logits = tf.concat([taken_logits['x'], taken_logits['y'], taken_logits['yaw'], taken_logits['opponent'], taken_logits['armor']], axis=-1)
            kl = self.categoricalPd.kl(all_logits, all_taken_logits)
            return actions, neglogp, entropy, value, new_states, state, taken_action_neglogp, kl
        return actions, neglogp, entropy, value, new_states, state, logits
    
    def get_initial_states(self):
        return self.deeplstm.get_initial_states()

    def call_build(self, learner=False):
        """
        IMPORTANT: This function has to be editted so that the below input features
        have the same shape as the actual inputs, otherwise the weights would not
        be restored properly.
        """
        n_agents = 4
        obs = {'observation_self': np.zeros([n_agents, 6]),
                'F1': np.zeros([4, 1]),
                'F2': np.zeros([4, 1]),
                'F3': np.zeros([4, 1]),
                'F4': np.zeros([4, 1]),
                'F5': np.zeros([4, 1]),
                'F6': np.zeros([4, 1]),
                'Agent:buff': np.zeros([n_agents, 4, 1]),
                'colli_dmg': np.zeros([n_agents, 1]),
                'nprojectiles': np.zeros([n_agents, 1]),
                'proj_dmg': np.array([['0', 'n'] for _ in range(n_agents)], dtype='<U11'),
                'agent_teams': np.array([['red'] for _ in range(n_agents)]),
                'agents_health': np.zeros([n_agents, 1]),
                'agent_local_qvel': np.zeros([n_agents, 2])
        }
        scalar_features = {'match_time': np.array([[120] for _ in range(self.batch_size * self.nsteps)], dtype=np.float32),
                           'n_opponents': np.array([[1] for _ in range(self.batch_size * self.nsteps)], dtype=np.float32)}
        entity_masks, entities = self.entity_formatter.concat_encoded_entity_obs(n_agents, 0, obs)
        entities = {k: np.concatenate([v for _ in range(self.batch_size * self.nsteps)], axis=0) for k, v in entities.items()}
        entity_masks = np.concatenate([entity_masks for _ in range(self.batch_size * self.nsteps)], axis=0)
        baseline = self.entity_formatter.get_baseline(n_agents, 0, obs, 0)
        baseline = np.concatenate([baseline for _ in range(self.batch_size * self.nsteps)], axis=0)
        state = self.get_initial_states()
        mask  = np.array([False for _ in range(self.batch_size * self.nsteps)], dtype=np.float32)
        with tf.GradientTape() as tape:
            actions, neglogp, entropy, value, states, prev_state, logits = self(scalar_features, entity_masks, entities, baseline, state, mask)
            loss = tf.dtypes.cast(tf.reduce_mean(actions['x']), tf.float32) +        \
                   tf.dtypes.cast(tf.reduce_mean(actions['y']), tf.float32) +        \
                   tf.dtypes.cast(tf.reduce_mean(actions['yaw']), tf.float32) +      \
                   tf.dtypes.cast(tf.reduce_mean(actions['opponent']), tf.float32) + \
                   tf.dtypes.cast(tf.reduce_mean(actions['armor']), tf.float32) +    \
                   tf.dtypes.cast(tf.reduce_mean(value), tf.float32)
            loss *= 0.0
        if learner:
            grads = tape.gradient(loss, self.trainable_variables)
            grads_and_var = zip(grads, self.trainable_variables)
            self.optimizer.apply_gradients(grads_and_var)

    def get_all_weights(self):
        all_weights = dict()
        # Network
        for k, scalar_encoder in self.scalar_encoders.items():
            all_weights[k] = scalar_encoder.get_weights()
        for k, entity_encoder in self.entity_encoders.items():
            all_weights[k] = entity_encoder.get_weights()
        all_weights['transformer'] = self.transformer.get_weights()
        all_weights['coreInput_encoder'] = self.coreInput_encoder.get_weights()
        all_weights['deeplstm'] = self.deeplstm.get_weights()
        all_weights['deepmlp_XYYAW'] = self.deepmlp_XYYAW.get_weights()
        #all_weights['glu_XYYAW'] = self.glu_XYYAW.get_weights()
        all_weights['logits_x'] = self.logits_x.get_weights()
        all_weights['logits_y'] = self.logits_y.get_weights()
        all_weights['logits_yaw'] = self.logits_yaw.get_weights()
        all_weights['deepmlp_opponent'] = self.deepmlp_opponent.get_weights()
        #all_weights['glu_opponent'] = self.glu_opponent.get_weights()
        all_weights['logits_opponent'] = self.logits_opponent.get_weights()
        all_weights['deepmlp_armor'] = self.deepmlp_armor.get_weights()
        #all_weights['glu_armor'] = self.glu_armor.get_weights()
        all_weights['logits_armor'] = self.logits_armor.get_weights()
        all_weights['value_encoder'] = self.value_encoder.get_weights()
        all_weights['resnet'] = self.resnet.get_weights()
        all_weights['value'] = self.value.get_weights()
        # Optimizer
        all_weights['adam'] = self.optimizer.get_weights()
        return all_weights

    def set_all_weights(self, all_weights, learner=False):
        # Network
        for k in self.scalar_encoders.keys():
            if k in all_weights:
                try:
                    self.scalar_encoders[k].set_weights(all_weights[k])
                except:
                    print("Failed to restore weights for layer:", k)
        for k in self.entity_encoders.keys():
            if k in all_weights:
                try:
                    self.entity_encoders[k].set_weights(all_weights[k])
                except:
                    print("Failed to restore weights for layer:", k)
        if 'transformer' in all_weights:
            try:
                self.transformer.set_weights(all_weights['transformer'])
            except:
                print("Failed to restore weights for layer:", 'transformer')
        if 'coreInput_encoder' in all_weights:
            try:
                self.coreInput_encoder.set_weights(all_weights['coreInput_encoder'])
            except:
                print("Failed to restore weights for layer:", 'coreInput_encoder')
        if 'deeplstm' in all_weights:
            try:
                self.deeplstm.set_weights(all_weights['deeplstm'])
            except:
                print("Failed to restore weights for layer:", 'deeplstm')
        if 'deepmlp_XYYAW' in all_weights:
            try:
                self.deepmlp_XYYAW.set_weights(all_weights['deepmlp_XYYAW'])
            except:
                print("Failed to restore weights for layer:", 'deepmlp_XYYAW')
        #if 'glu_XYYAW' in all_weights:
        #    try:
        #        self.glu_XYYAW.set_weights(all_weights['glu_XYYAW'])
        #    except:
        #        print("Failed to restore weights for layer:", 'glu_XYYAW')
        if 'logits_x' in all_weights:
            try:
                self.logits_x.set_weights(all_weights['logits_x'])
            except:
                print("Failed to restore weights for layer:", 'logits_x')
        if 'logits_y' in all_weights:
            try:
                self.logits_y.set_weights(all_weights['logits_y'])
            except:
                print("Failed to restore weights for layer:", 'logits_y')
        if 'logits_yaw' in all_weights:
            try:
                self.logits_yaw.set_weights(all_weights['logits_yaw'])
            except:
                print("Failed to restore weights for layer:", 'logits_yaw')
        if 'deepmlp_opponent' in all_weights:
            try:
                self.deepmlp_opponent.set_weights(all_weights['deepmlp_opponent'])
            except:
                print("Failed to restore weights for layer:", 'deepmlp_opponent')
        #if 'glu_opponent' in all_weights:
        #    try:
        #        self.glu_opponent.set_weights(all_weights['glu_opponent'])
        #    except:
        #        print("Failed to restore weights for layer:", 'glu_opponent')
        if 'logits_opponent' in all_weights:
            try:
                self.logits_opponent.set_weights(all_weights['logits_opponent'])
            except:
                print("Failed to restore weights for layer:", 'logits_opponent')
        if 'deepmlp_armor' in all_weights:
            try:
                self.deepmlp_armor.set_weights(all_weights['deepmlp_armor'])
            except:
                print("Failed to restore weights for layer:", 'deepmlp_armor')
        #if 'glu_armor' in all_weights:
        #    try:
        #        self.glu_armor.set_weights(all_weights['glu_armor'])
        #    except:
        #        print("Failed to restore weights for layer:", 'glu_armor')
        if 'logits_armor' in all_weights:
            try:
                self.logits_armor.set_weights(all_weights['logits_armor'])
            except:
                print("Failed to restore weights for layer:", 'logits_armor')
        if 'value_encoder' in all_weights:
            try:
                self.value_encoder.set_weights(all_weights['value_encoder'])
            except:
                print("Failed to restore weights for layer:", 'value_encoder')
        if 'resnet' in all_weights:
            try:
                self.resnet.set_weights(all_weights['resnet'])
            except:
                print("Failed to restore weights for layer:", 'resnet')
        if 'value' in all_weights:
            try:
                self.value.set_weights(all_weights['value'])
            except:
                print("Failed to restore weights for layer:", 'value')
        # Optimizer
        if learner:
            if 'adam' in all_weights:
                try:
                    self.optimizer.set_weights(all_weights['adam'])
                except:
                    print("Failed to restore weights for optimizer:", 'adam')
        sys.stdout.flush()
    
    def set_optimizer(self, learning_rate=3e-6):
        self.optimizer.lr = learning_rate
