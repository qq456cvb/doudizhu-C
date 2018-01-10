import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn


def update_params(scope_from, scope_to):
    vars_from = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_from)
    vars_to = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_to)

    ops = []
    for from_var, to_var in zip(vars_from, vars_to):
        ops.append(to_var.assign(from_var))
    return ops

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class CardNetwork:
    def __init__(self, s_dim, trainer, scope, a_dim=9085):
        with tf.variable_scope(scope):
            #card_cnt = 57
            #self.temp = tf.placeholder(tf.float32, None, name="boltz")
            with tf.name_scope("inputs"):
                self.input_state = tf.placeholder(tf.float32, [None, s_dim], name="input")
                self.training = tf.placeholder(tf.bool, None, name="mode")
                self.input_single = tf.placeholder(tf.float32, [None, 15], name="input_single")
                self.input_pair = tf.placeholder(tf.float32, [None, 13], name="input_pair")
                self.input_triple = tf.placeholder(tf.float32, [None, 13], name="input_triple")
                self.input_quadric = tf.placeholder(tf.float32, [None, 13], name="input_quadric")

            # TODO: test if embedding would help
            with tf.name_scope("input_state_embedding"):
                self.embeddings = slim.fully_connected(
                    inputs=self.input_state, 
                    num_outputs=512, 
                    activation_fn=tf.nn.elu,
                    weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("reshaping_for_conv"):
                self.input_state_conv = tf.reshape(self.embeddings, [-1, 1, 512, 1])
                self.input_single_conv = tf.reshape(self.input_single, [-1, 1, 15, 1])
                self.input_pair_conv = tf.reshape(self.input_pair, [-1, 1, 13, 1])
                self.input_triple_conv = tf.reshape(self.input_triple, [-1, 1, 13, 1])
                self.input_quadric_conv = tf.reshape(self.input_quadric, [-1, 1, 13, 1])

            # convolution for legacy state
            with tf.name_scope("conv_legacy_state"):
                self.state_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_state_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1a = tf.layers.batch_normalization(self.state_conv1a_branch1a, training=self.training)
                self.state_nonlinear1a_branch1a = tf.nn.relu(self.state_bn1a_branch1a)

                self.state_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.state_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1b = tf.layers.batch_normalization(self.state_conv1a_branch1b, training=self.training)
                self.state_nonlinear1a_branch1b = tf.nn.relu(self.state_bn1a_branch1b)

                self.state_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.state_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch1c = tf.layers.batch_normalization(self.state_conv1a_branch1c, training=self.training)

                ######

                self.state_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_state_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.state_bn1a_branch2 = tf.layers.batch_normalization(self.state_conv1a_branch2, training=self.training)

                self.state1a = self.state_bn1a_branch1c + self.state_bn1a_branch2
                self.state_output = slim.flatten(tf.nn.relu(self.state1a))

            # convolution for single
            with tf.name_scope("conv_single"):
                self.single_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_single_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1a = tf.layers.batch_normalization(self.single_conv1a_branch1a, training=self.training)
                self.single_nonlinear1a_branch1a = tf.nn.relu(self.single_bn1a_branch1a)

                self.single_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1b = tf.layers.batch_normalization(self.single_conv1a_branch1b, training=self.training)
                self.single_nonlinear1a_branch1b = tf.nn.relu(self.single_bn1a_branch1b)

                self.single_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch1c = tf.layers.batch_normalization(self.single_conv1a_branch1c, training=self.training)

                ######

                self.single_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_single_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.single_bn1a_branch2 = tf.layers.batch_normalization(self.single_conv1a_branch2, training=self.training)

                self.single1a = self.single_bn1a_branch1c + self.single_bn1a_branch2
                self.single_output = slim.flatten(tf.nn.relu(self.single1a))

            # convolution for pair
            with tf.name_scope("conv_pair"):
                self.pair_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1a = tf.layers.batch_normalization(self.pair_conv1a_branch1a, training=self.training)
                self.pair_nonlinear1a_branch1a = tf.nn.relu(self.pair_bn1a_branch1a)

                self.pair_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1b = tf.layers.batch_normalization(self.pair_conv1a_branch1b, training=self.training)
                self.pair_nonlinear1a_branch1b = tf.nn.relu(self.pair_bn1a_branch1b)

                self.pair_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch1c = tf.layers.batch_normalization(self.pair_conv1a_branch1c, training=self.training)

                ######

                self.pair_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.pair_bn1a_branch2 = tf.layers.batch_normalization(self.pair_conv1a_branch2, training=self.training)

                self.pair1a = self.pair_bn1a_branch1c + self.pair_bn1a_branch2
                self.pair_output = slim.flatten(tf.nn.relu(self.pair1a))

            # convolution for triple
            with tf.name_scope("conv_triple"):
                self.triple_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1a = tf.layers.batch_normalization(self.triple_conv1a_branch1a, training=self.training)
                self.triple_nonlinear1a_branch1a = tf.nn.relu(self.triple_bn1a_branch1a)

                self.triple_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1b = tf.layers.batch_normalization(self.triple_conv1a_branch1b, training=self.training)
                self.triple_nonlinear1a_branch1b = tf.nn.relu(self.triple_bn1a_branch1b)

                self.triple_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch1c = tf.layers.batch_normalization(self.triple_conv1a_branch1c, training=self.training)

                ######

                self.triple_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv, num_outputs=64,
                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_bn1a_branch2 = tf.layers.batch_normalization(self.triple_conv1a_branch2, training=self.training)

                self.triple1a = self.triple_bn1a_branch1c + self.triple_bn1a_branch2
                self.triple_output = slim.flatten(tf.nn.relu(self.triple1a))

            # convolution for quadric
            with tf.name_scope("conv_quadric"):
                self.quadric_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1a = tf.layers.batch_normalization(self.quadric_conv1a_branch1a, training=self.training)
                self.quadric_nonlinear1a_branch1a = tf.nn.relu(self.quadric_bn1a_branch1a)

                self.quadric_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1b = tf.layers.batch_normalization(self.quadric_conv1a_branch1b, training=self.training)
                self.quadric_nonlinear1a_branch1b = tf.nn.relu(self.quadric_bn1a_branch1b)

                self.quadric_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1b, num_outputs=64,
                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch1c = tf.layers.batch_normalization(self.quadric_conv1a_branch1c, training=self.training)

                ######

                self.quadric_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.quadric_bn1a_branch2 = tf.layers.batch_normalization(self.quadric_conv1a_branch2, training=self.training)

                self.quadric1a = self.quadric_bn1a_branch1c + self.quadric_bn1a_branch2
                self.quadric_output = slim.flatten(tf.nn.relu(self.quadric1a))

            # 3 + 1 convolution
            with tf.name_scope("conv_3plus1"):
                tiled_triple = tf.tile(tf.expand_dims(self.input_triple, 1), [1, 15, 1])
                tiled_single = tf.tile(tf.expand_dims(self.input_single, 2), [1, 1, 13])
                self.input_triple_single_conv = tf.to_float(tf.expand_dims(tf.bitwise.bitwise_and(tf.to_int32(tiled_triple),
                    tf.to_int32(tiled_single)), -1))
                
                self.triple_single_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_triple_single_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1a = tf.layers.batch_normalization(self.triple_single_conv1a_branch1a, training=self.training)
                self.triple_single_nonlinear1a_branch1a = tf.nn.relu(self.triple_single_bn1a_branch1a)

                self.triple_single_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.triple_single_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[3, 3], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1b = tf.layers.batch_normalization(self.triple_single_conv1a_branch1b, training=self.training)
                self.triple_single_nonlinear1a_branch1b = tf.nn.relu(self.triple_single_bn1a_branch1b)

                self.triple_single_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.triple_single_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch1c = tf.layers.batch_normalization(self.triple_single_conv1a_branch1c, training=self.training)

                ######

                self.triple_single_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_triple_single_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_single_bn1a_branch2 = tf.layers.batch_normalization(self.triple_single_conv1a_branch2, training=self.training)

                self.triple_single1a = self.triple_single_bn1a_branch1c + self.triple_single_bn1a_branch2
                self.triple_single_output = slim.flatten(tf.nn.relu(self.triple_single1a))

            # 3 + 2 convolution
            with tf.name_scope("conv_3plus2"):
                tiled_triple = tf.tile(tf.expand_dims(self.input_triple, 1), [1, 13, 1])
                tiled_double = tf.tile(tf.expand_dims(self.input_pair, 2), [1, 1, 13])
                self.input_triple_double_conv = tf.to_float(tf.expand_dims(tf.bitwise.bitwise_and(tf.to_int32(tiled_triple),
                    tf.to_int32(tiled_double)), -1))
                
                self.triple_double_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_triple_double_conv, num_outputs=16,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1a = tf.layers.batch_normalization(self.triple_double_conv1a_branch1a, training=self.training)
                self.triple_double_nonlinear1a_branch1a = tf.nn.relu(self.triple_double_bn1a_branch1a)
    
                self.triple_double_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.triple_double_nonlinear1a_branch1a, num_outputs=16,
                                     kernel_size=[3, 3], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1b = tf.layers.batch_normalization(self.triple_double_conv1a_branch1b, training=self.training)
                self.triple_double_nonlinear1a_branch1b = tf.nn.relu(self.triple_double_bn1a_branch1b)

                self.triple_double_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.triple_double_nonlinear1a_branch1b, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch1c = tf.layers.batch_normalization(self.triple_double_conv1a_branch1c, training=self.training)

                ######

                self.triple_double_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_triple_double_conv, num_outputs=64,
                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                self.triple_double_bn1a_branch2 = tf.layers.batch_normalization(self.triple_double_conv1a_branch2, training=self.training)

                self.triple_double1a = self.triple_double_bn1a_branch1c + self.triple_double_bn1a_branch2
                self.triple_double_output = slim.flatten(tf.nn.relu(self.triple_double1a))

            #################################################

            with tf.name_scope("concated"):
                self.fc_flattened = tf.concat([self.single_output, self.pair_output, self.triple_output,
                    self.quadric_output, self.triple_single_output, self.triple_double_output, self.state_output], 1)

            # passive decision making  0: pass, 1: bomb, 2: king, 3: normal
            with tf.name_scope("passive_decision_making"):
                self.fc_decision_passive = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_decision_passive, slim.fully_connected, [128, 64, 32])
                self.fc_decision_passive_output = slim.fully_connected(self.fc_decision_passive, 4, tf.nn.softmax)

            # passive response
            with tf.name_scope("passive_response"):
                self.fc_response_passive = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_response_passive, slim.fully_connected, [128, 64, 32])
                self.fc_response_passive_output = slim.fully_connected(self.fc_response_passive, 14, tf.nn.softmax)

            # passive bomb response
            with tf.name_scope("passive_bomb_reponse"):
                self.fc_bomb_passive = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_bomb_passive, slim.fully_connected, [128, 64, 32])
                self.fc_bomb_passive_output = slim.fully_connected(self.fc_bomb_passive, 13, tf.nn.softmax)

            # active decision making  mapped to [action space category - 1]
            with tf.name_scope("active_decision_making"):
                self.fc_decision_active = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_decision_active, slim.fully_connected, [128, 64, 32])
                self.fc_decision_active_output = slim.fully_connected(self.fc_decision_active, 13, tf.nn.softmax)

            # active response
            with tf.name_scope("active_response"):
                self.fc_response_active = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_response_active, slim.fully_connected, [128, 64, 32])
                self.fc_response_active_output = slim.fully_connected(self.fc_response_active, 15, tf.nn.softmax)

            # card length output
            with tf.name_scope("fc_sequence_length_output"):
                self.fc_seq_length = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_seq_length, slim.fully_connected, [128, 64, 32])
                self.fc_sequence_length_output = slim.fully_connected(self.fc_seq_length, 12, tf.nn.softmax)

            # minor card value map output [-1, 1]
            with tf.name_scope("fc_cards_value_output"):
                self.fc_cards_value = self.fc_flattened
                with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                    slim.stack(self.fc_cards_value, slim.fully_connected, [128, 64, 32])
                self.fc_cards_value_output = slim.fully_connected(self.fc_cards_value, 15, tf.nn.tanh)

            # active or passive
            with tf.name_scope("input_is_active"):
                self.is_active = tf.placeholder(tf.bool, [None], name='active')

            with tf.name_scope("has_seq_length"):
                self.has_seq_length = tf.placeholder(tf.bool, [None], name='has_seq_length')

            # # minor cards
            # with tf.name_scope("minor_cards"):
            #     self.has_minor_cards = tf.placeholder(tf.bool, [None], name='minor_cards')
            #     self.minor_mask = tf.to_float(self.has_minor_cards)
            #     self.minor_cards_target = tf.placeholder(tf.float32, [None, 15], name='minor_cards_target')
            #     self.minor_loss = tf.reduce_sum(tf.square(self.minor_cards_target - self.fc_cards_value_output), 1)
            #     self.minor_loss = self.minor_mask * self.minor_loss

            # passive mode
            with tf.name_scope("passive_mode_loss"):
                self.is_passive_bomb = tf.placeholder(tf.bool, [None], name='passive_bomb')
                self.is_passive_king = tf.placeholder(tf.bool, [None], name='passive_is_king')
                self.did_passive_response = tf.placeholder(tf.bool, [None], name='did_passive_response')

                self.passive_decision_input = tf.placeholder(tf.int32, [None], name='passive_decision_in')
                self.passive_decision_target = tf.one_hot(self.passive_decision_input, 4)
                self.passive_decision_loss = -tf.reduce_sum(self.passive_decision_target * tf.log(tf.clip_by_value(self.fc_decision_passive_output, 1e-10, 1-(1e-10))), 1)

                self.passive_response_input = tf.placeholder(tf.int32, [None], name='passive_response_in')
                self.passive_response_target = tf.one_hot(self.passive_response_input, 14)
                self.passive_response_loss = -tf.reduce_sum(self.passive_response_target * tf.log(tf.clip_by_value(self.fc_response_passive_output, 1e-10, 1-(1e-10))), 1)

                self.passive_bomb_input = tf.placeholder(tf.int32, [None], name='passive_bomb_in')
                self.passive_bomb_target = tf.one_hot(self.passive_bomb_input, 13)
                self.passive_bomb_loss = -tf.reduce_sum(self.passive_bomb_target * tf.log(tf.clip_by_value(self.fc_bomb_passive_output, 1e-10, 1-(1e-10))), 1)

            # active mode
            with tf.name_scope("active_mode_loss"):
                self.active_decision_input = tf.placeholder(tf.int32, [None], name='active_decision_in')
                self.active_decision_target = tf.one_hot(self.active_decision_input, 13)
                self.active_decision_loss = -tf.reduce_sum(self.active_decision_target * tf.log(tf.clip_by_value(self.fc_decision_active_output, 1e-10, 1-(1e-10))), 1)

                self.active_response_input = tf.placeholder(tf.int32, [None], name='active_response_in')
                self.active_response_target = tf.one_hot(self.active_response_input, 15)
                self.active_response_loss = -tf.reduce_sum(self.active_response_target * tf.log(tf.clip_by_value(self.fc_response_active_output, 1e-10, 1-(1e-10))), 1)

                self.seq_length_input = tf.placeholder(tf.int32, [None], name='sequence_length_in')
                self.seq_length_target = tf.one_hot(self.seq_length_input, 12)
                self.seq_length_loss = -tf.to_float(self.has_seq_length) * tf.reduce_sum(self.seq_length_target * tf.log(tf.clip_by_value(self.fc_sequence_length_output, 1e-10, 1-(1e-10))), 1)


            with tf.name_scope("passive_loss"):
                self.passive_loss = tf.reduce_sum(self.passive_decision_loss + tf.to_float(self.did_passive_response) * self.passive_response_loss + \
                    tf.to_float(self.is_passive_bomb) * self.passive_bomb_loss)

            with tf.name_scope("active_loss"):
                self.active_loss = tf.reduce_sum(self.active_decision_loss + self.active_response_loss)

            with tf.name_scope("total_loss"):
                self.loss = tf.to_float(self.is_active) * self.active_loss + (1 - tf.to_float(self.is_active)) * self.passive_loss

            # update moving avg/var in batch normalization!
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                with tf.name_scope("optimize"):
                    self.optimize = trainer.minimize(self.loss)

                with tf.name_scope("optimize_fake_response"):
                    self.optimize_fake = trainer.minimize(tf.reduce_sum(self.active_response_loss))

            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self.gradients = tf.gradients(self.loss, local_vars)

            # self.val_pred = tf.reshape(self.fc4, [-1])