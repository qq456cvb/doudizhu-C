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
    def __init__(self, s_dim, trainer, scope, gpus=1):
        with tf.variable_scope(scope):
            self.input_state = []
            self.training = []
            self.input_single = []
            self.input_pair = []
            self.input_triple = []
            self.input_quadric = []
            self.input_single_last = []
            self.input_pair_last = []
            self.input_triple_last = []
            self.input_quadric_last = []
            self.passive_decision_input = []
            self.passive_response_input = []
            self.passive_bomb_input = []
            self.active_decision_input = []
            self.active_response_input = []
            self.minor_response_input = []
            self.seq_length_input = []
            self.value_input = []
            self.mode = []
            self.loss = []
            self.var_norms = []
            self.grad_norms = []
            self.optimize = []
            self.fc_decision_passive_output = []
            self.fc_response_passive_output = []
            self.fc_bomb_passive_output = []
            self.fc_decision_active_output = []
            self.fc_response_active_output = []
            self.fc_sequence_length_output = []
            self.fc_value_output = []
            self.fc_response_minor_output = []
            self.advantages_input = []
            acc_update_ops = 0
            for i in range(gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.variable_scope("inputs"):
                        self.input_state.append(tf.placeholder(tf.float32, [None, s_dim], name="input_state"))
                        self.training.append(tf.placeholder(tf.bool, None, name="training"))
                        self.input_single.append(tf.placeholder(tf.float32, [None, 15], name="input_single"))
                        self.input_pair.append(tf.placeholder(tf.float32, [None, 13], name="input_pair"))
                        self.input_triple.append(tf.placeholder(tf.float32, [None, 13], name="input_triple"))
                        self.input_quadric.append(tf.placeholder(tf.float32, [None, 13], name="input_quadric"))

                    # TODO: test if embedding would help
                    with tf.variable_scope("input_state_embedding"):
                        self.embeddings = slim.fully_connected(
                            inputs=self.input_state[i],
                            num_outputs=512,
                            activation_fn=tf.nn.elu,
                            weights_initializer=tf.contrib.layers.xavier_initializer())

                    with tf.variable_scope("reshaping_for_conv"):
                        self.input_state_conv = tf.reshape(self.embeddings, [-1, 1, 512, 1])
                        self.input_single_conv = tf.reshape(self.input_single[i], [-1, 1, 15, 1])
                        self.input_pair_conv = tf.reshape(self.input_pair[i], [-1, 1, 13, 1])
                        self.input_triple_conv = tf.reshape(self.input_triple[i], [-1, 1, 13, 1])
                        self.input_quadric_conv = tf.reshape(self.input_quadric[i], [-1, 1, 13, 1])

                    # convolution for legacy state
                    with tf.variable_scope("conv_legacy_state"):
                        self.state_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_state_conv,
                                                                 num_outputs=16,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.state_bn1a_branch1a = tf.layers.batch_normalization(self.state_conv1a_branch1a,
                                                                                 training=self.training[i])
                        self.state_nonlinear1a_branch1a = tf.nn.relu(self.state_bn1a_branch1a)

                        self.state_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.state_nonlinear1a_branch1a,
                                                                 num_outputs=16,
                                                                 kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                        self.state_bn1a_branch1b = tf.layers.batch_normalization(self.state_conv1a_branch1b,
                                                                                 training=self.training[i])
                        self.state_nonlinear1a_branch1b = tf.nn.relu(self.state_bn1a_branch1b)

                        self.state_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.state_nonlinear1a_branch1b,
                                                                 num_outputs=64,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.state_bn1a_branch1c = tf.layers.batch_normalization(self.state_conv1a_branch1c,
                                                                                 training=self.training[i])

                        ######

                        self.state_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_state_conv,
                                                                num_outputs=64,
                                                                kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.state_bn1a_branch2 = tf.layers.batch_normalization(self.state_conv1a_branch2,
                                                                                training=self.training[i])

                        self.state1a = self.state_bn1a_branch1c + self.state_bn1a_branch2
                        self.state_output = slim.flatten(tf.nn.relu(self.state1a))

                    # convolution for single
                    with tf.variable_scope("conv_single"):
                        self.single_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_single_conv,
                                                                  num_outputs=16,
                                                                  kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.single_bn1a_branch1a = tf.layers.batch_normalization(self.single_conv1a_branch1a,
                                                                                  training=self.training[i])
                        self.single_nonlinear1a_branch1a = tf.nn.relu(self.single_bn1a_branch1a)

                        self.single_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1a,
                                                                  num_outputs=16,
                                                                  kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                        self.single_bn1a_branch1b = tf.layers.batch_normalization(self.single_conv1a_branch1b,
                                                                                  training=self.training[i])
                        self.single_nonlinear1a_branch1b = tf.nn.relu(self.single_bn1a_branch1b)

                        self.single_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1b,
                                                                  num_outputs=64,
                                                                  kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.single_bn1a_branch1c = tf.layers.batch_normalization(self.single_conv1a_branch1c,
                                                                                  training=self.training[i])

                        ######

                        self.single_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_single_conv,
                                                                 num_outputs=64,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.single_bn1a_branch2 = tf.layers.batch_normalization(self.single_conv1a_branch2,
                                                                                 training=self.training[i])

                        self.single1a = self.single_bn1a_branch1c + self.single_bn1a_branch2
                        self.single_output = slim.flatten(tf.nn.relu(self.single1a))

                    # convolution for pair
                    with tf.variable_scope("conv_pair"):
                        self.pair_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=16,
                                                                kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.pair_bn1a_branch1a = tf.layers.batch_normalization(self.pair_conv1a_branch1a,
                                                                                training=self.training[i])
                        self.pair_nonlinear1a_branch1a = tf.nn.relu(self.pair_bn1a_branch1a)

                        self.pair_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1a,
                                                                num_outputs=16,
                                                                kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                        self.pair_bn1a_branch1b = tf.layers.batch_normalization(self.pair_conv1a_branch1b,
                                                                                training=self.training[i])
                        self.pair_nonlinear1a_branch1b = tf.nn.relu(self.pair_bn1a_branch1b)

                        self.pair_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1b,
                                                                num_outputs=64,
                                                                kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.pair_bn1a_branch1c = tf.layers.batch_normalization(self.pair_conv1a_branch1c,
                                                                                training=self.training[i])

                        ######

                        self.pair_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=64,
                                                               kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.pair_bn1a_branch2 = tf.layers.batch_normalization(self.pair_conv1a_branch2, training=self.training[i])

                        self.pair1a = self.pair_bn1a_branch1c + self.pair_bn1a_branch2
                        self.pair_output = slim.flatten(tf.nn.relu(self.pair1a))

                    # convolution for triple
                    with tf.variable_scope("conv_triple"):
                        self.triple_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv,
                                                                  num_outputs=16,
                                                                  kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.triple_bn1a_branch1a = tf.layers.batch_normalization(self.triple_conv1a_branch1a,
                                                                                  training=self.training[i])
                        self.triple_nonlinear1a_branch1a = tf.nn.relu(self.triple_bn1a_branch1a)

                        self.triple_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1a,
                                                                  num_outputs=16,
                                                                  kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                        self.triple_bn1a_branch1b = tf.layers.batch_normalization(self.triple_conv1a_branch1b,
                                                                                  training=self.training[i])
                        self.triple_nonlinear1a_branch1b = tf.nn.relu(self.triple_bn1a_branch1b)

                        self.triple_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1b,
                                                                  num_outputs=64,
                                                                  kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.triple_bn1a_branch1c = tf.layers.batch_normalization(self.triple_conv1a_branch1c,
                                                                                  training=self.training[i])

                        ######

                        self.triple_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv,
                                                                 num_outputs=64,
                                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.triple_bn1a_branch2 = tf.layers.batch_normalization(self.triple_conv1a_branch2,
                                                                                 training=self.training[i])

                        self.triple1a = self.triple_bn1a_branch1c + self.triple_bn1a_branch2
                        self.triple_output = slim.flatten(tf.nn.relu(self.triple1a))

                    # convolution for quadric
                    with tf.variable_scope("conv_quadric"):
                        self.quadric_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv,
                                                                   num_outputs=16,
                                                                   kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.quadric_bn1a_branch1a = tf.layers.batch_normalization(self.quadric_conv1a_branch1a,
                                                                                   training=self.training[i])
                        self.quadric_nonlinear1a_branch1a = tf.nn.relu(self.quadric_bn1a_branch1a)

                        self.quadric_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1a,
                                                                   num_outputs=16,
                                                                   kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                        self.quadric_bn1a_branch1b = tf.layers.batch_normalization(self.quadric_conv1a_branch1b,
                                                                                   training=self.training[i])
                        self.quadric_nonlinear1a_branch1b = tf.nn.relu(self.quadric_bn1a_branch1b)

                        self.quadric_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1b,
                                                                   num_outputs=64,
                                                                   kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.quadric_bn1a_branch1c = tf.layers.batch_normalization(self.quadric_conv1a_branch1c,
                                                                                   training=self.training[i])

                        ######

                        self.quadric_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv,
                                                                  num_outputs=64,
                                                                  kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.quadric_bn1a_branch2 = tf.layers.batch_normalization(self.quadric_conv1a_branch2,
                                                                                  training=self.training[i])

                        self.quadric1a = self.quadric_bn1a_branch1c + self.quadric_bn1a_branch2
                        self.quadric_output = slim.flatten(tf.nn.relu(self.quadric1a))

                    # 3 + 1 convolution
                    with tf.variable_scope("conv_3plus1"):
                        tiled_triple = tf.tile(tf.expand_dims(self.input_triple[i], 1), [1, 15, 1])
                        tiled_single = tf.tile(tf.expand_dims(self.input_single[i], 2), [1, 1, 13])
                        self.input_triple_single_conv = tf.to_float(
                            tf.expand_dims(tf.bitwise.bitwise_and(tf.to_int32(tiled_triple),
                                                                  tf.to_int32(tiled_single)), -1))

                        self.triple_single_conv1a_branch1a = slim.conv2d(activation_fn=None,
                                                                         inputs=self.input_triple_single_conv, num_outputs=16,
                                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.triple_single_bn1a_branch1a = tf.layers.batch_normalization(self.triple_single_conv1a_branch1a,
                                                                                         training=self.training[i])
                        self.triple_single_nonlinear1a_branch1a = tf.nn.relu(self.triple_single_bn1a_branch1a)

                        self.triple_single_conv1a_branch1b = slim.conv2d(activation_fn=None,
                                                                         inputs=self.triple_single_nonlinear1a_branch1a,
                                                                         num_outputs=16,
                                                                         kernel_size=[3, 3], stride=[1, 1], padding='SAME')
                        self.triple_single_bn1a_branch1b = tf.layers.batch_normalization(self.triple_single_conv1a_branch1b,
                                                                                         training=self.training[i])
                        self.triple_single_nonlinear1a_branch1b = tf.nn.relu(self.triple_single_bn1a_branch1b)

                        self.triple_single_conv1a_branch1c = slim.conv2d(activation_fn=None,
                                                                         inputs=self.triple_single_nonlinear1a_branch1b,
                                                                         num_outputs=64,
                                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.triple_single_bn1a_branch1c = tf.layers.batch_normalization(self.triple_single_conv1a_branch1c,
                                                                                         training=self.training[i])

                        ######

                        self.triple_single_conv1a_branch2 = slim.conv2d(activation_fn=None,
                                                                        inputs=self.input_triple_single_conv, num_outputs=64,
                                                                        kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.triple_single_bn1a_branch2 = tf.layers.batch_normalization(self.triple_single_conv1a_branch2,
                                                                                        training=self.training[i])

                        self.triple_single1a = self.triple_single_bn1a_branch1c + self.triple_single_bn1a_branch2
                        self.triple_single_output = slim.flatten(tf.nn.relu(self.triple_single1a))

                    # 3 + 2 convolution
                    with tf.variable_scope("conv_3plus2"):
                        tiled_triple = tf.tile(tf.expand_dims(self.input_triple[i], 1), [1, 13, 1])
                        tiled_double = tf.tile(tf.expand_dims(self.input_pair[i], 2), [1, 1, 13])
                        self.input_triple_double_conv = tf.to_float(
                            tf.expand_dims(tf.bitwise.bitwise_and(tf.to_int32(tiled_triple),
                                                                  tf.to_int32(tiled_double)), -1))

                        self.triple_double_conv1a_branch1a = slim.conv2d(activation_fn=None,
                                                                         inputs=self.input_triple_double_conv, num_outputs=16,
                                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.triple_double_bn1a_branch1a = tf.layers.batch_normalization(self.triple_double_conv1a_branch1a,
                                                                                         training=self.training[i])
                        self.triple_double_nonlinear1a_branch1a = tf.nn.relu(self.triple_double_bn1a_branch1a)

                        self.triple_double_conv1a_branch1b = slim.conv2d(activation_fn=None,
                                                                         inputs=self.triple_double_nonlinear1a_branch1a,
                                                                         num_outputs=16,
                                                                         kernel_size=[3, 3], stride=[1, 1], padding='SAME')
                        self.triple_double_bn1a_branch1b = tf.layers.batch_normalization(self.triple_double_conv1a_branch1b,
                                                                                         training=self.training[i])
                        self.triple_double_nonlinear1a_branch1b = tf.nn.relu(self.triple_double_bn1a_branch1b)

                        self.triple_double_conv1a_branch1c = slim.conv2d(activation_fn=None,
                                                                         inputs=self.triple_double_nonlinear1a_branch1b,
                                                                         num_outputs=64,
                                                                         kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.triple_double_bn1a_branch1c = tf.layers.batch_normalization(self.triple_double_conv1a_branch1c,
                                                                                         training=self.training[i])

                        ######

                        self.triple_double_conv1a_branch2 = slim.conv2d(activation_fn=None,
                                                                        inputs=self.input_triple_double_conv, num_outputs=64,
                                                                        kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                        self.triple_double_bn1a_branch2 = tf.layers.batch_normalization(self.triple_double_conv1a_branch2,
                                                                                        training=self.training[i])

                        self.triple_double1a = self.triple_double_bn1a_branch1c + self.triple_double_bn1a_branch2
                        self.triple_double_output = slim.flatten(tf.nn.relu(self.triple_double1a))

                    #################################################

                    with tf.variable_scope("concated"):
                        self.fc_flattened = tf.concat([self.single_output, self.pair_output, self.triple_output,
                                                       self.quadric_output, self.triple_single_output,
                                                       self.triple_double_output, self.state_output], 1)

                    with tf.variable_scope("last_cards"):
                        with tf.variable_scope("inputs"):
                            self.input_single_last.append(tf.placeholder(tf.float32, [None, 15], name="input_single_last"))
                            self.input_pair_last.append(tf.placeholder(tf.float32, [None, 13], name="input_pair_last"))
                            self.input_triple_last.append(tf.placeholder(tf.float32, [None, 13], name="input_triple_last"))
                            self.input_quadric_last.append(tf.placeholder(tf.float32, [None, 13], name="input_quadric_last"))

                        with tf.variable_scope("reshaping_for_conv"):
                            self.input_single_conv = tf.reshape(self.input_single_last[i], [-1, 1, 15, 1])
                            self.input_pair_conv = tf.reshape(self.input_pair_last[i], [-1, 1, 13, 1])
                            self.input_triple_conv = tf.reshape(self.input_triple_last[i], [-1, 1, 13, 1])
                            self.input_quadric_conv = tf.reshape(self.input_quadric_last[i], [-1, 1, 13, 1])

                        with tf.variable_scope("conv_single"):
                            self.single_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_single_conv, num_outputs=16,
                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.single_bn1a_branch1a = tf.layers.batch_normalization(self.single_conv1a_branch1a, training=self.training[i])
                            self.single_nonlinear1a_branch1a = tf.nn.relu(self.single_bn1a_branch1a)

                            self.single_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1a, num_outputs=16,
                                                 kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                            self.single_bn1a_branch1b = tf.layers.batch_normalization(self.single_conv1a_branch1b, training=self.training[i])
                            self.single_nonlinear1a_branch1b = tf.nn.relu(self.single_bn1a_branch1b)

                            self.single_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.single_nonlinear1a_branch1b, num_outputs=64,
                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.single_bn1a_branch1c = tf.layers.batch_normalization(self.single_conv1a_branch1c, training=self.training[i])

                            ######

                            self.single_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_single_conv, num_outputs=64,
                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.single_bn1a_branch2 = tf.layers.batch_normalization(self.single_conv1a_branch2, training=self.training[i])

                            self.single1a = self.single_bn1a_branch1c + self.single_bn1a_branch2
                            self.single_output = slim.flatten(tf.nn.relu(self.single1a))

                        # convolution for pair
                        with tf.variable_scope("conv_pair"):
                            self.pair_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=16,
                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.pair_bn1a_branch1a = tf.layers.batch_normalization(self.pair_conv1a_branch1a, training=self.training[i])
                            self.pair_nonlinear1a_branch1a = tf.nn.relu(self.pair_bn1a_branch1a)

                            self.pair_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1a, num_outputs=16,
                                                 kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                            self.pair_bn1a_branch1b = tf.layers.batch_normalization(self.pair_conv1a_branch1b, training=self.training[i])
                            self.pair_nonlinear1a_branch1b = tf.nn.relu(self.pair_bn1a_branch1b)

                            self.pair_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.pair_nonlinear1a_branch1b, num_outputs=64,
                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.pair_bn1a_branch1c = tf.layers.batch_normalization(self.pair_conv1a_branch1c, training=self.training[i])

                            ######

                            self.pair_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_pair_conv, num_outputs=64,
                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.pair_bn1a_branch2 = tf.layers.batch_normalization(self.pair_conv1a_branch2, training=self.training[i])

                            self.pair1a = self.pair_bn1a_branch1c + self.pair_bn1a_branch2
                            self.pair_output = slim.flatten(tf.nn.relu(self.pair1a))

                        # convolution for triple
                        with tf.variable_scope("conv_triple"):
                            self.triple_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv, num_outputs=16,
                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.triple_bn1a_branch1a = tf.layers.batch_normalization(self.triple_conv1a_branch1a, training=self.training[i])
                            self.triple_nonlinear1a_branch1a = tf.nn.relu(self.triple_bn1a_branch1a)

                            self.triple_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1a, num_outputs=16,
                                                 kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                            self.triple_bn1a_branch1b = tf.layers.batch_normalization(self.triple_conv1a_branch1b, training=self.training[i])
                            self.triple_nonlinear1a_branch1b = tf.nn.relu(self.triple_bn1a_branch1b)

                            self.triple_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.triple_nonlinear1a_branch1b, num_outputs=64,
                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.triple_bn1a_branch1c = tf.layers.batch_normalization(self.triple_conv1a_branch1c, training=self.training[i])

                            ######

                            self.triple_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_triple_conv, num_outputs=64,
                                                     kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.triple_bn1a_branch2 = tf.layers.batch_normalization(self.triple_conv1a_branch2, training=self.training[i])

                            self.triple1a = self.triple_bn1a_branch1c + self.triple_bn1a_branch2
                            self.triple_output = slim.flatten(tf.nn.relu(self.triple1a))

                        # convolution for quadric
                        with tf.variable_scope("conv_quadric"):
                            self.quadric_conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv, num_outputs=16,
                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.quadric_bn1a_branch1a = tf.layers.batch_normalization(self.quadric_conv1a_branch1a, training=self.training[i])
                            self.quadric_nonlinear1a_branch1a = tf.nn.relu(self.quadric_bn1a_branch1a)

                            self.quadric_conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1a, num_outputs=16,
                                                 kernel_size=[1, 3], stride=[1, 1], padding='SAME')
                            self.quadric_bn1a_branch1b = tf.layers.batch_normalization(self.quadric_conv1a_branch1b, training=self.training[i])
                            self.quadric_nonlinear1a_branch1b = tf.nn.relu(self.quadric_bn1a_branch1b)

                            self.quadric_conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=self.quadric_nonlinear1a_branch1b, num_outputs=64,
                                             kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.quadric_bn1a_branch1c = tf.layers.batch_normalization(self.quadric_conv1a_branch1c, training=self.training[i])

                            ######

                            self.quadric_conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=self.input_quadric_conv, num_outputs=64,
                                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
                            self.quadric_bn1a_branch2 = tf.layers.batch_normalization(self.quadric_conv1a_branch2, training=self.training[i])

                            self.quadric1a = self.quadric_bn1a_branch1c + self.quadric_bn1a_branch2
                            self.quadric_output = slim.flatten(tf.nn.relu(self.quadric1a))

                        self.fc_flattened_last = tf.concat([self.single_output, self.pair_output, self.triple_output,
                            self.quadric_output], 1)

                        # add attention for last cards
                        self.attention_last_decision = slim.fully_connected(self.fc_flattened_last, 1024, tf.nn.softmax)
                        self.attention_last_response = slim.fully_connected(self.fc_flattened_last, 1024, tf.nn.softmax)

                    # passive decision making  0: pass, 1: bomb, 2: king, 3: normal
                    with tf.variable_scope("passive_decision_making"):
                        self.fc_decision_passive = self.fc_flattened
                        self.fc_decision_passive = tf.multiply(
                            slim.fully_connected(self.fc_decision_passive, 1024, tf.nn.relu),
                            self.attention_last_decision)
                        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                            self.fc_decision_passive = slim.stack(self.fc_decision_passive, slim.fully_connected, [256, 64])
                        self.fc_decision_passive_output.append(slim.fully_connected(self.fc_decision_passive, 4, tf.nn.softmax))

                    # passive response
                    with tf.variable_scope("passive_response"):
                        self.fc_response_passive = self.fc_flattened
                        self.fc_response_passive = tf.multiply(
                            slim.fully_connected(self.fc_response_passive, 1024, tf.nn.relu),
                            self.attention_last_response)
                        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                            self.fc_response_passive = slim.stack(self.fc_response_passive, slim.fully_connected, [256, 64])
                        self.fc_response_passive_output.append(slim.fully_connected(self.fc_response_passive, 15, tf.nn.softmax))

                    # passive bomb response
                    with tf.variable_scope("passive_bomb_reponse"):
                        self.fc_bomb_passive = self.fc_flattened
                        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                            self.fc_bomb_passive = slim.stack(self.fc_bomb_passive, slim.fully_connected, [1024, 256, 64])
                        self.fc_bomb_passive_output.append(slim.fully_connected(self.fc_bomb_passive, 13, tf.nn.softmax))

                    # active decision making  mapped to [action space category - 1]
                    with tf.variable_scope("active_decision_making"):
                        self.fc_decision_active = self.fc_flattened
                        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                            self.fc_decision_active = slim.stack(self.fc_decision_active, slim.fully_connected, [1024, 256, 64])
                        self.fc_decision_active_output.append(slim.fully_connected(self.fc_decision_active, 13, tf.nn.softmax))

                    # active response
                    with tf.variable_scope("active_response"):
                        self.fc_response_active = self.fc_flattened
                        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                            self.fc_response_active = slim.stack(self.fc_response_active, slim.fully_connected, [1024, 256, 64])
                        self.fc_response_active_output.append(slim.fully_connected(self.fc_response_active, 15, tf.nn.softmax))

                    # minor response
                    with tf.variable_scope("minor_response"):
                        self.fc_response_minor = self.fc_flattened
                        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                            self.fc_response_minor = slim.stack(self.fc_response_minor, slim.fully_connected, [1024, 256, 64])
                        self.fc_response_minor_output.append(slim.fully_connected(self.fc_response_active, 15, tf.nn.softmax))

                    # card length output
                    with tf.variable_scope("fc_sequence_length_output"):
                        self.fc_seq_length = self.fc_flattened
                        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                            self.fc_seq_length = slim.stack(self.fc_seq_length, slim.fully_connected, [1024, 256, 64])
                        self.fc_sequence_length_output.append(slim.fully_connected(self.fc_seq_length, 12, tf.nn.softmax))

                    with tf.variable_scope("value_output"):
                        self.fc_value = self.fc_flattened
                        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                            self.fc_value = slim.stack(self.fc_value, slim.fully_connected, [1024, 256])
                        with slim.arg_scope([slim.fully_connected], activation_fn=None):
                            self.fc_value = slim.stack(self.fc_value, slim.fully_connected, [64, 16])
                        self.fc_value_output.append(tf.squeeze(slim.fully_connected(self.fc_value, 1, None), [1]))

                    with tf.variable_scope("advantages"):
                        self.advantages_input.append(tf.placeholder(tf.float32, [None], name='advantages_in'))

                    # passive mode
                    with tf.variable_scope("passive_mode_loss"):
                        self.passive_decision_input.append(tf.placeholder(tf.int32, [None], name='passive_decision_in'))
                        self.passive_decision_target = tf.one_hot(self.passive_decision_input[i], 4)
                        self.passive_decision_loss = -tf.reduce_sum(self.passive_decision_target * tf.log(
                            tf.clip_by_value(self.fc_decision_passive_output[i], 1e-10, 1 - (1e-10))), 1) * self.advantages_input[i]

                        self.passive_response_input.append(tf.placeholder(tf.int32, [None], name='passive_response_in'))
                        self.passive_response_target = tf.one_hot(self.passive_response_input[i], 15)
                        self.passive_response_loss = -tf.reduce_sum(self.passive_response_target * tf.log(
                            tf.clip_by_value(self.fc_response_passive_output[i], 1e-10, 1 - (1e-10))), 1) * self.advantages_input[i]

                        self.passive_bomb_input.append(tf.placeholder(tf.int32, [None], name='passive_bomb_in'))
                        self.passive_bomb_target = tf.one_hot(self.passive_bomb_input[i], 13)
                        self.passive_bomb_loss = -tf.reduce_sum(self.passive_bomb_target * tf.log(
                            tf.clip_by_value(self.fc_bomb_passive_output[i], 1e-10, 1 - (1e-10))), 1) * self.advantages_input[i]

                    # active mode
                    with tf.variable_scope("active_mode_loss"):
                        self.active_decision_input.append(tf.placeholder(tf.int32, [None], name='active_decision_in'))
                        self.active_decision_target = tf.one_hot(self.active_decision_input[i], 13)
                        self.active_decision_loss = -tf.reduce_sum(self.active_decision_target * tf.log(
                            tf.clip_by_value(self.fc_decision_active_output[i], 1e-10, 1 - (1e-10))), 1) * self.advantages_input[i]

                        self.active_response_input.append(tf.placeholder(tf.int32, [None], name='active_response_in'))
                        self.active_response_target = tf.one_hot(self.active_response_input[i], 15)
                        self.active_response_loss = -tf.reduce_sum(self.active_response_target * tf.log(
                            tf.clip_by_value(self.fc_response_active_output[i], 1e-10, 1 - (1e-10))), 1) * self.advantages_input[i]

                        self.seq_length_input.append(tf.placeholder(tf.int32, [None], name='sequence_length_in'))
                        self.seq_length_target = tf.one_hot(self.seq_length_input[i], 12)
                        self.seq_length_loss = -tf.reduce_sum(self.seq_length_target * tf.log(
                                tf.clip_by_value(self.fc_sequence_length_output[i], 1e-10, 1 - (1e-10))), 1) * self.advantages_input[i]

                    with tf.variable_scope("minor_mode_loss"):
                        self.minor_response_input.append(tf.placeholder(tf.int32, [None], name='minor_response_in'))
                        self.minor_response_target = tf.one_hot(self.minor_response_input[i], 15)
                        self.minor_response_loss = -tf.reduce_sum(self.minor_response_target * tf.log(
                            tf.clip_by_value(self.fc_response_minor_output[i], 1e-10, 1 - (1e-10))), 1) * self.advantages_input[i]

                    with tf.variable_scope("value_loss"):
                        self.value_input.append(tf.placeholder(tf.float32, [None], name='value_in'))
                        self.value_loss = tf.reduce_sum(tf.square(self.value_input[i] - self.fc_value_output[i]))

                    # map mode to loss
                    # 0: passive decision + value loss
                    # 1: passive decision + passive bomb + value loss
                    # 2: passive decision + passive response + value loss
                    # 3: active decision + active response + value loss
                    # 4: active decision + active response + sequence length + value loss
                    # 5: minor response (do not add value loss since for minor cards, state won't change)
                    self.mode.append(tf.placeholder(tf.int32, [None], name='mode'))
                    self.losses = tf.stack([self.passive_decision_loss + self.value_loss,
                                            self.passive_decision_loss + self.passive_bomb_loss + self.value_loss,
                                            self.passive_decision_loss + self.passive_response_loss + self.value_loss,
                                            self.active_decision_loss + self.active_response_loss + self.value_loss,
                                            self.active_decision_loss + self.active_response_loss + self.seq_length_loss + self.value_loss,
                                            self.minor_response_loss])
                    self.mask = tf.transpose(tf.one_hot(self.mode[i], depth=6, dtype=tf.bool, on_value=True, off_value=False))
                    self.loss.append(tf.reduce_sum(tf.boolean_mask(self.losses, self.mask)))

                    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                    self.var_norms.append(tf.global_norm(local_vars))
                    gradients = tf.gradients(self.loss[i], local_vars)
                    cliped_grads, grad_norms = tf.clip_by_global_norm(gradients, 4000.0)
                    self.grad_norms.append(grad_norms)

                    # update moving avg/var in batch normalization!
                    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
                    curr_extra_update_ops = extra_update_ops[acc_update_ops:]
                    acc_update_ops = len(extra_update_ops)
                    # self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
                    with tf.control_dependencies(curr_extra_update_ops):
                        with tf.variable_scope("optimize"):
                            self.optimize.append(trainer.apply_gradients(zip(cliped_grads, local_vars)))

                    # share variables across multiple GPUs
                    tf.get_variable_scope().reuse_variables()



