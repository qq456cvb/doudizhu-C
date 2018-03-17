import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn


def identity_block(input, first_channel, last_channel, kernel_size, training):
    conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=input, num_outputs=first_channel,
                                            kernel_size=[1, 1], stride=[1, 1], padding='SAME')
    bn1a_branch1a = tf.layers.batch_normalization(conv1a_branch1a, training=training)
    nonlinear1a_branch1a = tf.nn.relu(bn1a_branch1a)

    conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=nonlinear1a_branch1a, num_outputs=first_channel,
                                            kernel_size=[1, kernel_size], stride=[1, 1], padding='SAME')
    bn1a_branch1b = tf.layers.batch_normalization(conv1a_branch1b, training=training)
    nonlinear1a_branch1b = tf.nn.relu(bn1a_branch1b)

    conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=nonlinear1a_branch1b, num_outputs=last_channel,
                                            kernel_size=[1, 1], stride=[1, 1], padding='SAME')
    bn1a_branch1c = tf.layers.batch_normalization(conv1a_branch1c, training=training)

    return tf.nn.relu(bn1a_branch1c)


def upsample_block(input, first_channel, last_channel, kernel_size, training):
    conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=input, num_outputs=first_channel,
                                  kernel_size=[1, 1], stride=[1, 1], padding='SAME')
    bn1a_branch1a = tf.layers.batch_normalization(conv1a_branch1a, training=training)
    nonlinear1a_branch1a = tf.nn.relu(bn1a_branch1a)

    conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=nonlinear1a_branch1a, num_outputs=first_channel,
                                  kernel_size=[1, kernel_size], stride=[1, 1], padding='SAME')
    bn1a_branch1b = tf.layers.batch_normalization(conv1a_branch1b, training=training)
    nonlinear1a_branch1b = tf.nn.relu(bn1a_branch1b)

    conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=nonlinear1a_branch1b, num_outputs=last_channel,
                                  kernel_size=[1, 1], stride=[1, 1], padding='SAME')
    bn1a_branch1c = tf.layers.batch_normalization(conv1a_branch1c, training=training)

    ######

    conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=input, num_outputs=last_channel,
                                 kernel_size=[1, 1], stride=[1, 1], padding='SAME')
    bn1a_branch2 = tf.layers.batch_normalization(conv1a_branch2, training=training)

    plus = bn1a_branch1c + bn1a_branch2

    return tf.nn.relu(slim.avg_pool2d(plus, kernel_size=[1, 3]))


def conv_block(input, conv_dim, input_dim, res_params, training, scope):
    conv_out = []
    with tf.variable_scope(scope):
        input_conv = tf.reshape(input, [-1, 1, input_dim, 1])
        single_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=conv_dim,
                                  kernel_size=[1, 1], stride=[1, 4], padding='VALID')

        pair_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=conv_dim,
                                kernel_size=[1, 2], stride=[1, 4], padding='VALID')

        triple_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=conv_dim,
                                  kernel_size=[1, 3], stride=[1, 4], padding='VALID')

        quadric_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=conv_dim,
                                   kernel_size=[1, 4], stride=[1, 4], padding='VALID')

        conv_list = [single_conv, pair_conv, triple_conv, quadric_conv]

        for conv in conv_list:
            for param in res_params:
                if param[-1] == 'identity':
                    conv = identity_block(conv, param[0], param[1], param[2], training)
                elif param[-1] == 'upsampling':
                    conv = upsample_block(conv, param[0], param[1], param[2], training)
                else:
                    raise Exception('unsupported layer type')
            conv_out.append(slim.flatten(conv))

    flattened = tf.concat(conv_out, 1)
    return flattened


class CardNetwork:
        def __init__(self, trainer, scope):
            with tf.device('/gpu:0'):
                with tf.variable_scope(scope):
                    with tf.variable_scope('inputs'):
                        self.input_state = tf.placeholder(tf.float32, [None, 6 * 60], name="input")
                        self.training = tf.placeholder(tf.bool, None, name='training')
                        self.last_outcards = tf.placeholder(tf.float32, [None, 60], name='last_cards')
                        self.minor_type = tf.placeholder(tf.int64, [None], name='minor_type')

                    with slim.arg_scope([slim.fully_connected, slim.conv2d], weights_regularizer=slim.l2_regularizer(1e-3), weights_initializer=tf.truncated_normal_initializer(stddev=0.1)):
                        with tf.variable_scope('branch_main'):
                            flattened = conv_block(self.input_state, 32, 6 * 60, [[16, 64, 3, 'identity'], [16, 64, 3, 'identity'], [32, 128, 3, 'upsampling']], self.training, 'branch_main')

                        with tf.variable_scope('branch_passive'):
                            flattened_last = conv_block(self.last_outcards, 32, 60, [[16, 64, 3, 'identity'], [16, 64, 3, 'identity'], [32, 128, 3, 'upsampling']], self.training, 'last_cards')

                            self.hidden_size = 256
                            self.lstm_passive = rnn.BasicLSTMCell(num_units=self.hidden_size, state_is_tuple=True)

                            # no regularization for LSTM yet
                            with tf.variable_scope('decision'):
                                attention_decision = slim.fully_connected(inputs=flattened_last, num_outputs=256,
                                                                          activation_fn=tf.nn.sigmoid)

                                fc_passive_decision = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                           activation_fn=tf.nn.relu)
                                fc_passive_decision = fc_passive_decision * attention_decision
                                lstm_passive_decision_output, hidden_decision_output = tf.nn.dynamic_rnn(self.lstm_passive,
                                                                                        tf.expand_dims(fc_passive_decision, 1),
                                                                                        initial_state=self.lstm_passive.zero_state(tf.shape(fc_passive_decision)[0], dtype=tf.float32),
                                                                                        sequence_length=tf.ones([tf.shape(self.input_state)[0]]))
                                fc_passive_decision = slim.fully_connected(inputs=tf.squeeze(lstm_passive_decision_output, axis=[1]), num_outputs=64, activation_fn=tf.nn.relu)
                                self.fc_passive_decision_output = slim.fully_connected(inputs=fc_passive_decision, num_outputs=4, activation_fn=tf.nn.softmax)

                            # bomb and response do not depend on each other
                            with tf.variable_scope('bomb'):
                                fc_passive_bomb = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                           activation_fn=tf.nn.relu)
                                fc_passive_bomb = slim.fully_connected(inputs=fc_passive_bomb, num_outputs=64, activation_fn=tf.nn.relu)
                                self.fc_passive_bomb_output = slim.fully_connected(inputs=fc_passive_bomb, num_outputs=13, activation_fn=tf.nn.softmax)

                            with tf.variable_scope('response'):
                                attention_response = slim.fully_connected(inputs=flattened_last, num_outputs=256,
                                                                          activation_fn=tf.nn.sigmoid)

                                fc_passive_response = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                           activation_fn=tf.nn.relu)
                                fc_passive_response = fc_passive_response * attention_response
                                lstm_passive_response_output, _ = tf.nn.dynamic_rnn(self.lstm_passive,
                                                                        tf.expand_dims(fc_passive_response, 1),
                                                                        initial_state=hidden_decision_output,
                                                                        sequence_length=tf.ones([tf.shape(self.input_state)[0]]))
                                fc_passive_response = slim.fully_connected(inputs=tf.squeeze(lstm_passive_response_output, axis=[1]), num_outputs=64, activation_fn=tf.nn.relu)
                                self.fc_passive_response_output = slim.fully_connected(inputs=fc_passive_response, num_outputs=15, activation_fn=tf.nn.softmax)

                        with tf.variable_scope('branch_active'):
                            self.lstm_active = rnn.BasicLSTMCell(num_units=self.hidden_size, state_is_tuple=True)

                            with tf.variable_scope('decision'):
                                fc_active_decision = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                 activation_fn=tf.nn.relu)
                                lstm_active_decision_output, hidden_active_output = tf.nn.dynamic_rnn(self.lstm_active,
                                                                                                 tf.expand_dims(fc_active_decision, 1),
                                                                                                 initial_state=self.lstm_passive.zero_state(tf.shape(fc_active_decision)[0], dtype=tf.float32),
                                                                                                 sequence_length=tf.ones([tf.shape(self.input_state)[0]]))
                                fc_active_decision = slim.fully_connected(inputs=tf.squeeze(lstm_active_decision_output, axis=[1]), num_outputs=64, activation_fn=tf.nn.relu)
                                self.fc_active_decision_output = slim.fully_connected(inputs=fc_active_decision, num_outputs=13, activation_fn=tf.nn.softmax)

                            with tf.variable_scope('response'):
                                fc_active_response = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                 activation_fn=tf.nn.relu)
                                lstm_active_response_output, hidden_active_output = tf.nn.dynamic_rnn(self.lstm_active,
                                                                                                 tf.expand_dims(fc_active_response, 1),
                                                                                                 initial_state=hidden_active_output,
                                                                                                 sequence_length=tf.ones([tf.shape(self.input_state)[0]]))
                                fc_active_response = slim.fully_connected(inputs=tf.squeeze(lstm_active_response_output, axis=[1]), num_outputs=64, activation_fn=tf.nn.relu)
                                self.fc_active_response_output = slim.fully_connected(inputs=fc_active_response, num_outputs=15, activation_fn=tf.nn.softmax)

                            with tf.variable_scope('seq_length'):
                                fc_active_seq = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                 activation_fn=tf.nn.relu)
                                lstm_active_seq_output, _ = tf.nn.dynamic_rnn(self.lstm_active,
                                                                              tf.expand_dims(fc_active_seq, 1),
                                                                              initial_state=hidden_active_output,
                                                                              sequence_length=tf.ones([tf.shape(self.input_state)[0]]))
                                fc_active_seq = slim.fully_connected(inputs=tf.squeeze(lstm_active_seq_output, axis=[1]), num_outputs=64, activation_fn=tf.nn.relu)
                                self.fc_active_seq_output = slim.fully_connected(inputs=fc_active_seq, num_outputs=12, activation_fn=tf.nn.softmax)

                        with tf.variable_scope('branch_minor'):
                            fc_minor = slim.fully_connected(inputs=flattened, num_outputs=256,
                                                                 activation_fn=tf.nn.relu)
                            minor_type_embedding = slim.fully_connected(inputs=tf.one_hot(self.minor_type, 2), num_outputs=256, activation_fn=tf.nn.sigmoid)
                            fc_minor = fc_minor * minor_type_embedding

                            fc_minor = slim.fully_connected(inputs=fc_minor, num_outputs=64, activation_fn=tf.nn.relu)
                            self.fc_minor_response_output = slim.fully_connected(inputs=fc_minor, num_outputs=15, activation_fn=tf.nn.softmax)

                        # passive mode
                        with tf.variable_scope("passive_mode_loss"):
                            self.passive_decision_input = tf.placeholder(tf.int64, [None],
                                                                         name='passive_decision_in')
                            self.passive_decision_target = tf.one_hot(self.passive_decision_input, 4)
                            self.passive_decision_loss = -tf.reduce_sum(self.passive_decision_target * tf.log(
                                tf.clip_by_value(self.fc_passive_decision_output, 1e-10, 1 - (1e-10))))

                            self.passive_response_input = tf.placeholder(tf.int64, [None],
                                                                         name='passive_response_in')
                            self.passive_response_target = tf.one_hot(self.passive_response_input, 15)
                            self.passive_response_loss = -tf.reduce_sum(self.passive_response_target * tf.log(
                                tf.clip_by_value(self.fc_passive_response_output, 1e-10, 1 - (1e-10))))

                            self.passive_bomb_input = tf.placeholder(tf.int64, [None], name='passive_bomb_in')
                            self.passive_bomb_target = tf.one_hot(self.passive_bomb_input, 13)
                            self.passive_bomb_loss = -tf.reduce_sum(self.passive_bomb_target * tf.log(
                                tf.clip_by_value(self.fc_passive_bomb_output, 1e-10, 1 - (1e-10))))

                        # active mode
                        with tf.variable_scope("active_mode_loss"):
                            self.active_decision_input = tf.placeholder(tf.int64, [None], name='active_decision_in')
                            self.active_decision_target = tf.one_hot(self.active_decision_input, 13)
                            self.active_decision_loss = -tf.reduce_sum(self.active_decision_target * tf.log(
                                tf.clip_by_value(self.fc_active_decision_output, 1e-10, 1 - (1e-10))))

                            self.active_response_input = tf.placeholder(tf.int64, [None], name='active_response_in')
                            self.active_response_target = tf.one_hot(self.active_response_input, 15)
                            self.active_response_loss = -tf.reduce_sum(self.active_response_target * tf.log(
                                tf.clip_by_value(self.fc_active_response_output, 1e-10, 1 - (1e-10))))

                            self.seq_length_input = tf.placeholder(tf.int64, [None], name='sequence_length_in')
                            self.seq_length_target = tf.one_hot(self.seq_length_input, 12)
                            self.seq_length_loss = -tf.reduce_sum(self.seq_length_target * tf.log(
                                tf.clip_by_value(self.fc_active_seq_output, 1e-10, 1 - (1e-10))))

                        with tf.variable_scope("minor_mode_loss"):
                            self.minor_response_input = tf.placeholder(tf.int64, [None], name='minor_response_in')
                            self.minor_response_target = tf.one_hot(self.minor_response_input, 15)
                            self.minor_response_loss = -tf.reduce_sum(self.minor_response_target * tf.log(
                                tf.clip_by_value(self.fc_minor_response_output, 1e-10, 1 - (1e-10))))

                    l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
                    l2_main_loss = [l for l in l2_loss if 'branch_main' in l.name]
                    l2_passive_fc_loss = [l for l in l2_loss if 'branch_passive' in l.name and 'decision' not in l.name and 'bomb' not in l.name and 'response'not in l.name] \
                        + [1e-3 * tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('network/branch_passive/decision/rnn/basic_lstm_cell/kernel:0'))]
                    l2_active_fc_loss = [l for l in l2_loss if 'branch_active' in l.name and 'decision' not in l.name and 'response' not in l.name and 'seq_length'not in l.name] \
                        + [1e-3 * tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name('network/branch_active/decision/rnn/basic_lstm_cell/kernel:0'))]

                    print('l2 loss', len(l2_loss))
                    print('l2 main loss', len(l2_main_loss))
                    print('l2 passive fc loss', len(l2_passive_fc_loss))
                    print('l2 active fc loss', len(l2_active_fc_loss))

                    name_scopes = ['branch_passive/decision', 'branch_passive/bomb', 'branch_passive/response',
                                   'branch_active/decision', 'branch_active/response', 'branch_active/seq_length', 'branch_minor']

                    self.losses = [self.passive_decision_loss, self.passive_bomb_loss, self.passive_response_loss,
                                   self.active_decision_loss, self.active_response_loss, self.seq_length_loss, self.minor_response_loss]
                    self.optimize = []

                    # update ops for batch normalization
                    main_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope + 'branch_main')
                    passive_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope + 'branch_main')
                    self.gradient_norms = []
                    for i, name in enumerate(name_scopes):
                        l2_branch_loss = l2_main_loss.copy()
                        if 'passive' in name:
                            if 'bomb' in name:
                                l2_branch_loss += [l for l in l2_loss if name in l.name]
                            else:
                                l2_branch_loss += l2_passive_fc_loss + [l for l in l2_loss if name in l.name]
                        else:
                            if 'minor' in name:
                                # do not include lstm regularization in minor loss
                                l2_branch_loss += l2_active_fc_loss[:-1] + [l for l in l2_loss if name in l.name]
                            else:
                                l2_branch_loss += l2_active_fc_loss + [l for l in l2_loss if name in l.name]

                        print('l2 branch loss', len(l2_branch_loss))

                        gvs = trainer.compute_gradients(self.losses[i] + tf.add_n(l2_branch_loss))
                        gvs = [(gv[0], gv[1]) for gv in gvs if gv[0] is not None]
                        g, v = zip(*gvs)
                        g, global_norm = tf.clip_by_global_norm(g, 5.0)
                        self.gradient_norms.append(global_norm)
                        if 'passive' in name:
                            with tf.control_dependencies(main_update_ops + passive_update_ops):
                                update = trainer.apply_gradients(zip(g, v))
                        else:
                            with tf.control_dependencies(main_update_ops):
                                update = trainer.apply_gradients(zip(g, v))
                        self.optimize.append(update)
                    self.weight_norm = tf.norm([v for v in tf.trainable_variables(scope=scope) if v.name.endswith('weights:0')][0])
                    self.lstm_norm = tf.global_norm([v for v in tf.trainable_variables(scope=scope) if v.name.endswith('kernel:0')])

