import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn


class CardNetwork:
    def __init__(self, trainer, scope):
        with tf.device('/gpu:0'):
            with tf.variable_scope(scope):
                with tf.variable_scope('inputs'):
                    self.mask_in = tf.placeholder(tf.float32, [None, 15 + 13 + 13 + 13], 'mask_in')
                    self.last_masks_in = tf.placeholder(tf.float32, [None, 15 + 13 + 13 + 13], 'last_in')
                    self.minor_type = tf.placeholder(tf.int32, [None], 'minor_type_in')

                n_layers = 32
                n_layers_branch = 16
                with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(1e-3)):
                    x = self.mask_in
                    with tf.variable_scope('branch_main'):
                        x = slim.stack(x, slim.fully_connected, [256] * n_layers)

                    with tf.variable_scope('branch_passive'):
                        x_last = self.last_masks_in
                        x_last = slim.stack(x_last, slim.fully_connected, [256] * n_layers)

                        # TODO: may different attention for decision and response?
                        x_passive = x * tf.nn.sigmoid(x_last)
                        with tf.variable_scope('decision'):
                            x_decision = x_passive
                            x_decision = slim.stack(x_decision, slim.fully_connected, [256] * n_layers_branch)
                            passive_decision_logits = slim.fully_connected(x_decision, 4, None)
                            self.passive_decision_probs = tf.nn.softmax(passive_decision_logits)

                        with tf.variable_scope('bomb'):
                            x_bomb = x
                            x_bomb = slim.stack(x_bomb, slim.fully_connected, [256] * n_layers_branch)
                            passive_bomb_logits = slim.fully_connected(x_bomb, 13, None)
                            self.passive_bomb_probs = tf.nn.softmax(passive_bomb_logits)

                        with tf.variable_scope('response'):
                            x_response = x_passive
                            x_response = slim.stack(x_response, slim.fully_connected, [256] * n_layers_branch)
                            passive_response_logits = slim.fully_connected(x_response, 15, None)
                            self.passive_response_probs = tf.nn.softmax(passive_response_logits)

                    with tf.variable_scope('branch_active'):
                        x_active = x
                        with tf.variable_scope('decision'):
                            x_decision = x_active
                            x_decision = slim.stack(x_decision, slim.fully_connected, [256] * n_layers_branch)
                            active_decision_logits = slim.fully_connected(x_decision, 13, None)
                            self.active_decision_probs = tf.nn.softmax(active_decision_logits)

                        with tf.variable_scope('response'):
                            decision_onehot = tf.one_hot(tf.argmax(active_decision_logits, axis=-1), 13)
                            decision_onehot = tf.stop_gradient(decision_onehot)

                            # TODO: add or multiply?
                            # TODO: embedding layer shallow or deep?
                            x_response = tf.concat([x_active, slim.stack(decision_onehot, slim.fully_connected, [256] * 4)], axis=0)
                            x_response = slim.stack(x_response, slim.fully_connected, [256] * n_layers_branch)
                            active_response_logits = slim.fully_connected(x_response, 15, None)
                            self.active_response_probs = tf.nn.softmax(active_response_logits)

                        with tf.variable_scope('seq_length'):
                            response_onehot = tf.one_hot(tf.argmax(active_response_logits, axis=-1), 15)
                            response_onehot = tf.stop_gradient(response_onehot)

                            x_seq = tf.concat([x_active, slim.stack(decision_onehot, slim.fully_connected, [256] * 4), slim.stack(response_onehot, slim.fully_connected, [256] * 4)], axis=0)
                            x_seq = slim.stack(x_seq, slim.fully_connected, [256] * n_layers_branch)
                            seq_length_logits = slim.fully_connected(x_seq, 12, None)
                            self.seq_length_probs = tf.nn.softmax(seq_length_logits)

                    with tf.variable_scope('branch_minor'):
                        x_minor = x * slim.fully_connected(tf.one_hot(self.minor_type, 2), 256, tf.nn.sigmoid)
                        x_minor = slim.stack(x_minor, slim.fully_connected, [256] * n_layers_branch)
                        minor_logits = slim.fully_connected(x_minor, 15, None)
                        self.minor_probs = tf.nn.softmax(minor_logits)

                # passive mode
                with tf.variable_scope("passive_mode_loss"):
                    self.passive_decision_input = tf.placeholder(tf.int64, [None],
                                                                 name='passive_decision_in')
                    self.passive_decision_target = tf.one_hot(self.passive_decision_input, 4)
                    self.passive_decision_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.passive_decision_target, logits=passive_decision_logits)

                    self.passive_response_input = tf.placeholder(tf.int64, [None],
                                                                 name='passive_response_in')
                    self.passive_response_target = tf.one_hot(self.passive_response_input, 15)
                    self.passive_response_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.passive_response_target, logits=passive_response_logits)

                    self.passive_bomb_input = tf.placeholder(tf.int64, [None], name='passive_bomb_in')
                    self.passive_bomb_target = tf.one_hot(self.passive_bomb_input, 13)
                    self.passive_bomb_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.passive_bomb_target, logits=passive_bomb_logits)

                # active mode
                with tf.variable_scope("active_mode_loss"):
                    self.active_decision_input = tf.placeholder(tf.int64, [None], name='active_decision_in')
                    self.active_decision_target = tf.one_hot(self.active_decision_input, 13)
                    self.active_decision_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.active_decision_target, logits=active_decision_logits)

                    self.active_response_input = tf.placeholder(tf.int64, [None], name='active_response_in')
                    self.active_response_target = tf.one_hot(self.active_response_input, 15)
                    self.active_response_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.active_response_target, logits=active_response_logits)

                    self.seq_length_input = tf.placeholder(tf.int64, [None], name='sequence_length_in')
                    self.seq_length_target = tf.one_hot(self.seq_length_input, 12)
                    self.seq_length_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.seq_length_target, logits=seq_length_logits)

                with tf.variable_scope("minor_mode_loss"):
                    self.minor_response_input = tf.placeholder(tf.int64, [None], name='minor_response_in')
                    self.minor_response_target = tf.one_hot(self.minor_response_input, 15)
                    self.minor_response_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.minor_response_target, logits=minor_logits)

                l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
                l2_main_loss = [l for l in l2_loss if 'branch_main' in l.name]
                l2_passive_fc_loss = [l for l in l2_loss if 'branch_passive' in l.name and 'decision' not in l.name and 'bomb' not in l.name and 'response'not in l.name]
                l2_active_fc_loss = [l for l in l2_loss if 'branch_active' in l.name and 'decision' not in l.name and 'response' not in l.name and 'seq_length'not in l.name]

                print('l2 loss', len(l2_loss))
                print('l2 main loss', len(l2_main_loss))
                print('l2 passive fc loss', len(l2_passive_fc_loss))
                print('l2 active fc loss', len(l2_active_fc_loss))

                name_scopes = ['branch_passive/decision', 'branch_passive/bomb', 'branch_passive/response',
                               'branch_active/decision', 'branch_active/response', 'branch_active/seq_length', 'branch_minor']

                self.losses = [self.passive_decision_loss, self.passive_bomb_loss, self.passive_response_loss,
                               self.active_decision_loss, self.active_response_loss, self.seq_length_loss, self.minor_response_loss]
                self.optimize = []

                self.gradient_norms = []
                for i, name in enumerate(name_scopes):
                    l2_branch_loss = l2_main_loss.copy()
                    if 'passive' in name:
                        if 'bomb' in name:
                            l2_branch_loss += [l for l in l2_loss if name in l.name]
                        else:
                            l2_branch_loss += l2_passive_fc_loss + [l for l in l2_loss if name in l.name]
                    else:
                        l2_branch_loss += l2_active_fc_loss + [l for l in l2_loss if name in l.name]

                    print('l2 branch loss', len(l2_branch_loss))

                    gvs = trainer.compute_gradients(self.losses[i] + tf.add_n(l2_branch_loss))
                    gvs = [(gv[0], gv[1]) for gv in gvs if gv[0] is not None]
                    g, v = zip(*gvs)
                    g, global_norm = tf.clip_by_global_norm(g, 5.0)
                    self.gradient_norms.append(global_norm)
                    update = trainer.apply_gradients(zip(g, v))
                    self.optimize.append(update)
                self.weight_norm = tf.global_norm(
                    [v for v in tf.trainable_variables(scope=scope) if v.name.endswith('weights:0')])


