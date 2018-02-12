import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


def resblock(input, first_channel, last_channel, kernel_size, training):
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
    output = slim.avg_pool2d(inputs=plus, kernel_size=[1, 2])

    return output


def conv_block(input, input_dim, res_params, training, scope):
    conv_out = []
    with tf.variable_scope(scope):
        input_conv = tf.reshape(input, [-1, 1, input_dim, 1])
        single_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=64,
                                  kernel_size=[1, 1], stride=[1, 4], padding='VALID')

        pair_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=64,
                                kernel_size=[1, 2], stride=[1, 4], padding='VALID')

        triple_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=64,
                                  kernel_size=[1, 3], stride=[1, 4], padding='VALID')

        quadric_conv = slim.conv2d(activation_fn=None, inputs=input_conv, num_outputs=64,
                                   kernel_size=[1, 4], stride=[1, 4], padding='VALID')

        conv_list = [single_conv, pair_conv, triple_conv, quadric_conv]

        for conv in conv_list:
            for param in res_params:
                conv = resblock(conv, param[0], param[1], param[2], training)
            conv_out.append(slim.flatten(conv))

    flattened = tf.concat(conv_out, 1)
    return flattened


class CardNetwork:
        def __init__(self, s_dim, trainer, scope):
            with tf.variable_scope(scope):
                with tf.variable_scope('inputs'):
                    self.input_state = tf.placeholder(tf.float32, [None, s_dim], name="input")
                    self.training = tf.placeholder(tf.bool, None, name='training')
                    self.last_outcards = tf.placeholder(tf.float32, [None, 60], name='last_cards')
                    self.target_policy = tf.placeholder(tf.float32, [None, 54], 'target_policy_input')

                with slim.arg_scope([slim.fully_connected, slim.conv2d], weights_regularizer=slim.l2_regularizer(1e-3)):
                    flattened = conv_block(self.input_state, s_dim, [[64, 128, 5], [128, 256, 5], [256, 512, 5]], self.training, 'active_conv')

                    with tf.variable_scope('active_branch'):
                        active_fc = slim.fully_connected(inputs=flattened, num_outputs=1024, activation_fn=tf.nn.relu)
                        active_fc = slim.fully_connected(inputs=active_fc, num_outputs=256, activation_fn=tf.nn.relu)
                        self.active_policy_out = slim.fully_connected(active_fc, 54, tf.nn.sigmoid)

                    with tf.variable_scope('passive_branch'):
                        passive_fc = slim.fully_connected(inputs=flattened, num_outputs=1024, activation_fn=tf.nn.relu)
                        attention = conv_block(self.last_outcards, 60, [[64, 128, 5], [128, 256, 5], [256, 512, 5]], self.training, 'passive_conv')
                        attention = slim.fully_connected(inputs=attention, num_outputs=1024, activation_fn=tf.nn.sigmoid)
                        passive_fc = attention * passive_fc
                        passive_fc = slim.fully_connected(inputs=passive_fc, num_outputs=256, activation_fn=tf.nn.relu)
                        self.passive_policy_out = slim.fully_connected(passive_fc, 54, tf.nn.sigmoid)

                with tf.variable_scope('policy_loss'):
                    self.pasive_policy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.passive_policy_out, 1e-8, 1.)) * (self.target_policy - 0.5) * 2)
                    self.active_policy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.active_policy_out, 1e-8, 1.)) * (self.target_policy - 0.5) * 2)

                l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
                l2_passive_loss = [l for l in l2_loss if 'active_branch' not in l.name]
                l2_active_loss = [l for l in l2_loss if 'passive_branch' not in l.name]
                self.passive_loss = self.pasive_policy_loss + l2_passive_loss
                self.active_loss = self.active_policy_loss + l2_active_loss

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                passive_vars = [v for v in local_vars if 'active_branch' not in v.name]
                active_vars = [v for v in local_vars if 'passive_branch' not in v.name]
                self.var_norms = tf.global_norm(local_vars)

                self.passive_gradients = tf.gradients(self.passive_loss, passive_vars)
                passive_cliped_grads, self.passive_grad_norms = tf.clip_by_global_norm(self.passive_gradients, 4.0)
                self.active_gradients = tf.gradients(self.active_loss, active_vars)
                active_cliped_grads, self.active_grad_norms = tf.clip_by_global_norm(self.active_gradients, 4.0)

                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
                passive_extra_update_ops = [op for op in extra_update_ops if 'active_branch' not in op.name]
                active_extra_update_ops = [op for op in extra_update_ops if 'passive_branch' not in op.name]
                print(len(passive_extra_update_ops))
                print(len(active_extra_update_ops))

                with tf.control_dependencies(passive_extra_update_ops):
                    with tf.variable_scope('optimize'):
                        self.optimize_passive = trainer.apply_gradients(zip(passive_cliped_grads, passive_vars))

                with tf.control_dependencies(active_extra_update_ops):
                    with tf.variable_scope('optimize'):
                        self.optimize_active = trainer.apply_gradients(zip(active_cliped_grads, active_vars))

