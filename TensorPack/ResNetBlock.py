import tensorflow as tf
import tensorflow.contrib.slim as slim


# adopt the idea to put batch normalization behind relu and full pre-activation
def identity_block(input, first_channel, last_channel, kernel_size):
    nonlinear1a_branch1a = tf.nn.relu(input)
    bn1a_branch1a = tf.contrib.layers.layer_norm(nonlinear1a_branch1a, scale=False)
    conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=bn1a_branch1a, num_outputs=first_channel,
                                            kernel_size=[1, 1], stride=[1, 1], padding='SAME')

    nonlinear1a_branch1b = tf.nn.relu(conv1a_branch1a)
    bn1a_branch1b = tf.contrib.layers.layer_norm(nonlinear1a_branch1b, scale=False)
    conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=bn1a_branch1b, num_outputs=first_channel,
                                            kernel_size=[1, kernel_size], stride=[1, 1], padding='SAME')

    nonlinear1a_branch1c = tf.nn.relu(conv1a_branch1b)
    bn1a_branch1c = tf.contrib.layers.layer_norm(nonlinear1a_branch1c, scale=False)
    conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=bn1a_branch1c, num_outputs=last_channel,
                                            kernel_size=[1, 1], stride=[1, 1], padding='SAME')

    #####

    return conv1a_branch1c + input


def upsample_block(input, first_channel, last_channel, kernel_size):
    nonlinear1a_branch1a = tf.nn.relu(input)
    bn1a_branch1a = tf.contrib.layers.layer_norm(nonlinear1a_branch1a, scale=False)
    conv1a_branch1a = slim.conv2d(activation_fn=None, inputs=bn1a_branch1a, num_outputs=first_channel,
                                  kernel_size=[1, 1], stride=[1, 2], padding='SAME')

    nonlinear1a_branch1b = tf.nn.relu(conv1a_branch1a)
    bn1a_branch1b = tf.contrib.layers.layer_norm(nonlinear1a_branch1b, scale=False)
    conv1a_branch1b = slim.conv2d(activation_fn=None, inputs=bn1a_branch1b, num_outputs=first_channel,
                                  kernel_size=[1, kernel_size], stride=[1, 1], padding='SAME')

    nonlinear1a_branch1c = tf.nn.relu(conv1a_branch1b)
    bn1a_branch1c = tf.contrib.layers.layer_norm(nonlinear1a_branch1c, scale=False)
    conv1a_branch1c = slim.conv2d(activation_fn=None, inputs=bn1a_branch1c, num_outputs=last_channel,
                                  kernel_size=[1, 1], stride=[1, 1], padding='SAME')

    ######

    nonlinear1a_branch2 = tf.nn.relu(input)
    bn1a_branch2 = tf.contrib.layers.layer_norm(nonlinear1a_branch2, scale=False)
    conv1a_branch2 = slim.conv2d(activation_fn=None, inputs=bn1a_branch2, num_outputs=last_channel,
                                 kernel_size=[1, kernel_size], stride=[1, 2], padding='SAME')

    return conv1a_branch1c + conv1a_branch2
