import tensorflow as tf
import tensorflow.contrib.slim as slim


# adopt the idea to put batch normalization behind relu and full pre-activation
def identity_block(input, num_channel, kernel_size):
    net = tf.contrib.layers.layer_norm(input, scale=True)
    net = tf.nn.relu(net)
    residual = slim.conv2d(activation_fn=None, inputs=net, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=[1, kernel_size], stride=[1, 1], padding='SAME')

    residual = tf.contrib.layers.layer_norm(residual, scale=True)
    residual = tf.nn.relu(residual)
    residual = slim.conv2d(activation_fn=None, inputs=residual, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=[1, kernel_size], stride=[1, 1], padding='SAME')

    #####
    identity = input
    if input.shape[-1].value != num_channel:
        identity = slim.conv2d(activation_fn=None, inputs=net, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=[1, kernel_size], stride=[1, 1], padding='SAME')

    return residual + identity


def downsample_block(input, num_channel, kernel_size):
    net = tf.contrib.layers.layer_norm(input, scale=True)
    net = tf.nn.relu(net)
    residual = slim.conv2d(activation_fn=None, inputs=net, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=[1, kernel_size], stride=[1, 2], padding='SAME')
    residual = tf.contrib.layers.layer_norm(residual, scale=True)
    residual = tf.nn.relu(residual)
    residual = slim.conv2d(activation_fn=None, inputs=residual, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=[1, kernel_size], stride=[1, 1], padding='SAME')

    ######

    shortcut = slim.conv2d(activation_fn=None, inputs=net, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=[1, 1], stride=[1, 2], padding='SAME')

    return residual + shortcut


def upsample_block(input, num_channel, kernel_size=2):
    net = tf.contrib.layers.layer_norm(input, scale=True)
    net = tf.nn.relu(net)
    residual = slim.conv2d_transpose(activation_fn=None, inputs=net, num_outputs=num_channel, biases_initializer=None,
                           kernel_size=[1, kernel_size], stride=[1, kernel_size], padding='SAME')
    residual = tf.contrib.layers.layer_norm(residual, scale=True)
    residual = tf.nn.relu(residual)
    residual = slim.conv2d_transpose(activation_fn=None, inputs=residual, num_outputs=num_channel, biases_initializer=None,
                           kernel_size=[1, kernel_size], stride=[1, 1], padding='SAME')

    ######

    shortcut = slim.conv2d_transpose(activation_fn=None, inputs=net, num_outputs=num_channel, biases_initializer=None,
                           kernel_size=[1, 1], stride=[1, kernel_size], padding='SAME')

    return residual + shortcut
