import tensorflow as tf
import tensorflow.contrib.slim as slim


# adopt the idea to put batch normalization behind relu and full pre-activation
def identity_block(input, num_channel, kernel_size):
    net = tf.contrib.layers.layer_norm(input, scale=True)
    net = tf.nn.relu(net)
    residual = slim.conv2d(activation_fn=None, inputs=net, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=kernel_size, stride=1, padding='SAME')

    residual = tf.contrib.layers.layer_norm(residual, scale=True)
    residual = tf.nn.relu(residual)
    residual = slim.conv2d(activation_fn=None, inputs=residual, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=kernel_size, stride=1, padding='SAME')

    #####
    x = input
    if input.shape[-1].value != num_channel:
        x = slim.conv2d(activation_fn=None, inputs=net, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=kernel_size, stride=1, padding='SAME')

    return residual + x


def upsample_block(input, num_channel, kernel_size):
    if input.shape[1].value == 1:
        stride = [1, 2]
    else:
        stride = 2
    net = tf.contrib.layers.layer_norm(input, scale=True)
    net = tf.nn.relu(net)
    residual = slim.conv2d(activation_fn=None, inputs=net, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=kernel_size, stride=stride, padding='SAME')

    residual = tf.contrib.layers.layer_norm(residual, scale=True)
    residual = tf.nn.relu(residual)
    residual = slim.conv2d(activation_fn=None, inputs=residual, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=kernel_size, stride=1, padding='SAME')

    ######

    shortcut = slim.conv2d(activation_fn=None, inputs=net, num_outputs=num_channel, biases_initializer=None,
                                            kernel_size=kernel_size, stride=stride, padding='SAME')

    return residual + shortcut


def conv_block(input, res_params, scope):

    with tf.variable_scope(scope):
        input_conv = tf.reshape(input, tf.stack([-1, tf.shape(input)[1] // 4, 4, 1]))

        conv_shapes = [[1, 1], [1, 2], [1, 3], [1, 4], [3, 1], [5, 1], [3, 2], [5, 2], [3, 3], [5, 3]]

        conv_out = []
        for conv_kernel in conv_shapes:
            block = slim.conv2d(input_conv, 32, conv_kernel, activation_fn=None)
            for _ in range(5):
                block = identity_block(block, 32, conv_kernel)
            conv_out.append(block)

        conv_concat = tf.concat(conv_out, -1)

        for param in res_params:
            if param[-1] == 'identity':
                conv_concat = identity_block(conv_concat, param[0], param[1])
            elif param[-1] == 'upsampling':
                conv_concat = upsample_block(conv_concat, param[0], param[1])
            else:
                raise Exception('unsupported layer type')

    flattened = tf.reduce_mean(conv_concat, [1, 2])
    return flattened
