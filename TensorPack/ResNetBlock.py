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


def conv_block(input, conv_dim, input_dim, res_params, scope):
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
                    conv = identity_block(conv, param[0], param[1], param[2])
                elif param[-1] == 'upsampling':
                    conv = upsample_block(conv, param[0], param[1], param[2])
                else:
                    raise Exception('unsupported layer type')
            conv_out.append(slim.flatten(conv))

    flattened = tf.concat(conv_out, 1)
    return flattened