import tensorflow as tf
from tensorpack.tfutils import SessionInit


from tensorpack import *


def update_params(scope_from, scope_to):
    vars_from = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_from)
    vars_to = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_to)

    ops = []
    for from_var, to_var in zip(vars_from, vars_to):
        ops.append(to_var.assign(from_var))
    return ops


class ModelLoader(SessionInit):
    def __init__(self, scope_in_sess_policy, scope_in_ckpt_policy, scope_in_sess_value, scope_in_ckpt_value,
                 pn_path='../PolicySL/train_log/train-SL-1.4/model-100000',
                 vn_path='../ValueSL/train_log/train-SL-1.4/model-14000'):
        self.scope_in_sess_policy = scope_in_sess_policy
        self.scope_in_ckpt_policy = scope_in_ckpt_policy
        self.scope_in_sess_value = scope_in_sess_value
        self.scope_in_ckpt_value = scope_in_ckpt_value
        self.pn_path = pn_path
        self.vn_path = vn_path

    def _setup_graph(self):
        logger.info('building restorer')
        variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_in_sess_policy)
        logger.info('policy variables to load has length:{}'.format(len(variables_to_restore)))
        variables_to_restore = {v.name.replace(self.scope_in_sess_policy, self.scope_in_ckpt_policy).replace(':0', ''): v for v in variables_to_restore}
        self.policy_restorer = tf.train.Saver(variables_to_restore)

        variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_in_sess_value)
        logger.info('value variables to load has length:{}'.format(len(variables_to_restore)))
        variables_to_restore = {v.name.replace(self.scope_in_sess_value, self.scope_in_ckpt_value).replace(':0', ''): v for v in variables_to_restore}
        self.value_restorer = tf.train.Saver(variables_to_restore)

        # ugly hardcode
        assert self.scope_in_sess_policy == 'policy_network_2'
        self.update_ops = update_params('policy_network_2', 'policy_network_3') + update_params('policy_network_2', 'policy_network_1')

    def _run_init(self, sess):
        logger.info("loading models")
        self.policy_restorer.restore(sess, self.pn_path)
        self.value_restorer.restore(sess, self.vn_path)
        sess.run(self.update_ops)


if __name__ == '__main__':
    a = tf.reshape(tf.range(10), [-1, 2])
    b = tf.equal(tf.constant([1, 2, 0, 0, 2]), 3)
    c = tf.where(b)
    d = tf.zeros([0, 2])
    e = tf.scatter_nd(c, d, shape=tf.cast(tf.stack([tf.shape(a)[0], tf.shape(a)[1]]), dtype=tf.int64))
    f = tf.shape(tf.ones([0]))
    with tf.Session() as sess:
        print(sess.run(c))

