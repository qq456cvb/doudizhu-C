import tensorflow as tf
from tensorpack.tfutils import SessionInit


from tensorpack import *


class ModelLoader(SessionInit):
    def __init__(self, scope_in_sess, scope_in_ckpt):
        self.scope_in_sess = scope_in_sess
        self.scope_in_ckpt = scope_in_ckpt

    def _setup_graph(self):
        logger.info('building restorer')
        variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_in_sess)
        logger.info('variables to load has length:{}'.format(len(variables_to_restore)))
        variables_to_restore = {v.name.replace(self.scope_in_sess, self.scope_in_ckpt).replace(':0', ''): v for v in variables_to_restore}
        self.restorer = tf.train.Saver(variables_to_restore)

    def _run_init(self, sess):
        logger.info("loading models")
        self.restorer.restore(sess, '../PolicySL/train_log/train-SL-1.4/model-100000')


