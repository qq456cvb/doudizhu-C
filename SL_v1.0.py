import argparse
import sys
from network_SL_lite import CardNetwork
import tensorflow as tf
sys.path.insert(0, './build/Release')
from env import Env
from logger import Logger
from utils import to_char
from card import Card
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='fight the lord feature vector')
    # parser.add_argument('--b', type=int, help='batch size', default=32)
    parser.add_argument('--epoches_train', type=int, help='num of epochs to train', default=10000)
    parser.add_argument('--epoches_test', type=int, help='num of epochs to test', default=1000)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.set_defaults(train=True)

    args = parser.parse_args(sys.argv[1:])
    epoches_train = args.epoches_train
    epoches_test = args.epoches_test

    SLNetwork = CardNetwork(60 * 6, tf.train.AdamOptimizer(learning_rate=1e-3), "SLNetwork")
    empty_thresh = 0.2

    env = Env()
    TRAIN = args.train
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=50)

    file_writer = tf.summary.FileWriter('accuracy_fake_minor', sess.graph)

    logger = Logger(moving_avg=True if TRAIN else False)
    if TRAIN:
        sess.run(tf.global_variables_initializer())
        for i in range(epoches_train):
            # print('episode: ', i)
            env.reset()
            env.prepare()

            r = 0
            while r == 0:
                last_cards_value = env.get_last_outcards()
                last_cards_char = to_char(last_cards_value)
                last_out_cards = Card.val2onehot60(last_cards_value)
                is_active = True if last_cards_value.size == 0 else False

                s = env.get_state_padded()
                intention, r, category_idx = env.step_auto()
                target_policy = Card.val2onehot(intention)

                policy_out = None
                if is_active:
                    _, policy_out = sess.run([SLNetwork.optimize_active,
                                              SLNetwork.active_policy_out],
                                             feed_dict={
                                                 SLNetwork.input_state: s.reshape([1, -1]),
                                                 SLNetwork.training: True,
                                                 SLNetwork.target_policy: target_policy.reshape([1, -1])
                                             })
                else:
                    _, policy_out = sess.run([SLNetwork.optimize_passive,
                                              SLNetwork.passive_policy_out],
                                             feed_dict={
                                                 SLNetwork.input_state: s.reshape([1, -1]),
                                                 SLNetwork.training: True,
                                                 SLNetwork.target_policy: target_policy.reshape([1, -1]),
                                                 SLNetwork.last_outcards: last_out_cards.reshape([1, -1])
                                             })

                n_cards = len(intention)
                if n_cards == 0:
                    logger.updateAcc('acc', 1 if policy_out[0][policy_out[0] > empty_thresh].size == 0 else 0)
                else:
                    max_ind = np.argpartition(policy_out[0], -n_cards)[-n_cards:]
                    pred_intention = np.array(Card.onehot2val(max_ind))
                    logger.updateAcc('acc', 1 if np.array_equal(np.sort(intention), np.sort(pred_intention)) else 0)


