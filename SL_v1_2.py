import argparse
import sys
from network_SL_v1_2 import CardNetwork
import tensorflow as tf
sys.path.insert(0, './build/Release')
from env import Env
from logger import Logger
from utils import to_char
from card import Card, action_space, Category
import numpy as np
from utils import get_mask, get_minor_cards, train_fake_action_60, get_masks, test_fake_action
from utils import get_seq_length, pick_minor_targets, to_char, to_value, get_mask_alter
from utils import inference_minor_cards, gputimeblock, scheduled_run, give_cards_without_minor, pick_main_cards


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='fight the lord feature vector')
    # parser.add_argument('--b', type=int, help='batch size', default=32)
    parser.add_argument('--epoches_train', type=int, help='num of epochs to train', default=20000)
    parser.add_argument('--epoches_test', type=int, help='num of epochs to test', default=1000)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.set_defaults(train=True)

    args = parser.parse_args(sys.argv[1:])
    epoches_train = args.epoches_train
    epoches_test = args.epoches_test

    network = CardNetwork(tf.train.RMSPropOptimizer(learning_rate=1e-4), "network")
    env = Env()
    TRAIN = args.train
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=50)

    # file_writer = tf.summary.FileWriter('accuracy_fake_minor', sess.graph)
    tf.logging.set_verbosity(tf.logging.INFO)

    logger = Logger(moving_avg=True if TRAIN else False)
    global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False, dtype=tf.int32)
    global_step_add = global_step.assign_add(1)
    if TRAIN:
        sess.run(tf.global_variables_initializer())
        # weight_norm = sess.run(network.weight_norm)
        # tf.logging.info('weight norm is {}'.format(weight_norm))
        # saver.restore(sess, tf.train.latest_checkpoint('./Model/SL_lite/'))
        # saver.restore(sess, './Model/SL_lite/model-9500')
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for v in vars:
            print(v.name)
            print(v.shape)
            # if v.name.endswith('weights:0'):
            #     print(v.name)
            #     print(sess.run(tf.norm(v)))
        i = sess.run(global_step)
        while i < epoches_train:
            print('episode: ', i)
            env.reset()
            env.prepare()

            r = 0
            while r == 0:
                last_cards_value = env.get_last_outcards()
                last_cards_char = to_char(last_cards_value)
                last_category_idx = env.get_last_outcategory_idx()
                curr_cards_char = to_char(env.get_curr_handcards())
                is_active = True if last_cards_value.size == 0 else False

                # add supervised information?
                input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, None)
                masks_handcards = np.concatenate([input_single, input_pair, input_triple, input_quadric])
                last_input_single, last_input_pair, last_input_triple, last_input_quadric = get_masks(last_cards_char, None)
                masks_lastcards = np.concatenate([last_input_single, last_input_pair, last_input_triple, last_input_quadric])
                intention, r, category_idx = env.step_auto()

                if category_idx == 14:
                    continue
                policy_out = None
                minor_cards_targets = pick_minor_targets(category_idx, to_char(intention))

                if not is_active:
                    if category_idx == Category.QUADRIC.value and category_idx != last_category_idx:
                        passive_decision_input = 1
                        passive_bomb_input = intention[0] - 3
                        _, _, decision_passive_output, bomb_passive_output = sess.run([network.optimize[0],
                                                                                       network.optimize[1],
                                                                                       network.passive_decision_probs,
                                                                                       network.passive_bomb_probs], feed_dict={
                            network.mask_in: masks_handcards.reshape(1, -1),
                            network.last_masks_in: masks_lastcards.reshape(1, -1),
                            network.passive_decision_input: np.array([passive_decision_input]),
                            network.passive_bomb_input: np.array([passive_bomb_input])
                        })
                        passive_bomb_acc_temp = 1 if np.argmax(bomb_passive_output[0]) == passive_bomb_input else 0
                        logger.updateAcc("passive_bomb", passive_bomb_acc_temp)

                    else:
                        if category_idx == Category.BIGBANG.value:
                            passive_decision_input = 2
                            _, decision_passive_output = sess.run([network.optimize[0],
                                          network.passive_decision_probs], feed_dict={
                                network.mask_in: masks_handcards.reshape(1, -1),
                                network.last_masks_in: masks_lastcards.reshape(1, -1),
                                network.passive_decision_input: np.array([passive_decision_input])
                            })
                        else:
                            if category_idx != Category.EMPTY.value:
                                passive_decision_input = 3
                                # OFFSET_ONE
                                # 1st, Feb - remove relative card output since shift is hard for the network to learn
                                passive_response_input = intention[0] - 3
                                if passive_response_input < 0:
                                    print("something bad happens")
                                    passive_response_input = 0
                                _, _, decision_passive_output, response_passive_output, grad_norm = sess.run([network.optimize[0],
                                                                                                network.optimize[2],
                                                                                                network.passive_decision_probs,
                                                                                                network.passive_response_probs,
                                                                                                network.gradient_norms[0]], feed_dict={
                                    network.mask_in: masks_handcards.reshape(1, -1),
                                    network.last_masks_in: masks_lastcards.reshape(1, -1),
                                    network.passive_decision_input: np.array([passive_decision_input]),
                                    network.passive_response_input: np.array([passive_response_input])
                                })
                                passive_response_acc_temp = 1 if np.argmax(response_passive_output[0]) == \
                                                                 passive_response_input else 0
                                if i % 100 == 0:
                                    tf.logging.info('decision passive gradient norm: {}'.format(grad_norm))
                                logger.updateAcc("passive_response", passive_response_acc_temp)
                            else:
                                passive_decision_input = 0
                                _, decision_passive_output = sess.run([network.optimize[0],
                                                                      network.passive_decision_probs], feed_dict={
                                    network.mask_in: masks_handcards.reshape(1, -1),
                                    network.last_masks_in: masks_lastcards.reshape(1, -1),
                                    network.passive_decision_input: np.array([passive_decision_input])
                                })
                    passive_decision_acc_temp = 1 if np.argmax(decision_passive_output[0]) == passive_decision_input else 0
                    logger.updateAcc("passive_decision", passive_decision_acc_temp)

                else:
                    seq_length = get_seq_length(category_idx, intention)

                    # ACTIVE OFFSET ONE!
                    active_decision_input = category_idx - 1
                    active_response_input = intention[0] - 3
                    _, _, decision_active_output, response_active_output, grad_norm = sess.run([network.optimize[3],
                                                             network.optimize[4],
                                                             network.active_decision_probs,
                                                             network.active_response_probs,
                                                             network.gradient_norms[3]], feed_dict={
                        network.mask_in: masks_handcards.reshape(1, -1),
                        network.active_decision_input: np.array([active_decision_input]),
                        network.active_response_input: np.array([active_response_input])
                    })
                    if i % 100 == 0:
                        tf.logging.info('decision active gradient norm: {}'.format(grad_norm))

                    active_decision_acc_temp = 1 if np.argmax(decision_active_output[0]) == active_decision_input else 0
                    logger.updateAcc("active_decision", active_decision_acc_temp)

                    active_response_acc_temp = 1 if np.argmax(response_active_output[0]) == active_response_input else 0
                    logger.updateAcc("active_response", active_response_acc_temp)

                    if seq_length is not None:
                        # length offset one
                        seq_length_input = seq_length - 1
                        _, seq_length_output = sess.run([network.optimize[5],
                                                        network.seq_length_probs], feed_dict={
                            network.mask_in: masks_handcards.reshape(1, -1),
                            network.seq_length_input: np.array([seq_length_input])
                        })
                        seq_acc_temp = 1 if np.argmax(seq_length_output[0]) == seq_length_input else 0
                        logger.updateAcc("seq_length", seq_acc_temp)

                # if minor_cards_targets is not None:
                #     main_cards = pick_main_cards(category_idx, to_char(intention))
                #     accs = train_fake_action_60(minor_cards_targets, curr_cards_char.copy(), s.copy(), sess, network, category_idx, main_cards)
                #     for acc in accs:
                #         logger.updateAcc("minor_cards", acc)

            if i % 100 == 0:
                print("train ", i, " ing...")
                print("train passive decision accuracy = ", logger["passive_decision"])
                print("train passive response accuracy = ", logger["passive_response"])
                print("train passive bomb accuracy = ", logger["passive_bomb"])
                print("train active decision accuracy = ", logger["active_decision"])
                print("train active response accuracy = ", logger["active_response"])
                print("train sequence length accuracy = ", logger["seq_length"])
                # print("train minor cards accuracy = ", logger["minor_cards"])
                weight_norm = sess.run(network.weight_norm)
                tf.logging.info('weight norm: {} '.format(weight_norm))

            if i % 200 == 0 and i > 0:
                saver.save(sess, "./Model/SL_lite/model", global_step=i)

            sess.run(global_step_add)
            i = sess.run(global_step)


