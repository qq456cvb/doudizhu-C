# from env import Env
import sys
sys.path.insert(0, './build/Release')
import env
from env import Env
# from env_test import Env
import card
from card import action_space, Category
import os, random
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import time
# from env_test import get_benchmark
from collections import Counter
import struct
import copy
import random
import argparse
from logger import Logger
from network_SL import CardNetwork
from utils import get_mask, get_minor_cards, train_fake_action, get_masks, test_fake_action
from utils import get_seq_length, pick_minor_targets, to_char, to_value, get_mask_alter
from utils import inference_minor_cards, gputimeblock, scheduled_run, give_cards_without_minor, pick_main_cards
import shutil
from pyenv import Pyenv

##################################################### UTILITIES ########################################################

def inference_once(s, sess, main_network):
    is_active = (s['control_idx'] == s['idx'])
    last_category_idx = s['last_category_idx'] if not is_active else -1
    last_cards_char = s['last_cards'] if not is_active else np.array([])
    last_cards_value = np.array(to_value(last_cards_char)) if not is_active else np.array([])
    curr_cards_char = s['player_cards'][s['idx']]

    input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, last_cards_char)
    input_single_last, input_pair_last, input_triple_last, input_quadric_last = get_masks(last_cards_char, None)

    state = Pyenv.get_state_static(s).reshape(1, -1)
    feeddict = (
        (main_network.training, True),
        (main_network.input_state, state),
        (main_network.input_single, input_single.reshape(1, -1)),
        (main_network.input_pair, input_pair.reshape(1, -1)),
        (main_network.input_triple, input_triple.reshape(1, -1)),
        (main_network.input_quadric, input_quadric.reshape(1, -1))
    )
    intention = None
    if is_active:
        # first get mask
        decision_mask, response_mask, _, length_mask = get_mask_alter(curr_cards_char, [], False, last_category_idx)

        with gputimeblock('gpu'):
            decision_active_output = scheduled_run(sess, main_network.fc_decision_active_output, feeddict)
            # decision_active_output = sess.run(self.main_network.fc_decision_active_output,
            #                                   feed_dict=feeddict)

        # make decision depending on output
        decision_active_output = decision_active_output[0]
        decision_active_output[decision_mask == 0] = -1
        decision_active = np.argmax(decision_active_output)

        active_category_idx = decision_active + 1

        # give actual response
        with gputimeblock('gpu'):
            response_active_output = scheduled_run(sess, main_network.fc_response_active_output, feeddict)
            # response_active_output = sess.run(self.main_network.fc_response_active_output,
            #                                   feed_dict=feeddict)

        response_active_output = response_active_output[0]
        response_active_output[response_mask[decision_active] == 0] = -1
        response_active = np.argmax(response_active_output)

        seq_length = 0
        # next sequence length
        if active_category_idx == Category.SINGLE_LINE.value or \
                active_category_idx == Category.DOUBLE_LINE.value or \
                active_category_idx == Category.TRIPLE_LINE.value or \
                active_category_idx == Category.THREE_ONE_LINE.value or \
                active_category_idx == Category.THREE_TWO_LINE.value:
            with gputimeblock('gpu'):
                seq_length_output = scheduled_run(sess, main_network.fc_sequence_length_output, feeddict)
                # seq_length_output = sess.run(self.main_network.fc_sequence_length_output,
                #                              feed_dict=feeddict)

            seq_length_output = seq_length_output[0]
            seq_length_output[length_mask[decision_active][response_active] == 0] = -1
            seq_length = np.argmax(seq_length_output) + 1

        # give main cards
        intention = give_cards_without_minor(response_active, last_cards_value, active_category_idx, seq_length)

        # then give minor cards
        if active_category_idx == Category.THREE_ONE.value or \
                active_category_idx == Category.THREE_TWO.value or \
                active_category_idx == Category.THREE_ONE_LINE.value or \
                active_category_idx == Category.THREE_TWO_LINE.value or \
                active_category_idx == Category.FOUR_TWO.value:
            dup_mask = np.ones([15])
            if seq_length > 0:
                for i in range(seq_length):
                    dup_mask[intention[0] - 3 + i] = 0
            else:
                dup_mask[intention[0] - 3] = 0
            intention = np.concatenate([intention, to_value(
                inference_minor_cards(active_category_idx, state.copy(),
                                      list(curr_cards_char.copy()), sess, main_network, seq_length,
                                      dup_mask, to_char(intention))[0])])
    else:
        is_bomb = False
        if len(last_cards_value) == 4 and len(set(last_cards_value)) == 1:
            is_bomb = True
        # print(to_char(last_cards_value), is_bomb, last_category_idx)
        decision_mask, response_mask, bomb_mask, _ = get_mask_alter(curr_cards_char, to_char(last_cards_value),
                                                                    is_bomb, last_category_idx)

        feeddict = (
            (main_network.training, True),
            (main_network.input_state, state),
            (main_network.input_single, input_single.reshape(1, -1)),
            (main_network.input_pair, input_pair.reshape(1, -1)),
            (main_network.input_triple, input_triple.reshape(1, -1)),
            (main_network.input_quadric, input_quadric.reshape(1, -1)),
            (main_network.input_single_last, input_single_last.reshape(1, -1)),
            (main_network.input_pair_last, input_pair_last.reshape(1, -1)),
            (main_network.input_triple_last, input_triple_last.reshape(1, -1)),
            (main_network.input_quadric_last, input_quadric_last.reshape(1, -1)),
        )
        with gputimeblock('gpu'):
            decision_passive_output, response_passive_output, bomb_passive_output \
                = scheduled_run(sess, [main_network.fc_decision_passive_output,
                                 main_network.fc_response_passive_output, main_network.fc_bomb_passive_output], feeddict)
            # decision_passive_output, response_passive_output, bomb_passive_output \
            #     = sess.run([self.main_network.fc_decision_passive_output,
            #                 self.main_network.fc_response_passive_output, self.main_network.fc_bomb_passive_output],
            #                feed_dict=feeddict)

        # print(decision_mask)
        # print(decision_passive_output)
        decision_passive_output = decision_passive_output[0]
        decision_passive_output[decision_mask == 0] = -1
        decision_passive = np.argmax(decision_passive_output)

        if decision_passive == 0:
            intention = np.array([])
        elif decision_passive == 1:
            # print('bomb_mask', bomb_mask)
            bomb_passive_output = bomb_passive_output[0]
            bomb_passive_output[bomb_mask == 0] = -1
            bomb_passive = np.argmax(bomb_passive_output)

            # converting 0-based index to 3-based value
            intention = np.array([bomb_passive + 3] * 4)

        elif decision_passive == 2:
            intention = np.array([16, 17])
        elif decision_passive == 3:
            # print('response_mask', response_mask)
            response_passive_output = response_passive_output[0]
            response_passive_output[response_mask == 0] = -1
            # 0-14
            response_passive = np.argmax(response_passive_output)

            # there is an offset when converting from 0-based index to 1-based index
            # bigger = response_passive + 1

            intention = give_cards_without_minor(response_passive, last_cards_value, last_category_idx, None)
            if last_category_idx == Category.THREE_ONE.value or \
                    last_category_idx == Category.THREE_TWO.value or \
                    last_category_idx == Category.THREE_ONE_LINE.value or \
                    last_category_idx == Category.THREE_TWO_LINE.value or \
                    last_category_idx == Category.FOUR_TWO.value:
                dup_mask = np.ones([15])
                seq_length = get_seq_length(last_category_idx, intention)
                if seq_length:
                    for j in range(seq_length):
                        dup_mask[intention[0] - 3 + j] = 0
                else:
                    dup_mask[intention[0] - 3] = 0
                intention = np.concatenate(
                    [intention, to_value(inference_minor_cards(last_category_idx, state.copy(),
                                                               list(curr_cards_char.copy()),
                                                               sess, main_network,
                                                               get_seq_length(last_category_idx, last_cards_value),
                                                               dup_mask, to_char(intention))[0])])
    return intention


def get_benchmark(sess, network):
    num_tests = 100
    env = Pyenv()
    wins = 0
    idx_role_self = 3
    idx_self = -1
    for i in range(num_tests):
        env.reset()
        env.prepare()
        # print('lord is ', s['lord_idx'])
        done = False
        while not done:
            s = env.dump_state()
            # use the network
            if idx_role_self == Pyenv.get_role_ID_static(s):
                idx_self = s['idx']
                # print('this is the lord ', end='')
                # print(s['player_cards'][s['idx']])
                intention = np.array(to_char(inference_once(s, sess, network)))
            else:
                intention = np.array(to_char(Env.step_auto_static(card.Card.char2color(s['player_cards'][s['idx']]),
                                                                      np.array(to_value(
                                                                          s['last_cards'] if s['control_idx'] != s[
                                                                              'idx'] else [])))))
            # print(s['idx'], intention)
            r, done = env.step(intention)
        # finished, see who wins
        if idx_self == s['idx']:
            wins += 1
        # elif idx_self != s['lord_idx'] and s['idx'] != s['lord_idx']:
        #     wins += 1
    return wins / num_tests


if __name__ == '__main__':
    '''
    demoGames = []
    read_seq3("seq")
    numOfDemos = len(demoGames)

    # print("hc : ", demoGames[1].handcards)

    N = 200000
    f = open("data", "w")
    collect_data()
    f.close()
    '''

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

    a_dim = len(action_space)

    graph_sl = tf.get_default_graph()
    SLNetwork = CardNetwork(54 * 6, tf.train.AdamOptimizer(learning_rate=1e-4), "SLNetwork")
    # variables = tf.all_variables()

    e = env.Env()
    TRAIN = args.train
    sess = tf.Session(graph=graph_sl, config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver(max_to_keep=50)

    file_writer = tf.summary.FileWriter('accuracy_fake_minor', sess.graph)

    # filelist = [ f for f in os.listdir('./accuracy_fake_minor') ]
    # for f in filelist:
    #     os.remove(os.path.join('./accuracy_fake_minor', f))

    logger = Logger(moving_avg=True if TRAIN else False)
    # TODO: support batch training
    # test_cards = [i for i in range(3, 18)]
    # test_cards = np.array(test_cards *)
    if TRAIN:
        sess.run(tf.global_variables_initializer())
        for i in range(epoches_train):
            # print('episode: ', i)
            e.reset()
            e.prepare()
            
            curr_cards_value = e.get_curr_handcards()
            curr_cards_char = to_char(curr_cards_value)
            last_cards_value = e.get_last_outcards()
            # print("curr_cards = ", curr_cards_value)
            # print("last_cards = ", last_cards_value)
            last_category_idx = -1
            last_cards_char = to_char(last_cards_value)
            # mask = get_mask(curr_cards_char, action_space, last_cards_char)

            input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, None)
            input_single_last, input_pair_last, input_triple_last, input_quadric_last = get_masks(last_cards_char, None)

            s = e.get_state()
            s = np.reshape(s, [1, -1]).astype(np.float32)
            
            # s = get_feature_state(e, mask)
            r = 0
            while r == 0:
                active = True
                if last_cards_value.size > 0:
                    active = False

                has_seq_length = np.array([False])
                seq_length_input = np.array([0])
                is_passive_bomb = np.array([False])
                did_passive_response = np.array([False])
                passive_decision_input = np.array([0])
                passive_response_input = np.array([0])
                passive_bomb_input = np.array([0])
                active_decision_input = np.array([0])
                active_response_input = np.array([0])

                # if active:
                #     print("No one respond", end=':')
                # else:
                #     print("Another one respond", end=':')
                intention, r, category_idx = e.step_auto()
                #print("intention = ", intention)
                #print("r = ", r)
                #print("category_idx = ",category_idx)
                # print(intention)
                # do not account for 4 + 2 + 2 
                if category_idx == 14:
                    curr_cards_value = e.get_curr_handcards()
                    curr_cards_char = to_char(curr_cards_value)
                    last_cards_value = e.get_last_outcards()
                    if last_cards_value.size > 0:
                        last_category_idx = category_idx
                    last_cards_char = to_char(last_cards_value)
                    # mask = get_mask(curr_cards_char, action_space, last_cards_char)

                    input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, None)
                    input_single_last, input_pair_last, input_triple_last, input_quadric_last = get_masks(
                        last_cards_char, None)
                    continue

                # if category_idx == Category.THREE_ONE.value or category_idx == Category.THREE_TWO.value or \
                #         category_idx == Category.THREE_ONE_LINE.value or category_idx == Category.THREE_TWO_LINE.value or \
                #         category_idx == Category.FOUR_TWO.value:
                #     has_minor_cards[0] = True
                #     minor_cards_target[0], minor_cards_length[0] = get_minor_cards(intention, category_idx)
                minor_cards_targets = pick_minor_targets(category_idx, to_char(intention))
                
                if not active:
                    if category_idx == Category.QUADRIC.value and category_idx != last_category_idx:
                        is_passive_bomb[0] = True
                        passive_decision_input[0] = 1
                        passive_bomb_input[0] = intention[0] - 3
                    else:
                        if category_idx == Category.BIGBANG.value:
                            passive_decision_input[0] = 2
                        else:
                            if category_idx != Category.EMPTY.value:
                                passive_decision_input[0] = 3
                                did_passive_response[0] = True
                                # OFFSET_ONE
                                # 1st, Feb - remove relative card output since shift is hard for the network to learn
                                passive_response_input[0] = intention[0] - 3
                                if passive_response_input[0] < 0:
                                    print("something bad happens")
                                    passive_response_input[0] = 0
                else:
                    seq_length = get_seq_length(category_idx, intention)
                    if seq_length is not None:
                        has_seq_length[0] = True
                        # length offset one
                        seq_length_input[0] = seq_length - 1

                    # ACTIVE OFFSET ONE!
                    active_decision_input[0] = category_idx - 1
                    active_response_input[0] = intention[0] - 3

                _, decision_passive_output, response_passive_output, bomb_passive_output, \
                    decision_active_output, response_active_output, seq_length_output, loss, \
                    active_decision_loss, active_response_loss, passive_decision_loss, passive_response_loss, passive_bomb_loss \
                     = sess.run([SLNetwork.optimize, SLNetwork.fc_decision_passive_output, 
                                SLNetwork.fc_response_passive_output, SLNetwork.fc_bomb_passive_output,
                                SLNetwork.fc_decision_active_output, SLNetwork.fc_response_active_output, 
                                SLNetwork.fc_sequence_length_output, SLNetwork.loss,
                                SLNetwork.active_decision_loss, SLNetwork.active_response_loss, SLNetwork.passive_decision_loss, SLNetwork.passive_response_loss, SLNetwork.passive_bomb_loss],
                        feed_dict = {
                            SLNetwork.training: True,
                            SLNetwork.input_state: s,
                            SLNetwork.input_single: np.reshape(input_single, [1, -1]),
                            SLNetwork.input_pair: np.reshape(input_pair, [1, -1]),
                            SLNetwork.input_triple: np.reshape(input_triple, [1, -1]),
                            SLNetwork.input_quadric: np.reshape(input_quadric, [1, -1]),
                            SLNetwork.input_single_last: np.reshape(input_single_last, [1, -1]),
                            SLNetwork.input_pair_last: np.reshape(input_pair_last, [1, -1]),
                            SLNetwork.input_triple_last: np.reshape(input_triple_last, [1, -1]),
                            SLNetwork.input_quadric_last: np.reshape(input_quadric_last, [1, -1]),
                            SLNetwork.is_active: np.array([active]),
                            SLNetwork.is_passive_bomb: is_passive_bomb,
                            SLNetwork.did_passive_response: did_passive_response,
                            SLNetwork.passive_decision_input: passive_decision_input,
                            SLNetwork.passive_response_input: passive_response_input,
                            SLNetwork.passive_bomb_input: passive_bomb_input,
                            SLNetwork.active_decision_input: active_decision_input,
                            SLNetwork.active_response_input: active_response_input,
                            SLNetwork.has_seq_length: has_seq_length,
                            SLNetwork.seq_length_input: seq_length_input
                })

                if minor_cards_targets is not None:
                    main_cards = pick_main_cards(category_idx, to_char(intention))
                    accs = train_fake_action(minor_cards_targets, curr_cards_char.copy(), s.copy(), sess, SLNetwork, category_idx, main_cards)
                    for acc in accs:
                        logger.updateAcc("minor_cards", acc)

                #print("gradients ： ", gradients)

                # update accuracies
                # if has_minor_cards[0]:
                #     # print("minor_cards_output : ", minor_cards_output)
                #     # print("minor_cards_output.argsort : ", minor_cards_output.argsort()[:int(minor_cards_length[0])])
                #     # print("minor_cards_target[0] : ", minor_cards_target[0])
                #     # print("minor_cards_target[0].argsort : ", minor_cards_target[0].argsort()[:int(minor_cards_length[0])])
                #     minor_cards_acc_temp = 1 if np.array_equal(minor_cards_output.argsort()[:int(minor_cards_length[0])][0], \
                #         minor_cards_target.argsort()[:int(minor_cards_length[0])][0]) else 0
                #     logger.updateAcc("minor_cards", minor_cards_acc_temp)
                if active:
                    active_decision_acc_temp = 1 if np.argmax(decision_active_output) == active_decision_input[0] else 0
                    logger.updateAcc("active_decision", active_decision_acc_temp)

                    active_response_acc_temp = 1 if np.argmax(response_active_output) == active_response_input[0] else 0
                    logger.updateAcc("active_response", active_response_acc_temp)

                    if has_seq_length[0]:
                        seq_acc_temp = 1 if np.argmax(seq_length_output) == seq_length_input[0] else 0
                        logger.updateAcc("seq_length", seq_acc_temp)
                else:
                    passive_decision_acc_temp = 1 if np.argmax(decision_passive_output) == passive_decision_input[0] else 0
                    logger.updateAcc("passive_decision", passive_decision_acc_temp)

                    if is_passive_bomb[0]:
                        passive_bomb_acc_temp = 1 if np.argmax(bomb_passive_output) == passive_bomb_input[0] else 0
                        logger.updateAcc("passive_bomb", passive_bomb_acc_temp)
                    elif passive_decision_input[0] == 3:
                        passive_response_acc_temp = 1 if np.argmax(response_passive_output) == passive_response_input[0] else 0
                        logger.updateAcc("passive_response", passive_response_acc_temp)

                curr_cards_value = e.get_curr_handcards()
                curr_cards_char = to_char(curr_cards_value)
                last_cards_value = e.get_last_outcards()
                last_cards_char = to_char(last_cards_value)
                # print("curr_cards = ", curr_cards_value)
                # print("last_cards = ", last_cards_value)
                if last_cards_value.size > 0:
                    last_category_idx = category_idx

                # mask = get_mask(curr_cards_char, action_space, last_cards_char)

                input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, last_cards_char if last_cards_value.size > 0 else None)
                input_single_last, input_pair_last, input_triple_last, input_quadric_last = get_masks(last_cards_char,
                                                                                                      None)

                s = e.get_state()
                s = np.reshape(s, [1, -1]).astype(np.float32)
            #print("End of one game")
            if i % 100 == 0:
                print("train1 ", i, " ing...")
                print("train passive decision accuracy = ", logger["passive_decision"])
                print("train passive response accuracy = ", logger["passive_response"])
                print("train passive bomb accuracy = ", logger["passive_bomb"])
                print("train active decision accuracy = ", logger["active_decision"])
                print("train active response accuracy = ", logger["active_response"])
                print("train sequence length accuracy = ", logger["seq_length"])
                print("train minor cards accuracy = ", logger["minor_cards"])
                summary = tf.Summary(value=[
                                tf.Summary.Value(tag="Accuracy/passive_decision_accuracy", simple_value=logger["passive_decision"]), 
                                tf.Summary.Value(tag="Accuracy/passive_response_accuracy", simple_value=logger["passive_response"]),
                                tf.Summary.Value(tag="Accuracy/passive_bomb_accuracy", simple_value=logger["passive_bomb"]),
                                tf.Summary.Value(tag="Accuracy/active_decision_accuracy", simple_value=logger["active_decision"]),
                                tf.Summary.Value(tag="Accuracy/active_response_accuracy", simple_value=logger["active_response"]),
                                tf.Summary.Value(tag="Accuracy/seq_length_accuracy", simple_value=logger["seq_length"]),
                                tf.Summary.Value(tag="Accuracy/minor_cards_accuracy", simple_value=logger["minor_cards"]),

                                #tf.Summary.Value(tag="minor_loss", simple_value=minor_loss)
                                tf.Summary.Value(tag="Loss/loss", simple_value=loss[0]),
                                tf.Summary.Value(tag="Loss/active_decision_loss", simple_value=active_decision_loss[0]),
                                tf.Summary.Value(tag="Loss/active_response_loss", simple_value=active_response_loss[0]),
                                tf.Summary.Value(tag="Loss/passive_decision_loss", simple_value=passive_decision_loss[0]),
                                tf.Summary.Value(tag="Loss/passive_response_loss", simple_value=passive_response_loss[0]),
                                tf.Summary.Value(tag="Loss/passive_bomb_loss", simple_value=passive_bomb_loss[0]) 
                                #tf.Summary.Value(tag="gradients", )
                ])
                file_writer.add_summary(summary, i / 100 - 1)
            if i % 500 == 0 and i > 0:
                saver.save(sess, "./Model/accuracy_fake_minor/model", global_step=i)

        
        # saver.save(sess, "./Model/SLNetwork_feat_deeper_1000000epoches.ckpt")

        print("train passive decision accuracy = ", logger["passive_decision"])
        print("train passive response accuracy = ", logger["passive_response"])
        print("train passive bomb accuracy = ", logger["passive_bomb"])
        print("train active decision accuracy = ", logger["active_decision"])
        print("train active response accuracy = ", logger["active_response"])
        print("train seq length accuracy = ", logger["seq_length"])
        print("train minor cards accuracy = ", logger["minor_cards"])

        file_writer.close()

    # RL part #################################################################
    # print('loading model...')
    # saver.restore(sess, "./Model/accuracy_bugfixed_lr0001_1110/model.ckpt")

    # graph_rl = tf.Graph()
    # with graph_rl.as_default():
    #     sess = tf.Session(graph = graph_rl)
    #     cardgame = env.Env()
    #     master = CardMaster(cardgame)
    #     master.run(sess, saver, 2000)
    # RL part #################################################################

    # test part
    # saver.restore(sess, "./Model/SLNetwork_feat_deeper_1000000epoches.ckpt")
    saver.restore(sess, "./Model/accuracy_fake_minor/model-9500")
    print(get_benchmark(sess, SLNetwork))
    # exit(0)

    if TRAIN:
        saver.restore(sess, "./Model/accuracy_fake_minor/model-9500")
        for i in range(epoches_test):
            e.reset()
            e.prepare()
        
            r = 0
            done = False
            while r == 0:
                curr_cards_value = e.get_curr_handcards()
                curr_cards_char = to_char(curr_cards_value)
                last_cards_value = e.get_last_outcards()
                last_cards_char = to_char(last_cards_value)
                # print("curr_cards = ", curr_cards_char)
                # print("last_cards = ", last_cards_value)
                last_category_idx = e.get_last_outcategory_idx()

                input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, last_cards_char if last_cards_value.size > 0 else None)
                input_single_last, input_pair_last, input_triple_last, input_quadric_last = get_masks(last_cards_char, None)

                s = e.get_state()
                s = np.reshape(s, [1, -1]).astype(np.float32)

                intention, r, category_idx = e.step_auto()
                # print(intention)

                is_passive_bomb = False
                has_seq_length = False
                seq_length = None
                is_active = (last_cards_value.size == 0)
                if is_active:
                    # first get mask
                    decision_mask, response_mask, _, length_mask = get_mask_alter(curr_cards_char, [], False, last_category_idx)

                    decision_active_output = sess.run(SLNetwork.fc_decision_active_output,
                        feed_dict={
                            SLNetwork.training: True,
                            SLNetwork.input_state: s,
                            SLNetwork.input_single: np.reshape(input_single, [1, -1]),
                            SLNetwork.input_pair: np.reshape(input_pair, [1, -1]),
                            SLNetwork.input_triple: np.reshape(input_triple, [1, -1]),
                            SLNetwork.input_quadric: np.reshape(input_quadric, [1, -1])
                        })

                    # make decision depending on output
                    decision_active_output = decision_active_output[0]
                    decision_active_output[decision_mask == 0] = -1
                    decision_active_pred = np.argmax(decision_active_output)
                    decision_active_target = category_idx - 1
                    
                    active_category_idx = decision_active_target + 1

                    # give actual response
                    response_active_output = sess.run(SLNetwork.fc_response_active_output,
                        feed_dict={
                            SLNetwork.training: True,
                            SLNetwork.input_state: s,
                            SLNetwork.input_single: np.reshape(input_single, [1, -1]),
                            SLNetwork.input_pair: np.reshape(input_pair, [1, -1]),
                            SLNetwork.input_triple: np.reshape(input_triple, [1, -1]),
                            SLNetwork.input_quadric: np.reshape(input_quadric, [1, -1])
                        })

                    response_active_output = response_active_output[0]
                    response_active_output[response_mask[decision_active_target] == 0] = -1
                    response_active_pred = np.argmax(response_active_output)
                    response_active_target = intention[0] - 3

                    seq_length = get_seq_length(category_idx, intention)
                    if seq_length is not None:
                        has_seq_length = True
                        seq_length_target = seq_length

                    # next sequence length
                    if active_category_idx == Category.SINGLE_LINE.value or \
                            active_category_idx == Category.DOUBLE_LINE.value or \
                            active_category_idx == Category.TRIPLE_LINE.value or \
                            active_category_idx == Category.THREE_ONE_LINE.value or \
                            active_category_idx == Category.THREE_TWO_LINE.value:
                        seq_length_output = sess.run(SLNetwork.fc_sequence_length_output,
                            feed_dict={
                                SLNetwork.training: True,
                                SLNetwork.input_state: s,
                                SLNetwork.input_single: np.reshape(input_single, [1, -1]),
                                SLNetwork.input_pair: np.reshape(input_pair, [1, -1]),
                                SLNetwork.input_triple: np.reshape(input_triple, [1, -1]),
                                SLNetwork.input_quadric: np.reshape(input_quadric, [1, -1])
                            })

                        seq_length_output = seq_length_output[0]
                        seq_length_output[length_mask[decision_active_target][response_active_target] == 0] = -1
                        seq_length_pred = np.argmax(seq_length_output) + 1
                    
                else:
                    is_bomb = False
                    if len(last_cards_value) == 4 and len(set(last_cards_value)) == 1:
                        is_bomb = True
                    decision_mask, response_mask, bomb_mask, _ = get_mask_alter(curr_cards_char, to_char(last_cards_value), is_bomb, last_category_idx)

                    decision_passive_output, response_passive_output, bomb_passive_output \
                        = sess.run([SLNetwork.fc_decision_passive_output,
                                    SLNetwork.fc_response_passive_output, SLNetwork.fc_bomb_passive_output],
                                    feed_dict={
                                        SLNetwork.training: True,
                                        SLNetwork.input_state: s,
                                        SLNetwork.input_single: np.reshape(input_single, [1, -1]),
                                        SLNetwork.input_pair: np.reshape(input_pair, [1, -1]),
                                        SLNetwork.input_triple: np.reshape(input_triple, [1, -1]),
                                        SLNetwork.input_quadric: np.reshape(input_quadric, [1, -1]),
                                        SLNetwork.input_single_last: np.reshape(input_single_last, [1, -1]),
                                        SLNetwork.input_pair_last: np.reshape(input_pair_last, [1, -1]),
                                        SLNetwork.input_triple_last: np.reshape(input_triple_last, [1, -1]),
                                        SLNetwork.input_quadric_last: np.reshape(input_quadric_last, [1, -1])
                                    })
                    
                    decision_passive_target = 0
                    if category_idx == Category.QUADRIC.value and category_idx != last_category_idx:
                        is_passive_bomb = True
                        decision_passive_target = 1
                        bomb_passive_target = intention[0] - 3
                    else:
                        if category_idx == Category.BIGBANG.value:
                            decision_passive_target = 2
                        else:
                            if category_idx != Category.EMPTY.value:
                                decision_passive_target = 3
                                response_passive_target = intention[0] - 3
                    
                    decision_passive_output = decision_passive_output[0]
                    decision_passive_output[decision_mask == 0] = -1
                    decision_passive_pred = np.argmax(decision_passive_output)

                    if decision_passive_target == 0:
                        pass
                    elif decision_passive_target == 1:
                        # print('bomb_mask', bomb_mask)
                        bomb_passive_output = bomb_passive_output[0]
                        bomb_passive_output[bomb_mask == 0] = -1
                        bomb_passive_pred = np.argmax(bomb_passive_output)
                    elif decision_passive_target == 2:
                        pass
                    elif decision_passive_target == 3:
                        # print('response_mask', response_mask)
                        response_passive_output = response_passive_output[0]
                        response_passive_output[response_mask == 0] = -1
                        response_passive_pred = np.argmax(response_passive_output)

                    if decision_mask[decision_passive_target] == 0:
                        raise Exception('decision mask fault')

                    if decision_passive_target == 3:
                        if response_mask[response_passive_target] == 0:
                            print(to_char(intention))
                            print(last_cards_char)
                            raise Exception('response mask fault')

                minor_cards_targets = pick_minor_targets(category_idx, to_char(intention))
                if minor_cards_targets is not None:
                    main_cards = pick_main_cards(category_idx, to_char(intention))
                    # print(minor_cards_targets)
                    dup_mask = np.ones([15])
                    seq_length = get_seq_length(category_idx, intention)
                    if seq_length:
                        for j in range(seq_length):
                            dup_mask[intention[0] - 3 + j] = 0
                    else:
                        dup_mask[intention[0] - 3] = 0
                    accs = test_fake_action(minor_cards_targets, curr_cards_char.copy(), s.copy(), sess, SLNetwork, category_idx, dup_mask, main_cards)
                    for acc in accs:
                        logger.updateAcc("minor_cards", acc)

                if is_active:
                    active_decision_acc_temp = 1 if decision_active_pred == decision_active_target else 0
                    logger.updateAcc("active_decision", active_decision_acc_temp)

                    active_response_acc_temp = 1 if response_active_pred == response_active_target else 0
                    logger.updateAcc("active_response", active_response_acc_temp)

                    if has_seq_length:
                        seq_acc_temp = 1 if seq_length_pred == seq_length_target else 0
                        logger.updateAcc("seq_length", seq_acc_temp)
                else:
                    passive_decision_acc_temp = 1 if decision_passive_pred == decision_passive_target else 0
                    logger.updateAcc("passive_decision", passive_decision_acc_temp)

                    if is_passive_bomb:
                        passive_bomb_acc_temp = 1 if bomb_passive_pred == bomb_passive_target else 0
                        logger.updateAcc("passive_bomb", passive_bomb_acc_temp)
                    elif decision_passive_target == 3:
                        passive_response_acc_temp = 1 if response_passive_pred == response_passive_target else 0
                        logger.updateAcc("passive_response", passive_response_acc_temp)
            
            if i % 100 == 0:
                print("test ", i, " ing...")
                print("test passive decision accuracy = ", logger["passive_decision"])
                print("test passive response accuracy = ", logger["passive_response"])
                print("test passive bomb accuracy = ", logger["passive_bomb"])
                print("test active decision accuracy = ", logger["active_decision"])
                print("test active response accuracy = ", logger["active_response"])
                print("test sequence length accuracy = ", logger["seq_length"])
                print("test minor cards accuracy = ", logger["minor_cards"])


