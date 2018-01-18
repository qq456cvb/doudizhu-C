from network_SL import CardNetwork
import tensorflow as tf
import numpy as np
import sys
from card import Category
from utils import get_mask_alter, get_masks, to_char, to_value, get_seq_length, give_cards_without_minor
sys.path.insert(0, './build/Release')
import env


def read_cards_input():
    intention = sys.stdin.readline()
    if intention == 'q':
        exit()
    intention = intention.split()
    return np.array(to_value(intention))


# return char minor cards output
def inference_minor_util(s, handcards, sess, network, num, is_pair, dup_mask):
    outputs = []
    input_single, input_pair, input_triple, input_quadric = get_masks(handcards, None)
    for i in range(num):
        response_active_output = sess.run(network.fc_response_active_output,
                feed_dict = {
                    network.training: True,
                    network.input_state: s,
                    network.input_single: np.reshape(input_single, [1, -1]),
                    network.input_pair: np.reshape(input_pair, [1, -1]),
                    network.input_triple: np.reshape(input_triple, [1, -1]),
                    network.input_quadric: np.reshape(input_quadric, [1, -1])
            })
        
        # give minor cards
        response_active_output = response_active_output[0]
        response_active_output[dup_mask == 0] = -1
        # print(handcards)
        if is_pair:
            response_active_output[input_pair == 0] = -1
        else:
            response_active_output[input_single == 0] = -1
        
        response_active = np.argmax(response_active_output)
        dup_mask[response_active] = 0

        # convert network output to char cards
        handcards.remove(to_char(response_active + 3))
        if is_pair:
            handcards.remove(to_char(response_active + 3))

        # update mask for the next loop
        input_single, input_pair, input_triple, input_quadric = get_masks(handcards, None)

        # save to output
        outputs.append(to_char(response_active + 3))
        if is_pair:
            outputs.append(to_char(response_active + 3))
    return outputs


def inference_minor_cards(category, s, handcards, sess, network, seq_length, dup_mask):
    if category == Category.THREE_ONE.value:
        return inference_minor_util(s, handcards, sess, network, 1, False, dup_mask)
    if category == Category.THREE_TWO.value:
        return inference_minor_util(s, handcards, sess, network, 1, True, dup_mask)
    if category == Category.THREE_ONE_LINE.value:
        return inference_minor_util(s, handcards, sess, network, seq_length, False, dup_mask)
    if category == Category.THREE_TWO_LINE.value:
        return inference_minor_util(s, handcards, sess, network, seq_length, True, dup_mask)
    if category == Category.FOUR_TWO.value:
        return inference_minor_util(s, handcards, sess, network, 2, False, dup_mask)


if __name__ == '__main__':
    g = tf.get_default_graph()
    network = CardNetwork(54 * 6, tf.train.AdamOptimizer(learning_rate=0.0001), "SLNetwork")
    saver = tf.train.Saver()

    e = env.Env()
    player_id = 1
    with tf.Session() as sess:
        saver.restore(sess, './Model/accuracy_fake_minor/model-9800')
        for i in range(10):
            e.reset()
            e.prepare()
        
            r = 0
            done = False
            while not done:
                idx = e.get_role_ID()
                if idx == player_id:
                    print("current handcards: ", to_char(e.get_curr_handcards()))
                    intention = read_cards_input()
                else:
                
                    curr_cards_value = e.get_curr_handcards()
                    curr_cards_char = to_char(curr_cards_value)
                    last_cards_value = e.get_last_outcards()
                    # print("curr_cards = ", curr_cards_char)
                    # print("last_cards = ", last_cards_value)
                    last_category_idx = e.get_last_outcategory_idx()
                    last_cards_char = to_char(last_cards_value)
                    # mask = get_mask(curr_cards_char, action_space, last_cards_char)

                    input_single, input_pair, input_triple, input_quadric = get_masks(curr_cards_char, last_cards_char if last_cards_value.size > 0 else None)

                    s = e.get_state()
                    s = np.reshape(s, [1, -1])

                    is_active = (last_cards_value.size == 0)
                    if is_active:
                        # first get mask
                        decision_mask, response_mask, _, length_mask = get_mask_alter(curr_cards_char, [], False, last_category_idx)

                        decision_active_output = sess.run(network.fc_decision_active_output,
                            feed_dict={
                                network.training: True,
                                network.input_state: s,
                                network.input_single: np.reshape(input_single, [1, -1]),
                                network.input_pair: np.reshape(input_pair, [1, -1]),
                                network.input_triple: np.reshape(input_triple, [1, -1]),
                                network.input_quadric: np.reshape(input_quadric, [1, -1])
                            })

                        # make decision depending on output
                        decision_active_output = decision_active_output[0]
                        decision_active_output[decision_mask == 0] = -1
                        decision_active = np.argmax(decision_active_output)
                        
                        active_category_idx = decision_active + 1

                        # give actual response
                        response_active_output = sess.run(network.fc_response_active_output,
                            feed_dict={
                                network.training: True,
                                network.input_state: s,
                                network.input_single: np.reshape(input_single, [1, -1]),
                                network.input_pair: np.reshape(input_pair, [1, -1]),
                                network.input_triple: np.reshape(input_triple, [1, -1]),
                                network.input_quadric: np.reshape(input_quadric, [1, -1])
                            })

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
                            seq_length_output = sess.run(network.fc_sequence_length_output,
                                feed_dict={
                                    network.training: True,
                                    network.input_state: s,
                                    network.input_single: np.reshape(input_single, [1, -1]),
                                    network.input_pair: np.reshape(input_pair, [1, -1]),
                                    network.input_triple: np.reshape(input_triple, [1, -1]),
                                    network.input_quadric: np.reshape(input_quadric, [1, -1])
                                })

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
                            # TODO: dup mask incorrect for sequence output
                            dup_mask[intention[0] - 3] = 0
                            intention = np.concatenate([intention, to_value(inference_minor_cards(active_category_idx, s, curr_cards_char.copy(), sess, network, seq_length, dup_mask))])
                    else:
                        is_bomb = False
                        if len(last_cards_value) == 4 and len(set(last_cards_value)) == 1:
                            is_bomb = True
                        # print(to_char(last_cards_value), is_bomb, last_category_idx)
                        decision_mask, response_mask, bomb_mask, _ = get_mask_alter(curr_cards_char, to_char(last_cards_value), is_bomb, last_category_idx)

                        decision_passive_output, response_passive_output, bomb_passive_output \
                            = sess.run([network.fc_decision_passive_output,
                                        network.fc_response_passive_output, network.fc_bomb_passive_output],
                                        feed_dict={
                                            network.training: True,
                                            network.input_state: s,
                                            network.input_single: np.reshape(input_single, [1, -1]),
                                            network.input_pair: np.reshape(input_pair, [1, -1]),
                                            network.input_triple: np.reshape(input_triple, [1, -1]),
                                            network.input_quadric: np.reshape(input_quadric, [1, -1])
                                        })
                        
                        # print(decision_mask)
                        # print(decision_passive_output)
                        decision_passive_output = decision_passive_output[0]
                        decision_passive_output[decision_mask == 0] = -1
                        decision_passive = np.argmax(decision_passive_output)
                        

                        if decision_passive == 0:
                            intention = np.array([])
                        elif decision_passive == 1:
                            is_passive_bomb = True
                            # print('bomb_mask', bomb_mask)
                            bomb_passive_output = bomb_passive_output[0]
                            bomb_passive_output[bomb_mask == 0] = -1
                            bomb_passive = np.argmax(bomb_passive_output)

                            # converting 0-based index to 3-based value
                            intention = np.array([bomb_passive + 3] * 4)

                        elif decision_passive == 2:
                            is_passive_king = True
                            intention = np.array([16, 17])
                        elif decision_passive == 3:
                            # print('response_mask', response_mask)
                            response_passive_output = response_passive_output[0]
                            response_passive_output[response_mask == 0] = -1
                            response_passive = np.argmax(response_passive_output)

                            # there is an offset when converting from 0-based index to 1-based index
                            bigger = response_passive + 1

                            intention = give_cards_without_minor(bigger, last_cards_value, last_category_idx, None)
                            if last_category_idx == Category.THREE_ONE.value or \
                                    last_category_idx == Category.THREE_TWO.value or \
                                    last_category_idx == Category.THREE_ONE_LINE.value or \
                                    last_category_idx == Category.THREE_TWO_LINE.value or \
                                    last_category_idx == Category.FOUR_TWO.value:
                                dup_mask = np.ones([15])
                                dup_mask[intention[0] - 3] = 0
                                intention = np.concatenate([intention, to_value(inference_minor_cards(last_category_idx, s, curr_cards_char.copy(), 
                                    sess, network, get_seq_length(last_category_idx, last_cards_value), dup_mask))])
                    
                print("idx %d intention is " % idx, end='')
                print(to_char(intention))
                r, done, category_idx = e.step_manual(intention)
                # intention, r, category_idx = e.step_auto()
                # print("auto intention is ", intention)
            if idx == player_id:
                print("YOU WIN!")
            else:
                print("YOU LOSE!")




