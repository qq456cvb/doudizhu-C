import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.slim as slim
import numpy as np


class RNNNetwork:
    def __init__(self, scope):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        with tf.variable_scope(scope):
            # 1 batch time, dynamic time step
            self.batch_size = tf.placeholder(tf.int32, None, name='batch_size')
            self.state_in = tf.placeholder(tf.float32, [1, None, 10], name='input')
            # 1 * t * 10
            # s = tf.one_hot(self.state_in, 10)
            s = self.state_in

            step_cnt = tf.shape(s)[1:2]
            s = tf.reshape(s, tf.stack([self.batch_size, -1, 10]))

            self.lstm = rnn.BasicLSTMCell(num_units=256, state_is_tuple=True)
            c_input = tf.placeholder(tf.float32, [1, self.lstm.state_size.c])
            h_input = tf.placeholder(tf.float32, [1, self.lstm.state_size.h])

            self.lstm_state_input = rnn.LSTMStateTuple(c_input, h_input)
            lstm_input = slim.fully_connected(inputs=s, num_outputs=64,
                                             activation_fn=tf.nn.relu)
            self.lstm_output, self.lstm_state_output = tf.nn.dynamic_rnn(self.lstm, lstm_input,
                                                                    initial_state=self.lstm_state_input,
                                                                    sequence_length=step_cnt)
            # size: 1 * t * 10
            self.policy_pred = slim.fully_connected(inputs=self.lstm_output, num_outputs=10,
                                             activation_fn=tf.nn.softmax)
            # action size: 1 * t
            self.action = tf.placeholder(shape=[None], dtype=tf.int32)
            self.action = tf.reshape(self.action, tf.stack([self.batch_size, -1]))
            self.action_onehot = tf.one_hot(self.action, 10, dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None, 1], dtype=tf.float32)

            # 1 * t prob
            self.pi_stoch = tf.reduce_sum(self.policy_pred * self.action_onehot, [2])

            # Loss functions
            # self.value_loss = tf.reduce_sum(tf.square(self.target_val - tf.reshape(self.val_pred, [-1])))
            # self.action_entropy = -tf.reduce_sum(self.policy_pred * tf.log(self.policy_pred))

            # -log(P(A) * P(B|A)...) = -log(P(A)) - log(P(B|A)) - ...
            self.policy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.pi_stoch, 1e-6, 1-1e-6)) * self.advantages)
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.policy_loss, local_vars)
            self.apply_grads = self.optimizer.apply_gradients(zip(self.gradients, local_vars))


if __name__ == '__main__':
    net = RNNNetwork('test')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # state = np.random.randint(10, size=(1, 1))

        for i in range(5000):
            seq_length = 5
            state = np.ones([1, 1, 10])
            state_train = np.zeros([1, seq_length, 10])
            action_train = np.zeros([1, seq_length])
            # init LSTM state
            c_init = np.zeros((1, net.lstm.state_size.c), np.float32)
            h_init = np.zeros((1, net.lstm.state_size.h), np.float32)
            rnn_state = [c_init, h_init]

            advantage = np.ones([1, seq_length])
            val = 0
            for j in range(seq_length):
                state_train[0, j] = state[0, 0]
                rnn_state_bk = rnn_state
                policy_output, rnn_state = sess.run([net.policy_pred, net.lstm_state_output], feed_dict={
                        net.state_in: state,
                        net.lstm_state_input: rnn_state,
                        net.batch_size: 1
                })
                p = policy_output[0, 0]
                if i % 100 == 0:
                    print(p[j], end=',')

                # a = np.random.choice(10, 1, p=p)[0]
                if np.random.rand() < 0.1:
                    a = np.random.choice(10, 1)[0]
                else:
                    a = np.random.choice(10, 1, p=p)[0]
                # using exploring action will make training biased
                # uncomment the following line will make training converges
                # a = j
                if a == j:
                    advantage[0, j] = 1
                    val += 1
                else:
                    advantage[0, j] = -1
                # if p[j] > 0.9:
                #     advantage[0, j] = 0
                action_train[0, j] = a
                # state[0, 0, a] -= 1

                # _, loss = sess.run([net.apply_grads, net.policy_loss], feed_dict={
                #             net.state_in: state,
                #             net.action: np.array([[a]]),
                #             net.lstm_state_input: rnn_state_bk,
                #             net.advantages: advantage
                #     })
                state[0, 0, a] -= 1

            if i % 100 == 0:
                print(' ')
            _, loss = sess.run([net.apply_grads, net.policy_loss], feed_dict={
                        net.state_in: state_train,
                        net.action: action_train,
                        net.lstm_state_input: [c_init, h_init],
                        net.advantages: np.array([[advantage[0, 0]]]),
                        net.batch_size: 1
                })
            if i % 100 == 0:
                print(loss)
                print("got value of: %d" % val)

