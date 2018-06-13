import numpy as np
import pandas as pd
import tensorflow as tf

# 设定随机种子，用以复现结果
np.random.seed(1)
tf.set_random_seed(1)


# DQN 类
class DeepQNetwork(object):
    # 初始化参数
    def __init__(self,
                 n_actions,                 # 选择的个数
                 n_features,                # 特征的个数
                 learning_rate=0.01,        # 学习率
                 reward_decay=0.9,          # 影响因子，gamma
                 e_greedy=0.9,              # 变异因子？
                 replace_target_iter=300,   # 新旧参数更迭的代数
                 memory_size=500,           # 记忆大小
                 batch_size=32,             #
                 e_greedy_increment=None,   # 渐变的e-greedy算法
                 output_graph=False):       # 是否适用tensorboard可视化网络结构
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()   # 构建网络

        t_params = tf.get_collection('target_net_params')   # 获取旧网络参数
        e_params = tf.get_collection('eval_net_params')     # 获取新网络参数
        self.replace_target_op = [tf.assign(t, e)
                                  for t, e in zip(t_params, e_params)]  # 更新旧网络参数

        self.sess = tf.Session()    # 把tf的Session作为类的一个成员，初始化之后就可以直接使用了

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph) # 可视化网络结构

        self.sess.run(tf.global_variables_initializer())    # 初始化网络参数
        self.cost_his = []                                  # 用来可视化loss的

    ## 构建网络
    def _build_net(self):
        ######
        # 主要神经网络结构
        # s --> NN --> q_eval -->\
        #                         ==>loss
        # q_target ------------->/
        ######
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(
            tf.float32, [None, self.n_actions], name='Q_target')

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(
                    0., 0.3), tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(
                self.lr).minimize(self.loss)

        ######
        # 辅助神经网络结构，不进行训练，只前向传递，用来计算主神经网络中的q_target
        # s_ --> old_NN --> q
        # q_target = reward + gamma*max(q)
        ######
        self.s_ = tf.placeholder(
            tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable(
                    'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable(
                    'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable(
                    'w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable(
                    'b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2


    # DQN的两大特色之一，存储，（也许用队列更好？？）
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))         # 大小 n_features+2+n_features
        index = self.memory_counter % self.memory_size  # 如果用队列，可以保证所有存储按时间顺序排列
        self.memory[index, :] = transition

        self.memory_counter += 1

    ## 做出选择
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]    # 维度加1，相当于把某一个s变成含有这个s的size为1的batch，方便传入NN

        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(
                self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    ## 参数更新的主要过程
    def learn(self):

        # 是否把主NN的参数更新到辅助NN
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 现在看来既然是随机取memory，那就不用队列
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 通过NN获取Q值
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],    # 下一状态，参见store_transition
                self.s: batch_memory[:, :self.n_features]       # 当前状态，参见store_transition
            }
        )

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)   # action
        reward = batch_memory[:, self.n_features + 1]                   # reward

        q_target[batch_index, eval_act_index] = reward + \
            self.gamma * np.max(q_next, axis=1)                         # 依据Q-Learning算法构建q_target

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={
            self.s: batch_memory[:, :self.n_features],
            self.q_target: q_target
        })

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # 损失曲线，没懂。。。
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
