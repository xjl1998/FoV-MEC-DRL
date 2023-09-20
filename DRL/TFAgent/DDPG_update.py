"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np
import time
tf.set_random_seed(1)
np.random.seed(1)


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
# LR_A = 0.001    # learning rate for actor
# LR_C = 0.002    # learning rate for critic
GAMMA = 0.9 #0.99 reward discount
TAU = 0.2   #0.0001 soft replacement
# MEMORY_CAPACITY = 10000
# BATCH_SIZE = 64
#
# RENDER = False
# ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################

class DDPGUP(object):
    def __init__(self, a_dim, s_dim, a_bound,MEMORY_CAPACITY,BATCH_SIZE,name_str,replace_iter,L_A,L_C):

        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE=BATCH_SIZE
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=complex)
        self.pointer = 0
        self.sess = tf.Session()

        self.learn_step_counter = 0
        self.name_str = name_str
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's'+self.name_str)
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_'+self.name_str)
        self.R = tf.placeholder(tf.float32, [None, 1], 'r'+self.name_str)
        self.replace_target_iter = replace_iter

        self.L_A = L_A
        self.L_C = L_C
        self.cost_his = []

        with tf.variable_scope('Actor'+name_str):
            self.a = self._build_a(self.S, scope='eval'+name_str, trainable=True)
            self.a_ = self._build_a(self.S_, scope='target'+name_str, trainable=False)
        with tf.variable_scope('Critic'+name_str):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, scope='eval'+name_str, trainable=True)
            self.q_ = self._build_c(self.S_,self.a_, scope='target'+name_str, trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor'+name_str+'/eval'+name_str)
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor'+name_str+'/target'+name_str)
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'+name_str+'/eval'+name_str)
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic'+name_str+'/target'+name_str)

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
        self.a_loss = - tf.reduce_mean(self.q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.L_A).minimize(self.a_loss, var_list=self.ae_params)

        self.q_target = self.R + GAMMA * self.q_
        # in the feed_dic for the td_error, the self.a should change to actions in memoryAdamOptimizer
        self.td_error = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q)
        self.ctrain = tf.train.AdamOptimizer(self.L_C).minimize(self.td_error, var_list=self.ce_params)
        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        # print(self.sess.run(self.a, {self.S: s}))
        # return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        # print(np.transpose(s).reshape(-1,len(s)))
        test = np.transpose(s).reshape(-1,len(s))
        test2 = self.sess.run(self.a, {self.S: np.transpose(s).reshape(-1,len(s))})
        return self.sess.run(self.a, {self.S: np.transpose(s).reshape(-1,len(s))})#self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]


    def learn(self,sample_index):
        if len(sample_index) == 0:
            if self.BATCH_SIZE<=self.pointer<self.MEMORY_CAPACITY:
                indices = np.random.choice(self.pointer,size=self.BATCH_SIZE)
            if self.pointer>=self.MEMORY_CAPACITY:
                indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
            if self.pointer < self.BATCH_SIZE:
                return []
        else:
            indices = sample_index
        for cou in range(1):
            bt = self.memory[indices, :]
            # max_r = max(self.memory[:, self.s_dim + 1])
            # max_r = 250
            # bt[:, -self.s_dim - 1: -self.s_dim] = bt[:, -self.s_dim - 1: -self.s_dim]/max_r
            bs = bt[:, :self.s_dim]
            ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
            br = bt[:, -self.s_dim - 1: -self.s_dim]
            bs_ = bt[:, -self.s_dim:]
            self.sess.run(self.atrain, {self.S: bs})
            _,cost=self.sess.run([self.ctrain,self.td_error], {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
            # soft target replacement
        if self.learn_step_counter % self.replace_target_iter == 0: # 更新权重值
            self.sess.run(self.soft_replace)
        self.cost_his.append(cost)
        self.learn_step_counter += 1
        return sample_index


    def store_transition(self, s,a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 256, activation=tf.nn.tanh, name='l1'+self.name_str, trainable=trainable)
            net1 = tf.layers.dense(net, 128, activation=tf.nn.tanh, name='l2'+self.name_str, trainable=trainable)
            # drop_layer = tf.layers.dropout(net1,0.3)
            net2 = tf.layers.dense(net1, 64, activation=tf.nn.tanh, name='l3'+self.name_str, trainable=trainable)
            a = tf.layers.dense(net2, self.a_dim, activation=tf.nn.tanh, name='a'+self.name_str, trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a'+self.name_str)

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 128
            w1_s = tf.get_variable('w1_s'+self.name_str, [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a'+self.name_str, [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1'+self.name_str, [1, n_l1], trainable=trainable)
            net = tf.nn.tanh(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # net2 = tf.layers.dense(net, 128, activation=tf.nn.tanh, name='l4', trainable=trainable)
            # drop_layer = tf.layers.dropout(net2,0.3)
            net1 = tf.layers.dense(net, 64, activation=tf.nn.tanh, name='l3', trainable=trainable)
            return tf.layers.dense(net1, 1, trainable=trainable)  # Q(s,a)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost'+self.name_str)
        plt.xlabel('training steps'+self.name_str)
        plt.title(self.name_str)
        plt.show()

    def clear_sess(self):
        self.sess.close()
        tf.reset_default_graph()
###############################  training  ####################################

# env = gym.make(ENV_NAME)
# env = env.unwrapped
# env.seed(1)
#
# s_dim = env.observation_space.shape[0]
# a_dim = env.action_space.shape[0]
# a_bound = env.action_space.high
#
# ddpg = DDPG(a_dim, s_dim, a_bound)
#
# var = 3  # control exploration
# t1 = time.time()
# for i in range(MAX_EPISODES):
#     s = env.reset()
#     ep_reward = 0
#     for j in range(MAX_EP_STEPS):
#         if RENDER:
#             env.render()
#
#         # Add exploration noise
#         a = ddpg.choose_action(s)
#         a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
#         s_, r, done, info = env.step(a)
#
#         ddpg.store_transition(s, a, r / 10, s_)
#
#         if ddpg.pointer > MEMORY_CAPACITY:
#             var *= .9995    # decay the action randomness
#             ddpg.learn()
#
#         s = s_
#         ep_reward += r
#         if j == MAX_EP_STEPS-1:
#             print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
#             # if ep_reward > -300:RENDER = True
#             break
# print('Running time: ', time.time() - t1)
