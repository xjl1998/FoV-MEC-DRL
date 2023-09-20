from os.path import abspath, dirname
import numpy as np
import tensorflow as tf
from DRL.TFAgent.Prioritized_Replay import Memory

tf.set_random_seed(1)
np.random.seed(1)

'''
DQN off-policy
使用CNN方法实现多维度动作
'''
class DqnAgentMD:
    def __init__(
            self,
            dim_actions,#
            n_actions, # 可以看作动作空间的统计个数
            n_features, #其实是state
            learning_rate=0.02,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,#表示多少次更新目标权重
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            name_str = '',
            double_q = False,
            prioritized_replay = False,
    ):
        '''
        :param dim_actions: 动作空间的维度，至少为1维
        :param n_actions: 可以选择的离散动作个数
        :param n_features: state的长度
        :param learning_rate: 学习率
        :param reward_decay:
        :param e_greedy: 采用e-greedy时需要设置的epsilon
        :param replace_target_iter: evaluate-network更新多少次后，复制evaluate-network的权重给target-network
        :param memory_size: 重播队列
        :param batch_size: 一次训练多少批
        :param e_greedy_increment:
        :param output_graph:
        :param name_str:
        :param double_q: 是否为double-Dqn
        :param prioritized_replay: 是否采用prioritized_replay这一经验重放策略
        '''



        self.params = {
            'dim_actions':dim_actions,
            'n_actions': n_actions,
            'n_features': n_features,
            'learning_rate': learning_rate,
            'reward_decay': reward_decay,
            'e_greedy': e_greedy,
            'replace_target_iter': replace_target_iter,
            'memory_size': memory_size,
            'batch_size': batch_size,
            'e_greedy_increment': e_greedy_increment,
            'output_graph':output_graph,
            'name_str':name_str,
            'double_q':double_q,
            'prioritized_replay':prioritized_replay,
        }

        self.actions = n_actions
        # self.actions_f = actions_f
        self.double_q = double_q
        self.prioritized_replay = prioritized_replay
        # self.actions_space = actions_space
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.name_str = name_str
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        if self.prioritized_replay:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 1 + 1),dtype=complex)

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net'+self.name_str)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net'+self.name_str)
        with tf.variable_scope('hard_replacement'+self.name_str):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # def weight_assign():
        #     self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        tf.reset_default_graph()
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(shape=(None, self.n_features), dtype=tf.complex64, name='s'+self.name_str)  # input State
        self.s_ = tf.placeholder(shape=(None, self.n_features), dtype=tf.complex64, name='s_'+self.name_str)
        self.r = tf.placeholder(shape=(None,), dtype=tf.float32, name='r'+self.name_str)
        self.a = tf.placeholder(shape=(None,self.params['dim_actions']), dtype=tf.int32, name='a'+self.name_str)
        self.qwe = tf.placeholder(shape=(None,self.actions ), dtype=tf.float32, name='double_q_next'+self.name_str)
        # self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        # self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        # self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        if self.prioritized_replay:
            self.ISWeights = tf.compat.v1.placeholder(tf.float32, [None, 1], name='IS_weights')
        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'+self.name_str):
            e1 =tf.layers.dense(self.s,128, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1'+self.name_str)
            e2 = tf.layers.dense(e1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2'+self.name_str)
            # e1 =tf.layers.dense(self.s,256, tf.nn.relu, kernel_initializer=w_initializer,
            #                     bias_initializer=b_initializer, name='e1'+self.name_str)
            # e2 = tf.layers.dense(e1, 128, tf.nn.relu, kernel_initializer=w_initializer,
            #                  bias_initializer=b_initializer, name='e2'+self.name_str)
            self.q_eval = tf.layers.dense(e2, self.actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q'+self.name_str)#(?,actions_f)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'+self.name_str):
            t1 = tf.layers.dense(self.s_, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1'+self.name_str)
            t2 = tf.layers.dense(t1, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2'+self.name_str)
            # t1 = tf.layers.dense(self.s_, 256, tf.nn.relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='t1'+self.name_str)
            # t2 = tf.layers.dense(t1, 128, tf.nn.relu, kernel_initializer=w_initializer,
            #                      bias_initializer=b_initializer, name='t2'+self.name_str)
            self.q_next = tf.layers.dense(t2, self.actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_'+self.name_str)

        with tf.variable_scope('q_target'+self.name_str):
            if self.double_q:
                index = tf.argmax(self.qwe, axis=1, name='Qmax_s_'+self.name_str)
                index = tf.to_int32(index)
                q_indices = tf.stack([tf.range(tf.shape(index)[0]), index], axis=1)
                q_next_v = tf.gather_nd(params=self.q_next, indices= q_indices)
                q_target = self.r + self.gamma * q_next_v
            else:
                q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_'+self.name_str)    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'+self.name_str):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0]), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )索引切片
        with tf.variable_scope('loss'+self.name_str):
            if self.prioritized_replay:
                self.abs_errors = tf.abs(self.q_target - self.q_eval_wrt_a)   # for updating Sumtree
                #tf.squared_difference(x,y)=(x-y)的平方
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval_wrt_a), name='TD_error'+self.name_str)
            else:
                # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
                self.loss = tf.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'+self.name_str))#先求差的平方，然后取均值
        with tf.variable_scope('train'+self.name_str):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    # GradientDescentOptimizer, AdagradOptimizer, MomentumOptimizer.RMSPropOptimizer，AdamOptimizer
    def store_transition(self, s, a, r, s_): #记忆存储
        if self.prioritized_replay:
            # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)
        else:
            if not hasattr(self, 'memory_counter'): #
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            if self.memory_counter < self.memory_size:
                index = self.memory_counter % self.memory_size
            else:
                index = self.memory_counter % self.memory_size
            # replace the old memory with new memory
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):# channe动作选择
        # to have batch dimension when feed into tf placeholder
        # observation = observation[np.newaxis, :]#多加了一个维度
        action = []
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: np.transpose(observation).reshape(-1,len(observation))})#每个动作的价值
            action = np.argmax(actions_value) #返回具有最大价值的动作序号
        else:
            action = np.random.randint(0, self.actions)
        return action

    def learn(self,sample_index):
        if self.prioritized_replay:
            tree_idx, batch_memory, ISWeights,sample_index = self.memory.sample(self.batch_size)
        else:
            if len(sample_index) == 0:
                if self.memory_counter <= self.batch_size:
                    return []
                if self.memory_counter >= self.memory_size:
                    sample_index = np.random.choice(self.memory_size, size=self.batch_size)
                if self.batch_size < self.memory_counter < self.memory_size:
                    sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            else:
                sample_index =sample_index
            # max_r = max(self.memory[:, self.n_features + 1])
            # max_r = 250
            batch_memory = self.memory[sample_index, :]
            # batch_memory[:, self.n_features + 1] = batch_memory[:, self.n_features + 1]/max_r
        for e in range(1):
            if self.prioritized_replay is True and self.double_q is False:
                _, abs_errors, cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                    feed_dict={self.s: batch_memory[:, :self.n_features],
                                                               self.a: batch_memory[:, self.n_features],
                                                               self.r: batch_memory[:, self.n_features + 1],
                                                               self.s_: batch_memory[:, -self.n_features:],
                                                               # self.q_target: q_target,

                                                               self.ISWeights: ISWeights})
                self.memory.batch_update(tree_idx, abs_errors)     # update priority
            elif self.double_q is True and self.prioritized_replay is False:
                q_eval_double = self.sess.run(self.q_eval,feed_dict={
                    self.s: batch_memory[:, -self.n_features:],  #

                })

                _, cost = self.sess.run(
                    [self._train_op, self.loss],
                    feed_dict={
                        self.s: batch_memory[:, :self.n_features],
                        self.a: batch_memory[:, self.n_features],
                        self.r: batch_memory[:, self.n_features + 1],
                        self.s_: batch_memory[:, -self.n_features:],
                        self.qwe : q_eval_double,
                    })
            elif self.double_q  and self.prioritized_replay :
                q_eval_double = self.sess.run(self.q_eval,feed_dict={
                    self.s: batch_memory[:, -self.n_features:],  #
                })
                _, abs_errors, cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                    feed_dict={self.s: batch_memory[:, :self.n_features],
                                                               self.a: batch_memory[:, self.n_features],
                                                               self.r: batch_memory[:, self.n_features + 1],
                                                               self.s_: batch_memory[:, -self.n_features:],
                                                               # self.q_target: q_target,
                                                               self.qwe : q_eval_double,
                                                               self.ISWeights: ISWeights})
                self.memory.batch_update(tree_idx, abs_errors)     # update priority
            else:
                _, cost = self.sess.run(
                    [self._train_op, self.loss],
                    feed_dict={
                        self.s: batch_memory[:, :self.n_features],
                        self.a: batch_memory[:, self.n_features],
                        self.r: batch_memory[:, self.n_features + 1],
                        self.s_: batch_memory[:, -self.n_features:],
                    })
        self.cost_his.append(cost)
        if self.learn_step_counter % self.replace_target_iter == 0: # 更新权重值
            self.sess.run(self.target_replace_op)
        if self.epsilon <= self.epsilon_max:
            self.epsilon += self.epsilon_increment
        else:
            self.epsilon =self.epsilon_max
        self.learn_step_counter += 1
        return sample_index
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost'+self.name_str)
        plt.xlabel('training steps'+self.name_str)
        plt.title('Cost'+self.name_str)
        # plt.show()

    def save_model(self):
        dir_path = dirname(abspath(__file__)) + '\\model'
        self.eval_model.save_weights(dir_path+'\\eval_weights_'+self.params['name_str']+'.h5')
        self.eval_model.save_weights(dir_path+'\\target_weights_'+self.params['name_str']+'.h5')
    def clear_sess(self):
        self.sess.close()
        tf.reset_default_graph()