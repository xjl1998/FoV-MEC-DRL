

import argparse
import math
import time
from itertools import count

import os, sys, random
from os.path import exists, dirname, abspath

import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


from env import IrsNoma, reflect_calculate
from function_all import npysave
from torch_method.Class_DDPG import DDPG
from torch_method.torch_DQN import DQN_generate

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="jsac")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=20000, type=int) # replay buffer size
parser.add_argument('--capacity_DQN', default=10000, type=int)
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.5, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=30, type=int) # 迭代次数
parser.add_argument('--quantization_level', default=2, type=int)
args = parser.parse_args()

dir_path = dirname(abspath(__file__)) + '\\success_rate_result_converge——no_fix'
if not exists(dir_path):
    os.makedirs(dir_path)



def main(stat,level):
    args.quantization_level = level
    # args.capacity_DQN = int(level/2*10000)
    args.batch_size = int(level/2 * 100)
    args.max_episode = int(level/2) * 10000+ 100000
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 12000
    # args.learning_rate = lr

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * \
                                     math.exp(-1. * frame_idx / epsilon_decay)
    scene = IrsNoma(bs=1, gbu=2, gfu=3, irs=1, p_max=10, irs_m=30, ch_k=2, access='NOMA', channel_max=2,
                    gfu_max=2, reflect_max=np.pi*2, r_min=0.01, tau=1e-10,
                    amptitude_max=1, point=0,loc_X =10)
    scope_name = 'new_reward2' + stat+str(args.quantization_level)+str(args.learning_rate)+str(args.batch_size)+str(args.capacity) +\
                 str(args.max_episode)+ \
                 str(args.render_interval) +str(scene.gbu)+ str(scene.gfu) \
                 +str(scene.ch_k)+str(scene.p_max)+str(scene.tau)+str(scene.irs_m) +str(scene.r_min)+str(scene.access)
    RL_ideal_lst_channel = []
    RL_pd_lst = []
    for f in range(scene.gfu):
        if stat == 'no_irs' or 'random':
            RL_c = DQN_generate(len(scene.states)+scene.gfu+scene.gfu*scene.ch_k, len(scene.action_space),args=args,name=stat+'channel'+str(f))
            RL_p = DDPG(len(scene.states)+scene.gfu+scene.gfu*scene.ch_k, scene.ch_k,scene.p_max,args,name=stat+'power'+str(f))


        elif stat == 'ideal':
            RL_c = DQN_generate(len(scene.states)+scene.gfu+scene.gfu*scene.ch_k, len(scene.action_space),args=args,name=stat+'channel'+str(f))
            RL_p = DDPG(len(scene.states)+scene.gfu+scene.gfu*scene.ch_k,scene.ch_k,scene.p_max,args,name=stat+'power'+str(f))
            # RL_p = DDPGUP(scene.ch_k,
            #                  len(scene.states)+scene.irs_m*2+scene.gfu+scene.gfu*scene.ch_k,
            #                  scene.p_max, MEMORY_CAPACITY=memory_size,
            #                  name_str='--power'+'-'+stat+str(dqn_lrc)+str(f),replace_iter=replace_target_iter,L_A=lr_a,L_C=lr_c)
        else:
            RL_c = DQN_generate(len(scene.states)+scene.gfu+scene.gfu*scene.ch_k, len(scene.action_space),args=args,name=stat+'channel'+str(f))
            RL_p = DDPG(len(scene.states)+scene.gfu+scene.gfu*scene.ch_k, scene.ch_k,scene.p_max,args,name=stat+'power'+str(f))
            # RL_ideal_c = DeepQNetwork(n_actions=len(scene.action_space), actions_f=1,
            #                           n_features=len(scene.states)+scene.irs_m+scene.gfu+scene.gfu*scene.ch_k,
            #                           learning_rate=dqn_lrc,reward_decay=0.9,e_greedy=e_greedy,
            #                           batch_size= batch_size,replace_target_iter=replace_target_iter,  # 每 200 步替换一次 target_net 的参数
            #                           memory_size=memory_size,  # 记忆上限
            #                           e_greedy_increment=e_greedy_increment_c,output_graph=False,  # 是否输出 tensorboard 文件
            #                           name_str='--channel'+'-'+stat+str(dqn_lrc)+str(f),double_q = double_q,prioritized_replay = prioritized_r)
            # RL_pg_p = DDPGUP(scene.ch_k,
            #                  len(scene.states)+scene.irs_m+scene.gfu+scene.gfu*scene.ch_k,
            #                  scene.p_max, MEMORY_CAPACITY=memory_size,
            #                  name_str='--power'+'-'+stat+str(dqn_lrc)+str(f),replace_iter=replace_target_iter,L_A=lr_a,L_C=lr_c)
        RL_ideal_lst_channel.append(RL_c)
        RL_pd_lst.append(RL_p)
    DDPG_reflect_sita = []
    DDPG_reflect_amp = []
    if stat == 'non_ideal_cont':
        DDPG_reflect_sita = DDPG(len(scene.states)+scene.irs_m, scene.irs_m,scene.reflect_max,args,name=stat+'RIS'+str(1))
    if stat == 'ideal':
        DDPG_reflect_sita = DDPG(len(scene.states)+scene.irs_m, scene.irs_m,scene.reflect_max,args,name=stat+'RIS_sita'+str(1))
        DDPG_reflect_amp = DDPG(len(scene.states)+scene.irs_m, scene.irs_m,scene.amptitude_max,args,name=stat+'RIS_amp'+str(1))
    DQN_reflect_lst = []
    if stat == 'non_ideal_discrete':

        for m in range(scene.irs_m):
            DQN_reflect = DQN_generate(len(scene.states)+scene.irs_m,args.quantization_level,args=args,name=stat+'RIS_sita'+str(m))
            DQN_reflect_lst.append(DQN_reflect)

    result_no_mode = []
    t1 = time.time()
    before_step_channel_arr = np.ones((scene.gfu)) * -1
    before_step_power_arr = np.ones((scene.gfu*scene.ch_k))* -1
    before_step_irs_m_sita = np.ones((scene.irs_m))* -1
    before_step_irs_m_amp = np.ones((scene.irs_m))* -1
    reward_before = 0
    for episode in range(1):
        observation = scene.reset()

        observation1 = np.append(observation,before_step_channel_arr)
        observation1 = np.append(observation1,before_step_power_arr)
        # if stat == 'ideal':
        #     observation_RIS = np.append(observation,before_step_irs_m_sita)
        #     observation_RIS = np.append(observation_RIS,before_step_irs_m_amp)
        # else:
        observation_RIS = np.append(observation,before_step_irs_m_sita)

        # if stat != 'no_irs':
        #     observation = np.append(observation,before_step_irs_m_sita)
        # if stat == 'ideal':
        #     observation = np.append(observation,before_step_irs_m_amp)
        step = 0
        for detect in range(int(args.max_episode)):
            print(detect)
            epsilon = epsilon_by_frame(detect+1) # epsilon calculation
            args.exploration_noise = 0.6*epsilon
            # if epsilon <= 0.01:
            #     args.exploration_noise = 0
            print("%d,%f" %(detect,epsilon))

            # 信道计算
            action_c_channel_lst = []

            for f in range(scene.gfu):
                action_c_channel = RL_ideal_lst_channel[f].current_model.act(observation1, epsilon)
                action_c_channel_lst.append(int(action_c_channel))
            channel_gfu = []
            cou = 0
            for it in action_c_channel_lst:
                channel_gfu.append(list(scene.action_space[it]))
            candicate = [x for x in range(scene.ch_k)]
            radom_select = candicate
            channel_gbu = np.zeros((1,scene.gbu))
            for b in range(scene.gbu):
                channel_gbu[:,b] = radom_select[b]
            # 计算功率
            power_value_arr = np.zeros((scene.gfu,scene.ch_k))
            for f in range(scene.gfu):
                cur_p = RL_pd_lst[f].select_action(observation1)
                cur_p = (cur_p + np.random.normal(0, args.exploration_noise, size=scene.ch_k)).clip(
                    0, scene.p_max)
                power_value_arr[f,:] = cur_p
            # 反射系数选择
            reflect_amp_arr = np.zeros((1,scene.irs_m))
            reflect_action_arr = np.zeros((1,scene.irs_m))
            if stat == 'ideal':
                reflect_action = DDPG_reflect_sita.select_action(observation_RIS)
                reflect_action = (reflect_action + np.random.normal(0, args.exploration_noise, size=scene.irs_m)).clip(
                    0, scene.reflect_max-0.0001)
                reflect_action_arr = np.array(reflect_action).reshape(1,scene.irs_m)
                reflect_amp = DDPG_reflect_amp.select_action(observation_RIS)
                reflect_amp = (reflect_amp + np.random.normal(0, args.exploration_noise, size=scene.irs_m)).clip(
                    0, scene.amptitude_max)
                reflect_amp_arr = np.array(reflect_amp).reshape(1,scene.irs_m)
                for m in range(scene.irs_m):
                    if reflect_amp_arr[0,m] == 0:
                        reflect_amp_arr[:,m] = 1
                    if np.random.random() > 0.5:
                        reflect_amp_arr[:,m] = 1
                # else:

            if  stat == 'non_ideal_cont':
                reflect_action = DDPG_reflect_sita.select_action(observation_RIS)
                reflect_action = (reflect_action + np.random.normal(0, args.exploration_noise, size=scene.irs_m)).clip(
                    0, scene.reflect_max-0.0001)
                reflect_action_arr = np.array(reflect_action).reshape(1,scene.irs_m)
                reflect_amp_arr = np.ones((1,scene.irs_m))
            action_reflect_lst = []
            if stat == 'non_ideal_discrete':
                for m in range(scene.irs_m):
                    reflect_action = DQN_reflect_lst[m].current_model.act(observation_RIS,epsilon)
                    action_reflect_lst.append(reflect_action)
                    # if args.quantization_level >= 2:
                    if np.random.random()>0.5:
                        reflect_action_arr[:,m] = reflect_action*(2*np.pi/args.quantization_level)
                    else:
                        reflect_action_arr[:,m] = 0
                        action_reflect_lst[m] = 0
                    # else:
                    #     reflect_action_arr[:,m] = reflect_action*(2*np.pi/args.quantization_level)
                    reflect_amp_arr = np.ones((1,scene.irs_m))
            if stat == 'random':
                reflect_amp_arr = np.ones((1,scene.irs_m))
                reflect_action_arr = np.random.rand(1,scene.irs_m)*scene.reflect_max
            throughput,p_judge_result,new_channel_gfu,new_power,observation_,r_min_judge= \
                scene.step(channel_gbu,channel_gfu,reflect_action_arr,reflect_amp_arr,power_value_arr,1,step)
            new_channel_action_lst = []
            for gh in range(len(new_channel_gfu)):
                cur = new_channel_gfu[gh]
                for kl in range(len(scene.action_space)):
                    if cur == list(scene.action_space[kl]):
                        new_channel_action_lst.append(kl)

            count_p = 0
            count_r = 0#
            for d in range(p_judge_result.shape[1]):
                if p_judge_result[:,d]<0 and d not in r_min_judge and new_channel_action_lst[d]!=0 :
                    count_p += 1
                if d in r_min_judge and new_channel_action_lst[d]!=0 :
                    count_r +=1
            if count_p == 0 and count_r ==0:
                throughput_sum = sum(throughput)
            else:
                throughput = -throughput
                throughput_sum = 0
            judge_lst = np.where(throughput!= 0)[0]
            if step == 0:

                reward = sum(throughput)
            else:
                reward = 10000 * (sum(throughput)- result_no_mode[-1])
            # reward = throughput_sum
            # reward_before = reward
            print(p_judge_result)
            print('第%s,%d次no_mode结果为：%f,reward:%f' % (stat,step, throughput_sum,reward))
            result_no_mode.append(throughput_sum)

            observation1_ = np.append(observation_,np.array(new_channel_action_lst))
            observation1_ = np.append(observation1_,np.array(new_power).reshape(-1))

            reflect_value_arr = reflect_calculate(reflect_action_arr,reflect_amp_arr)
            observation_RIS_ = np.append(observation_,np.array(reflect_value_arr).reshape(-1))
            # if stat == 'ideal':
            #     observation_RIS_ = np.append(observation_,np.array(reflect_action_arr).reshape(-1))
            #     observation_RIS_ = np.append(observation_RIS_,np.array(reflect_amp_arr).reshape(-1))
            # else:
            #     observation_RIS_ = np.append(observation_,np.array(reflect_action_arr).reshape(-1))
            #save experience
            for f1 in range(scene.gfu):
                RL_ideal_lst_channel[f1].replay_buffer.push(observation1, new_channel_action_lst[f1], reward, observation1_, 0)
                RL_pd_lst[f1].replay_buffer.push((observation1, observation1_, new_power[f1,:], reward, np.float(0)))
                # RL_ideal_lst_channel[f1].store_transition(observation,new_channel_action_lst[f1],reward,observation_)
                # RL_pd_lst[f1].store_transition(observation,[],new_power[f1,:],reward,observation_)
            if stat == 'ideal':
                DDPG_reflect_sita.replay_buffer.push((observation_RIS, observation_RIS_, np.array(reflect_action_arr).reshape(-1), reward, np.float(0)))
                DDPG_reflect_amp.replay_buffer.push((observation_RIS, observation_RIS_, np.array(reflect_amp_arr).reshape(-1), reward, np.float(0)))
            if stat == 'non_ideal_cont':
                DDPG_reflect_sita.replay_buffer.push((observation_RIS, observation_RIS_, np.array(reflect_action_arr).reshape(-1), reward, np.float(0)))
            if stat == 'non_ideal_discrete':
                for im in range(scene.irs_m):
                    DQN_reflect_lst[im].replay_buffer.push(observation_RIS,action_reflect_lst[im],reward,observation_RIS_,0)

            observation1 = observation1_
            observation_RIS = observation_RIS_

            if len(RL_ideal_lst_channel[0].replay_buffer) > args.batch_size:
                for DQN_C in RL_ideal_lst_channel:
                    DQN_C.compute_td_loss(args)
                if stat == 'non_ideal_discrete':
                    for DQN_s in DQN_reflect_lst:
                        DQN_s.compute_td_loss(args)

            if (detect+1) % 40000 == 0:
                # for DQN_C in RL_ideal_lst_channel:
                RL_ideal_lst_channel[0].plot(detect, result_no_mode)
                if stat == 'non_ideal_discrete':
                    # for DQN_s in DQN_reflect_lst:
                    DQN_reflect_lst[0].plot(detect, result_no_mode)

            if (detect+1) % 50 == 0:
                for DQN_C in RL_ideal_lst_channel:
                    DQN_C.update_target()
                if stat == 'non_ideal_discrete':
                    for DQN_s in DQN_reflect_lst:
                        DQN_s.update_target()
            if (detect+1) % 100 == 0:
                for DPG_power in RL_pd_lst:
                    DPG_power.update(args)
                if stat == 'ideal':
                    DDPG_reflect_sita.update(args)
                    DDPG_reflect_amp.update(args)
                if stat == 'non_ideal_cont':
                    DDPG_reflect_sita.update(args)
            # "Total T: %d Episode Num: %d Episode T: %d Reward: %f



            step += 1


    for DQN_C in RL_ideal_lst_channel:
        DQN_C.save(args)
    for DDPG_p in RL_pd_lst:
        DDPG_p.save(args)
    if stat == 'non_ideal_discrete':
        for DQN_s in DQN_reflect_lst:
            DQN_s.save(args)

    if stat == 'ideal':
        DDPG_reflect_sita.save(args)
        DDPG_reflect_amp.save(args)
    if stat == 'non_ideal_cont':
        DDPG_reflect_sita.save(args)
    save_dict = {}

    save_dict['result_no_mode'] = result_no_mode
    filename = 'new_loc_mobility_count_1'+scope_name
    npysave(save_dict,dir_path+'/'+filename)
    print('finish')
    print('Running time:%f '%(time.time() - t1))
    import matplotlib.pyplot as plt

    plt.figure()
    # plt.plot(np.arange(len(result_all_plot[::2])), result_all_plot[::2],label='all')
    plt.plot(np.arange(len(result_no_mode)), result_no_mode,label='no_mode')

    plt.ylabel('Sum Rate')
    plt.xlabel('steps')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # stat_lst = ['non_ideal_discrete']
    stat_lst = ["no_irs","ideal","non_ideal_cont",'random']
    # stat_lst = ["ideal"]

    #"no_irs","no_irs","no_irs","ideal","non_ideal_cont",
    for stat in stat_lst:
        if stat == 'non_ideal_discrete':
            level_lst = [2,4]
        else:
            level_lst = [4]
        for level in level_lst:
            # for lr in [0.0001]:
                main(stat,level)

