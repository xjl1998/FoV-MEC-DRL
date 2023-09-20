from math import exp, cos, sin
from os.path import dirname, abspath
import  math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio
from numpy import random
from DRL.function_all import *
from mpl_toolkits.mplot3d import  Axes3D
import matlab
import matlab.engine
import tensorflow as tf
import time

tf.disable_v2_behavior()
tf.set_random_seed(1)
np.random.seed(1)
dir_path = dirname(abspath(__file__)) + '/位置文件'
CUE_all_coord = np.load(dir_path + '/cue.npy')
# coord_cue_near_random = np.load(dir_path + '/cue_near.npy')
# coord_cue_far_random = np.load(dir_path + '/cue_far.npy')
coord_cue_random = np.load(dir_path + '/cue.npy')
# GBU_all_coord = np.load(dir_path + '/gbu.npy')
# coord_gbu_random = np.load(dir_path + '/gbu_100001.npy')

class IrsCompMISOEnv:
    def __init__(self,bs_num, ue_num, mec_p_max,transmit_p_max, irs_units_num,antenna_num,fov_patch_num,reflect_max,
                 r_min,BW,continue_cvx=False, open_matlab=False,train= True,mec_rule = "default",rand_omega=False,load_H_path=None):
        '''
        :param bs_num: 基站数目
        :param ue_num: 用户设备数目
        :param mec_p_max: MEC最大功率限制
        :param transmit_p_max: 传输最大功率限制
        :param irs_units_num: IRS反射单元数目
        :param antenna_num: 发射天线数
        :param fov_patch_num: Fov数，目前fov_patch_num=ue_num
        :param reflect_max: 最大反射偏转角度
        :param r_min: 最小传输速率
        :param BW: 带宽
        :param with_CoMP: 是否使用多点协作传输
        :param open_matlab: 是否使用matlab
        :param train_mode: 是否训练
        :param mec_rule: 为缓存指配mec的策略
        :param rand_omega: 是否随机预编码矩阵
        '''
        self.mec_p_max =mec_p_max
        self.transmit_p_max = transmit_p_max
        self.train = train
        self.mec_rule = mec_rule
        self.init_bs_power=0
        self.init_bs_rates=0
        self.opt_bs_power= 0
        self.opt_bs_rates=0
        self.max_diff= -999
        self.continue_cvx= continue_cvx
        self.rs=0
        self.cr=1.5
        self.Kb=10**(-9)
        self.Ub=10**5
        self.ub=15
        self.time=0
        self.load_H = False
        self.total_diff = 0
        self.epsilon_fix = None
        self.rand_omega = rand_omega
        self.mec_storage=generate_storage_mec(bs_num,80,80)
        self.mec_max_computing_resources = generate_max_computing_resources_mec(bs_num,3200,3200)
        #这里改一下，直接定死每个MEC能缓存的内容
        self.fov_sizes = generate_fov_size(fov_patch_num,1,1)
        self.available_space = available_space(bs_num,fov_patch_num)
        self.epsilon = np.zeros([bs_num,fov_patch_num],np.int)
        self.epsilon_noCoMP = np.zeros([bs_num,fov_patch_num],np.int)
        self.stored_dic = dict()
        self.stored_dic_filename = '%d%d%d.npy' % (ue_num, bs_num, irs_units_num)
        self.stored_dic_mainkey =  '%d%d%d' % (ue_num, bs_num, irs_units_num)
        # if(os.path.exists(self.stored_dic_filename)):
        #     self.stored_dic = np.load(self.stored_dic_filename,allow_pickle=True)
        self.bs_num = int(bs_num)
        self.ue_num = int(ue_num)
        self.fov_patch_num = int(fov_patch_num)
        self.action_table= []
        self.irs_units_num = int(irs_units_num)
        self.antenna_num = int(antenna_num)
        self.storage_limit = np.ones(shape=self.bs_num,dtype=np.int)
        if(self.ue_num==6):
            self.storage_limit= self.storage_limit *4
        elif(self.ue_num==8):
            self.storage_limit =self.storage_limit*6
            # self.storage_limit = gen_mec_store_limit(5e-3, self.bs_num, self.mec_max_computing_resources,self.mec_p_max)


        self.action_table = gen_action_table_v2(bs_num,fov_patch_num)
        self.uefov_table = generate_uefov_table(self.ue_num)
        self.bsfov_table = generate_bsfov_table(self.epsilon)
        self.omegas =generate_omega_fixed(self.bs_num, self.ue_num, self.antenna_num,scale=0.1) #初始化一个omega
        self.total_power_record = []
        self.total_power_record_NoRIS = []
        self.init_bs_power_record = []
        self.total_init_power_record=[]
        self.bs_power_record_NoRIS=[]
        self.bs_power_record=[]
        self.bs_power_randOmega_record=[]
        self.available_action = []
        self.opt_G_record= []
        self.action_record=[]
        self.total_power_randOmega_record = []


        self.ue_avg_rates_record = []
        self.ue_avg_rates_record_NoRIS= []
        self.ue_avg_rates_record_noCoMP = []
        self.ue_avg_rates_record_noCoMP_NoRIS = []

        for bs in range (self.bs_num):
            self.available_action.append(np.ones(len(self.action_table)))
        self.available_action = np.array(self.available_action)

        self.r_min = r_min
        self.rendered_fov_sizes = cal_total_rendered_fov_sizes(self.fov_sizes,self.cr)
        self.total_computing_resources = cal_total_computing_resources(self.fov_sizes,self.Kb,self.Ub,self.ub,self.cr)
        self.BW = BW
        N_0_dbm = -174 + 10 * np.log10(1e7)
        self.N_0= np.power(10,((N_0_dbm - 30) / 10))

        self.cue_coord = coord_cue_random
        self.ch_space = np.zeros(self.ue_num)
        self.bs_coord = None
        self.gfu_max = 1
        self.engine = 0
        self.Gsize =self.bs_num*self.ue_num*self.antenna_num*2
        self.open_matlab=open_matlab
        if(open_matlab):
            self.engine = matlab.engine.start_matlab()
        self.action = np.zeros(self.bs_num)
        # self.n_reflect = 5 #将反射矩阵的系数划分成几等级
        self.action_irs = 0
        self.reflect =  np.diag(np.ones([self.irs_units_num],dtype=np.complex))
        # self.action_c_p = np.zeros((self.ue_num+self.antenna_num, self.antenna_num))
        self._coord_set()
        # self._gain_calculate()
        self.reset()
        # self.G,self.G2,self.g_ue_ris,self.g_bs_ris,self.g_bs_ue = all_G_gain_cal_MISO_splitI(self.time,self.bs_num, self.ue_num, self.antenna_num, self.irs_coord, self.cue_coord,
        #                              self.bs_coord, self.reflect, self.irs_units_num)
        # self.states = np.concatenate([np.array(self.G).flatten(), self.epsilon.flatten(),self.cue_coord[0, :self.ue_num, :]], axis=0)
        # self.states=self._gain_contact()+self.ch_add_states()+self.p_add_states()+self.reflect_amp_add_states()
        # self.states = self._gain_contact()
        # info = 'record_H_b%du%da%dru%d.npz' % (self.bs_num, self.ue_num, self.antenna_num, self.irs_units_num)
        if (load_H_path != None):
            self.load_H = True
            # load_H_path = os.path.join(load_H_path, 'record_H_b%du%da%dru%d.npz' % (
            #     self.bs_num, self.ue_num, self.antenna_num, self.irs_units_num))
            load_H_path = os.path.join(load_H_path, 'record_H.npz' )
            data = np.load(load_H_path)
            # self.H_record = data['H'][:,:self.Gsize]
            # self.H2_record = data['H2'][:,:,:self.ue_num,:self.antenna_num]
            self.Hrn_record =data['Hrn'][:,:,:self.ue_num,:self.irs_units_num]
            self.Hbr_record =data['Hbr'][:,:,:self.irs_units_num,:self.antenna_num]
            self.Hbn_record=data['Hbn'][:,:,:self.ue_num,:self.antenna_num]
            print("加载H_record成功，记录时间共%d秒，%d BS，%d UE，%d Antennas" % (
                self.Hrn_record.shape[0], self.bs_num, self.ue_num, self.antenna_num))


        print("MISO协作缓存环境创建完毕！")

    def _coord_set(self):
        '''
        :return: 根据预先的坐标按照不同数量进行选择
        '''
        # self.bs_coord = np.array([[0, 0, 0], [5, 20, 0], [20, 10, 0], [15, 15, 0]])
        self.bs_coord = np.array([[0, 3, 0], [5, 20, 0], [20, 10, 0]])
        # self.cue_coord = coord_cue_random[:self.ue_num, :]
        # a=GBU_all_coord[:self.antenna_num, :]
        # self.cue_coord = np.r_[self.cue_coord,a]
        # self.irs_coord = np.matrix([[31, 6, 0]])
        self.irs_coord = np.array([[6, 0, 0]])

        # self.plot_dynamic_movement()
        #显示1000s内的移动轨迹




        # if self.point>0:
        #     new_coord_lst = []
        #     random_count = self.point*15+1
        #     for g0 in range(self.ue_num):
        #         new_coord_lst.append(coord_cue_random[random_count,g0,:])
        #     for g1 in range(self.antenna_num):
        #         new_coord_lst.append(coord_gbu_random[random_count,g1+self.ue_num,:])
        #     self.cue_coord = np.array(new_coord_lst).reshape(self.ue_num+self.antenna_num,3)
        # else:
        # self._gain_calculate()


    def plot_dynamic_movement(self):
        print('基站用户位置分布图')
        #将位置plot出来
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        # matplotlib画图中中文显示会有问题，需要这两行设置默认字体
        fig,ax = plt.subplots(1,1)
        # ax=Axes3D(fig)
        delta=0.125
        # ax.set_xlabel('X-axis (m)', fontdict={'family': 'Times New Roman', 'size': 15})
        # ax.set_ylabel('Y-axis (m)', fontdict={'family': 'Times New Roman', 'size': 15})
        # ax.set_zlabel('Z-axis (m)', fontdict={'family': 'Times New Roman', 'size': 15})
        plt.xticks(fontproperties="Times New Roman", size=15)
        plt.yticks(fontproperties="Times New Roman", size=15)
        # ax.tick_params(axis='z', labelsize=15)
        # plt.xlim(-5,25)
        # plt.ylim(-5,25)
        # ax.set_zlim(0,30)
        colors1 = '#00CED1'  # 点的颜色
        colors2 = '#DC143C'
        colors3 = '#7FFFD4'
        colors4 = '#A52A2A'
        colors5 = '#008000'
        area = np.pi ** 2  # 点面积
        for i in range(0,1000):
            plt.ion()
            plt.clf()

            # 画散点图
            plt.scatter(self.bs_coord[:, 0], self.bs_coord[:, 1], s=area * 2, marker='o', c=colors1, alpha=0.4, label='BS')
            plt.scatter(self.irs_coord[:, 0], self.irs_coord[:, 1], s=area * 4, marker='s', c=colors5, alpha=0.4,
                        label='RIS')
            plt.scatter(self.cue_coord[i,:self.ue_num, 0], self.cue_coord[i,:self.ue_num, 1],s=area*2, marker='v', c=colors4, alpha=0.4, label='UE')
            plt.legend(loc=0,edgecolor='#000000',prop={'family': 'Times New Roman', 'size': 12})
            plt.grid('-')
            # plt.savefig(dir_path + '/location.png', bbox_inches="tight")
            # plt.savefig(dir_path + '/location.pdf', bbox_inches="tight")
            plt.show()
            plt.pause(1)
    def plot_location(self):
        #将位置plot出来
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # matplotlib画图中中文显示会有问题，需要这两行设置默认字体
        plt.xlabel('X')
        plt.ylabel('Y')
        print('位置分布图')
        colors1 = '#00CED1'  # 点的颜色
        colors2 = '#DC143C'
        colors3 = '#7FFFD4'
        colors4 = '#A52A2A'
        colors5 = '#008000'
        area = np.pi ** 2  # 点面积
        # 画散点图
        plt.scatter(self.irs_coord[:, 0], self.irs_coord[:, 1], s=area * 2, marker='s', c=colors2, alpha=0.4,
                    label='反射面')
        plt.scatter(self.cue_coord[:, 0], self.cue_coord[:, 1], s=area, marker='v', c=colors3, alpha=0.4, label='CUE用户')
        plt.legend()
        plt.savefig(dir_path + '/location.png', dpi=300)
        plt.show()

    def _gain_contact(self):
        #将计算出来的信道增益进行拼接作为state
        a=[]
        for ue_num in range(self.ue_num):
            for bs_i in range(self.bs_num):
                for ch_i in range(self.antenna_num):
                    which_bs = self.ch_space[ue_num]/self.antenna_num
                    which_ch = self.ch_space[ue_num]%self.antenna_num
                    if bs_i==which_bs and ch_i == which_ch:
                        a.append(self.G[ue_num])
                    else:
                        a.append(0)
        return a
    def reset(self):
        # 重新设置环境
        # if stat=="all":
        #     self._coord_set()
        if(self.load_H):
            # self.G = self.H_record[0,:]
            # G2 = np.zeros([ue + 1, antenna + 1, unit + 1, bs, ue, antenna, 1], dtype=np.complex)

            self.g_bs_ue =  self.Hbn_record[0,:]
            self.g_ue_ris = self.Hrn_record[0,:]
            self.g_bs_ris = self.Hbr_record[0,:]

            self.G2 = np.zeros([self.bs_num, self.ue_num, self.antenna_num, 1],dtype=complex)
            for b in range(self.bs_num):
                for u in range(self.ue_num):
                    self.G2[b,u,:,:]=G_gain_cal(self.g_bs_ue[b,u,:,:], self.g_bs_ris[b,:,:], self.g_ue_ris[b,u,:,:], 1)
            self.G = np.concatenate([self.G2.imag, self.G2.real], axis=-1).flatten()
            # self.omegas = gen_omega_ZF(self.fov_patch_num, self.bs_num, self.antenna_num, self.G2, 0.1)
            # self.G2 = self.g_bs_uere

        else:
            self.G,self.G2,self.g_ue_ris,self.g_bs_ris,self.g_bs_ue = all_G_gain_cal_MISO_splitI(self.time,self.bs_num, self.ue_num, self.antenna_num, self.irs_coord, self.cue_coord,
                                     self.bs_coord, self.reflect, self.irs_units_num)

            # self.G2 = self.g_bs_ue
            for b in range(self.bs_num):
                for u in range(self.ue_num):
                    self.G2[b,u,:,:]=G_gain_cal(self.g_bs_ue[b,u,:,:], self.g_bs_ris[b,:,:], self.g_ue_ris[b,u,:,:], 1)
            # self.omegas = gen_omega_ZF(self.fov_patch_num, self.bs_num, self.antenna_num, self.G2, 1)
            self.G = np.concatenate([self.G2.imag, self.G2.real], axis=-1).flatten()
        self.states = np.concatenate([np.array(self.G).flatten(),self.epsilon.flatten(),self.cue_coord[0, :self.ue_num, :].flatten()],axis=0)
        return self.states

    # def user_location_random(self):
    #     #随机生成下一步的位置
    #     limit = bs_dist_limit-100
    #     limit_1 = bs_dist_limit-50
    #     zeros_arr = np.array([0]).reshape(-1,1)
    #     for i in range(self.ue_num):
    #         cx = (-1 + 2*np.random.random())* limit
    #         cy = (-1 + 2*np.random.random())* limit
    #         cxy = np.array([cx,cy]).reshape(1,2)
    #         while np.linalg.norm(cxy, axis=1, keepdims=True) > limit_1:
    #             cx = (-1 + 2*np.random.random())* limit
    #             cy = (-1 + 2*np.random.random())* limit
    #             cxy = np.array([cx,cy]).reshape(1,2)
    #         self.cue_coord[i,:] = np.hstack((cxy,zeros_arr))
    #     print('cue新位置随机成功')
    def cal_reward(self,actions,step,concern_all=True):
        '''
        考虑约束的适应度函数值计算
        :return:
        '''
        # if step != 1:
        #     T_old = T_old*(step-1)
        # else:
        #     T_old = T_old
        bs = []
        value = 0
        ''' 
        
        先满足所有用户需求的fov都能在MEC上找到缓存，且缓存内容所消耗的计算资源不超过MEC，缓存总大小不超过MEC的存储容量
        动作为每一时刻MEC选择缓存的内容
        首先要验证每个基站选择的缓存行动是否满足约束
        '''
        sum_rendered_size = 0
        rho = 0.001
        sum_computing_resources = 0
        # self.omegas = gen_omega_ZF(self.fov_patch_num, self.bs_num, self.antenna_num, self.G2, 1)
        self.epsilon = gen_epsilon(self.bs_num, self.fov_patch_num, actions, self.action_table)

        # if(self.ue_num==6):
        #     self.storage_limit=np.ones(shape=self.bs_num,dtype=np.int) *4
        # elif(self.ue_num==8):
        #     self.storage_limit =np.ones(shape=self.bs_num,dtype=np.int)*6

        if(self.mec_rule=="max"):
            # self.epsilon = gen_epsilon_rule_larggest(self.epsilon,self.storage_limit.copy())
            epsilon  = gen_epsilon_rule_larggest2(self.epsilon,self.storage_limit.copy(),self.BW, self.G2, self.omegas,self.N_0)
            print(np.sum(self.epsilon - epsilon))
            self.epsilon = epsilon
        elif(self.mec_rule=="min"):
            # self.epsilon = gen_epsilon_rule_smallest(self.epsilon, self.storage_limit.copy())
            epsilon = gen_epsilon_rule_smallest2(self.epsilon, self.storage_limit.copy(), self.BW, self.G2,
                                                           self.omegas, self.N_0)
            print(np.sum(self.epsilon - epsilon))
            self.epsilon=epsilon
            # print(np.sum(self.epsilon_test - self.epsilon))



        if(self.train == False and self.rand_omega==True and self.open_matlab==False):
            self.omegas = generate_omega_random(bs_num=self.bs_num,ue_num=self.ue_num,antenna_num=self.antenna_num,scale_factor=0.1)
            transmitting_power_randOmega = cal_transmit_power(self.epsilon,self.omegas,self.bs_num,self.ue_num)


            mec_powers = cal_render_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
                                          bs_num=self.bs_num, fov_num=self.fov_patch_num,
                                          fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub,
                                          cr=self.cr,
                                          computing_resources=self.total_computing_resources)
            self.init_bs_power = np.array(self.init_bs_power, dtype=np.float).flatten()
            total_powers = (1 - rho) * self.init_bs_power + rho * mec_powers
            self.total_power_randOmega_record.append(np.mean(total_powers))
            print("transmitting_power_randOmega=%.4f,total_power:%.4f", transmitting_power_randOmega,total_powers)
            # self.bs_power_record.append()
            self.bs_power_randOmega_record.append(np.mean(transmitting_power_randOmega))
            return -50

        elif (self.train == False and self.rand_omega==False and self.open_matlab):
            rates = []
            self.epsilon_fix = rechoose_epsilon_noCoMP(self.epsilon, self.cue_coord, self.bs_coord,
                    step)

            scio.savemat(r'.\data.mat',
                         {'G': self.G2.tolist(), 'E': self.epsilon.tolist(), 'gnr': self.g_ue_ris.tolist(),
                          'gbr': self.g_bs_ris.tolist(), 'gbn': self.g_bs_ue.tolist(),'omegas':self.omegas.tolist()})
            self.init_bs_power, self.init_bs_rates, self.opt_bs_power, self.opt_rates,self.opt_rates_noCoMP,self.opt_G, self.rs = self.engine.main_optmization(
                matlab.double([self.bs_num]), matlab.double([self.fov_patch_num]),
                matlab.double([self.antenna_num]),
                matlab.double([self.irs_units_num]),
                matlab.double([self.N_0]), matlab.double([self.transmit_p_max]),
                matlab.double([self.r_min]), matlab.double(self.epsilon.tolist()),
                matlab.double(self.epsilon_fix.tolist()),matlab.double([self.BW]),
                nargout=7)

            if (self.mec_rule == "exhaustion"):
                start = time.time()
                self.available_epsilons = gen_epsilon_rule_exhaustion(self.epsilon, self.storage_limit.copy(), self.BW,
                                                                      self.G2,
                                                                      self.omegas, self.N_0, self.fov_sizes, self.Kb,
                                                                      self.Ub, self.ub, self.cr,
                                                                      self.total_computing_resources, self.mec_p_max)

                min_bs_power = 999
                min_ep = 0
                # self.init_bs_power, self.init_bs_rates, self.opt_bs_power, self.opt_rates, self.opt_rates_noCoMP, self.opt_G, self.rs = self.engine.main_optmization(
                #     matlab.double([self.bs_num]), matlab.double([self.fov_patch_num]),
                #     matlab.double([self.antenna_num]),
                #     matlab.double([self.irs_units_num]),
                #     matlab.double([self.N_0]), matlab.double([self.transmit_p_max]),
                #     matlab.double([self.r_min]), matlab.double(self.epsilon.tolist()),
                #     matlab.double(self.epsilon.tolist()), matlab.double([self.BW]),
                #     nargout=7)

                for ep in self.available_epsilons:
                    init_bs_power, init_bs_rates, opt_bs_power, opt_rates, opt_rates_noCoMP, opt_G, rs = self.engine.main_optmization(
                        matlab.double([self.bs_num]), matlab.double([self.fov_patch_num]),
                        matlab.double([self.antenna_num]),
                        matlab.double([self.irs_units_num]),
                        matlab.double([self.N_0]), matlab.double([self.transmit_p_max]),
                        matlab.double([self.r_min]), matlab.double(ep.tolist()),
                        matlab.double(self.epsilon.tolist()), matlab.double([self.BW]),
                        nargout=7)
                    if (rs == 1 and min_bs_power >= np.sum(opt_bs_power)):
                        min_bs_power = np.sum(opt_bs_power)
                        min_ep = ep
            # print(min_ep)
            # print(np.sum(self.epsilon - min_ep),time.time()-start)

                print(np.sum(self.epsilon) - np.sum(min_ep), time.time() - start)
                if (np.sum(min_ep) != 0):
                    print('替换')
                    self.epsilon_old = self.epsilon
                    self.epsilon = min_ep

                print(np.sum(self.opt_bs_power) - min_bs_power, time.time() - start)

                self.epsilon_fix = rechoose_epsilon_noCoMP(self.epsilon, self.cue_coord, self.bs_coord,step)
                self.init_bs_power, self.init_bs_rates, self.opt_bs_power, self.opt_rates,self.opt_rates_noCoMP,self.opt_G, self.rs = self.engine.main_optmization(
                    matlab.double([self.bs_num]), matlab.double([self.fov_patch_num]),
                    matlab.double([self.antenna_num]),
                    matlab.double([self.irs_units_num]),
                    matlab.double([self.N_0]), matlab.double([self.transmit_p_max]),
                    matlab.double([self.r_min]), matlab.double(self.epsilon.tolist()),
                    matlab.double(self.epsilon_fix.tolist()),matlab.double([self.BW]),
                    nargout=7)


            if(self.rs == 0 ):
                if(self.mec_rule== "default" or self.mec_rule== "exhaustion"):
                    self.ue_avg_rates_record.append(-1)
                    self.ue_avg_rates_record_NoRIS.append(-1)
                    self.total_power_record_NoRIS.append(-1)
                    self.total_power_record.append(-1)
                    self.bs_power_record.append(-1)
                    self.bs_power_record_NoRIS.append(-1)
                    self.init_bs_power_record.append(self.init_bs_power)
                    self.total_init_power_record.append(-1)
                    # self.opt_G_record.append(self.H_record[step,:])
                    return -50
                else:

                    mec_powers = cal_render_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
                                                  bs_num=self.bs_num, fov_num=self.fov_patch_num,
                                                  fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub,
                                                  cr=self.cr,
                                                  computing_resources=self.total_computing_resources)
                    self.init_bs_power = np.array(self.init_bs_power, dtype=np.float).flatten()
                    total_powers = (1 - rho) * self.init_bs_power+ rho * mec_powers
                    self.ue_avg_rates_record.append(-1)
                    self.ue_avg_rates_record_NoRIS.append(-1)
                    self.total_power_record_NoRIS.append(np.mean(total_powers))
                    self.total_power_record.append(np.mean(total_powers))
                    self.bs_power_record.append(np.mean(self.init_bs_power))
                    self.bs_power_record_NoRIS.append(np.mean(self.init_bs_power))
                    self.init_bs_power_record.append(np.mean(self.init_bs_power))
                    self.total_init_power_record.append(-1)
                    print("最大/最小策略失败，不优化", "total_power_record:", total_powers)
                    return -50
            self.opt_bs_power_noRIS, self.opt_rates_noRIS, self.opt_rates_noCoMP_noRIS, self.opt_noRIS_rs = self.engine.main_optmization_NoRIS(
                matlab.double([self.bs_num]), matlab.double([self.fov_patch_num]),
                matlab.double([self.antenna_num]), matlab.double([self.N_0]),
                matlab.double([self.transmit_p_max]), matlab.double([self.r_min]),
                matlab.double(self.epsilon.tolist()), matlab.double(self.epsilon_fix.tolist()),
                matlab.double([self.BW]), nargout=4)



            mec_powers = cal_render_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
                                         bs_num=self.bs_num, fov_num=self.fov_patch_num,
                                         fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub, cr=self.cr,
                                         computing_resources=self.total_computing_resources)
            print("mec power:",mec_powers)
            self.opt_bs_power = np.array(self.opt_bs_power,dtype=np.float).flatten()
            self.opt_bs_power_noRIS = np.array(self.opt_bs_power_noRIS,dtype=np.float).flatten()
            self.init_bs_power = np.array(self.init_bs_power,dtype=np.float).flatten()


            if (self.rs == 1 and self.opt_noRIS_rs and self.train==False):
                for bs in range(self.bs_num):
                    if (self.opt_bs_power[bs] > self.transmit_p_max):
                        return -50


                total_powers =(1-rho)*self.opt_bs_power+rho*mec_powers
                total_powers_NoRIS = (1-rho)*self.opt_bs_power_noRIS+rho*mec_powers
                total_init_powers = (1-rho)*self.init_bs_power+rho*mec_powers
                diff = np.sum(self.opt_bs_power-self.opt_bs_power_noRIS)
                if(diff<0 and self.max_diff<np.abs(diff)):
                    self.max_diff = np.abs(diff)
                    # self.max_diff_G = 0
                    scio.savemat(r'.\ue%d_data.mat' % (self.ue_num),
                                 {'G': self.G2.tolist(), 'E': self.epsilon.tolist(), 'gnr': self.g_ue_ris.tolist(),
                                  'gbr': self.g_bs_ris.tolist(), 'gbn': self.g_bs_ue.tolist(),
                                  'omegas': self.omegas.tolist()})


                # if (self.mec_rule in["default",'exhaustion']  and diff >0):
                #     # print("求解不成功_diff")
                #     self.ue_avg_rates_record.append(-1)
                #     self.ue_avg_rates_record_NoRIS.append(-1)
                #     self.total_power_record_NoRIS.append(-1)
                #     self.total_power_record.append(-1)
                #     self.bs_power_record.append(-1)
                #     self.bs_power_record_NoRIS.append(-1)
                #     self.init_bs_power_record.append(-1)
                #     self.total_init_power_record.append(-1)
                #
                #     self.opt_G_record.append(self.G2)
                #     return -50
                self.total_diff+=diff
                # if(np.sum(self.epsilon)>self.ue_num):
                #     print("CoMP")
                print("此次求解成功","total_power_record:",total_powers,"diff=%.6f"%(self.total_diff),"bs_power_record:",self.opt_bs_power,"bs_power_record_NoRIS",self.opt_bs_power_noRIS)


                self.ue_avg_rates_record.append(np.mean(self.opt_rates))
                self.ue_avg_rates_record_NoRIS.append(np.mean(self.opt_rates_noRIS))
                self.ue_avg_rates_record_noCoMP.append(np.mean(self.opt_rates_noCoMP))
                self.ue_avg_rates_record_noCoMP_NoRIS.append(np.mean(self.opt_rates_noCoMP_noRIS))
                self.total_power_record_NoRIS.append(np.mean(total_powers_NoRIS))
                self.total_power_record.append(np.mean(total_powers))
                self.bs_power_record.append(np.mean(self.opt_bs_power))
                self.bs_power_record_NoRIS.append(np.mean(self.opt_bs_power_noRIS))
                self.init_bs_power_record.append(np.mean(self.init_bs_power))
                self.total_init_power_record.append(np.mean(total_init_powers))
                self.opt_G_record.append(self.opt_G)
            else:
                print("求解不成功")
                self.ue_avg_rates_record_noCoMP.append(-1)
                self.ue_avg_rates_record_noCoMP_NoRIS.append(-1)
                self.ue_avg_rates_record .append(-1)
                self.ue_avg_rates_record_NoRIS.append(-1)
                self.total_power_record_NoRIS.append(-1)
                self.total_power_record.append(-1)
                self.bs_power_record.append(-1)
                self.bs_power_record_NoRIS.append(-1)
                self.init_bs_power_record.append(-1)
                self.total_init_power_record.append(-1)
                self.opt_G_record.append(self.G2)
                return -50
            # else:
            #     total_powers = 0.8*np.array(opt_power).reshape(self.bs_num)+0.2*cal_total_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
            #                                    omega=self.omegas, bs_num=self.bs_num, fov_num=self.fov_patch_num,
            #                                    fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub, cr=self.cr,
            #                                    computing_resources=self.total_computing_resources)
            reward=0
            #     for bs in range(self.bs_num):
            #         if (total_powers[bs] > self.trabsmit_p_max):
            #             return -50
            #     reward = (self.bs_num * self.trabsmit_p_max - np.sum(total_powers)) * 2

        elif (self.train == False and  self.open_matlab==False):
            self.action_record.append(actions)
            return -50

        else:
            # scio.savemat(r'.\data.mat',
            #              {'G': self.G2.tolist(), 'E': self.epsilon.tolist(), 'gnr': self.g_ue_ris.tolist(),
            #               'gbr': self.g_bs_ris.tolist(), 'gbn': self.g_bs_ue.tolist()})
            rates=[]
            for fov in range(self.fov_patch_num):
                rate = cal_transmit_rate(self.BW, self.G2, self.omegas, fov, self.epsilon, self.N_0)
                rates.append(rate)
                if (rate < self.r_min):
                    return  -50
            total_powers = 0
            if(concern_all==True):
                total_powers = cal_total_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
                                               omega=self.omegas, bs_num=self.bs_num, fov_num=self.fov_patch_num,
                                               fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub, cr=self.cr,
                                               computing_resources=self.total_computing_resources)
            else:
                total_powers = cal_total_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
                                               omega=generate_omega_fixed(self.bs_num, self.ue_num, self.antenna_num, scale=0.3), bs_num=self.bs_num, fov_num=self.fov_patch_num,
                                               fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub, cr=self.cr,
                                               computing_resources=self.total_computing_resources)
            for bs in range(self.bs_num):
                if(total_powers[bs]>self.mec_p_max):
                    return  -50
            reward = (self.bs_num*self.mec_p_max-np.sum(total_powers))*4
        return reward

    def cal_reward_3(self, actions, action_real,action_imag,step):
        '''
        考虑约束的适应度函数值计算
        :return:
        '''
        # if step != 1:
        #     T_old = T_old*(step-1)
        # else:
        #     T_old = T_old
        bs = []
        value = 0

        ''' 

        先满足所有用户需求的fov都能在MEC上找到缓存，且缓存内容所消耗的计算资源不超过MEC，缓存总大小不超过MEC的存储容量
        动作为每一时刻MEC选择缓存的内容
        首先要验证每个基站选择的缓存行动是否满足约束
        '''
        sum_rendered_size = 0
        rho = 0.001
        sum_computing_resources = 0
        # self.omegas = gen_omega_ZF(self.fov_patch_num, self.bs_num, self.antenna_num, self.G2, 1)
        self.epsilon = gen_epsilon(self.bs_num, self.fov_patch_num, actions, self.action_table)
        # if(self.ue_num==6):
        #     self.storage_limit=np.ones(shape=self.bs_num,dtype=np.int) *4
        # elif(self.ue_num==8):
        #     self.storage_limit =np.ones(shape=self.bs_num,dtype=np.int)*6

        if (self.mec_rule == "max"):
            self.epsilon = gen_epsilon_rule_larggest(self.epsilon, self.storage_limit.copy())
            # self.epsilon_test = gen_epsilon_rule_larggest2(self.epsilon,self.storage_limit.copy(),self.BW, self.G2, self.omegas,self.N_0)
            # print(np.sum(self.epsilon_test-self.epsilon))
            # return 10
        elif (self.mec_rule == "min"):
            self.epsilon = gen_epsilon_rule_smallest(self.epsilon, self.storage_limit.copy())
            # self.epsilon_test = gen_epsilon_rule_smallest2(self.epsilon, self.storage_limit.copy(), self.BW, self.G2,
            #                                                self.omegas, self.N_0)
            # print(np.sum(self.epsilon_test - self.epsilon))
            # return 10

        if (self.train == False and self.rand_omega == True and self.open_matlab == False):
            self.omegas = generate_omega_random(bs_num=self.bs_num, ue_num=self.ue_num, antenna_num=self.antenna_num,
                                                scale_factor=0.1)
            transmitting_power_randOmega = cal_transmit_power(self.epsilon, self.omegas, self.bs_num, self.ue_num)

            mec_powers = cal_render_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
                                          bs_num=self.bs_num, fov_num=self.fov_patch_num,
                                          fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub,
                                          cr=self.cr,
                                          computing_resources=self.total_computing_resources)
            self.init_bs_power = np.array(self.init_bs_power, dtype=np.float).flatten()
            total_powers = (1 - rho) * self.init_bs_power + rho * mec_powers
            self.total_power_randOmega_record.append(np.mean(total_powers))
            print("transmitting_power_randOmega=%.4f,total_power:%.4f", transmitting_power_randOmega, total_powers)
            # self.bs_power_record.append()
            self.bs_power_randOmega_record.append(np.mean(transmitting_power_randOmega))
            return -50

        elif (self.train == False and self.rand_omega == False and self.open_matlab):
            rates = []
            self.epsilon_fix = rechoose_epsilon_noCoMP(self.epsilon, self.cue_coord, self.bs_coord,
                                                       step)

            # for fov in range(self.fov_patch_num):
            #     rate = cal_transmit_rate(self.BW, self.G2, self.omegas, fov, self.epsilon, self.N_0)
            #     rates.append(rate)
            #     if (rate < self.r_min):
            #         print("错误,速率")
            #         return -50
            scio.savemat(r'.\data.mat',
                         {'G': self.G2.tolist(), 'E': self.epsilon.tolist(), 'gnr': self.g_ue_ris.tolist(),
                          'gbr': self.g_bs_ris.tolist(), 'gbn': self.g_bs_ue.tolist(), 'omegas': self.omegas.tolist()})
            # b=0
            # f=0
            # k  = G_gain_cal(self.g_bs_ue[b, f, :,:], self.g_bs_ris[b, :,:], self.g_ue_ris[b, f, :,:], self.reflect)
            self.init_bs_power, self.init_bs_rates, self.opt_bs_power, self.opt_rates, self.opt_rates_noCoMP, self.opt_G, self.rs = self.engine.main_optmization(
                matlab.double([self.bs_num]), matlab.double([self.fov_patch_num]),
                matlab.double([self.antenna_num]),
                matlab.double([self.irs_units_num]),
                matlab.double([self.N_0]), matlab.double([self.transmit_p_max]),
                matlab.double([self.r_min]), matlab.double(self.epsilon.tolist()),
                matlab.double(self.epsilon_fix.tolist()), matlab.double([self.BW]),
                nargout=7)

            if (self.rs == 0):
                # print("求解不成功",self.epsilon)
                if (self.mec_rule == "default"):
                    print("求解不成功")
                    self.ue_avg_rates_record.append(-1)
                    self.ue_avg_rates_record_NoRIS.append(-1)
                    self.total_power_record_NoRIS.append(-1)
                    self.total_power_record.append(-1)
                    self.bs_power_record.append(-1)
                    self.bs_power_record_NoRIS.append(-1)
                    self.init_bs_power_record.append(self.init_bs_power)
                    self.total_init_power_record.append(-1)
                    # self.opt_G_record.append(self.H_record[step,:])
                    return -50
                else:

                    mec_powers = cal_render_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
                                                  bs_num=self.bs_num, fov_num=self.fov_patch_num,
                                                  fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub,
                                                  cr=self.cr,
                                                  computing_resources=self.total_computing_resources)
                    self.init_bs_power = np.array(self.init_bs_power, dtype=np.float).flatten()
                    total_powers = (1 - rho) * self.init_bs_power + rho * mec_powers
                    self.ue_avg_rates_record.append(-1)
                    self.ue_avg_rates_record_NoRIS.append(-1)
                    self.total_power_record_NoRIS.append(np.mean(total_powers))
                    self.total_power_record.append(np.mean(total_powers))
                    self.bs_power_record.append(np.mean(self.init_bs_power))
                    self.bs_power_record_NoRIS.append(np.mean(self.init_bs_power))
                    self.init_bs_power_record.append(np.mean(self.init_bs_power))
                    self.total_init_power_record.append(-1)
                    print("最大/最小策略失败，不优化", "total_power_record:", total_powers)
                    return -50
            self.opt_bs_power_noRIS, self.opt_rates_noRIS, self.opt_rates_noCoMP_noRIS, self.opt_noRIS_rs = self.engine.main_optmization_NoRIS(
                matlab.double([self.bs_num]), matlab.double([self.fov_patch_num]),
                matlab.double([self.antenna_num]), matlab.double([self.N_0]),
                matlab.double([self.transmit_p_max]), matlab.double([self.r_min]),
                matlab.double(self.epsilon.tolist()), matlab.double(self.epsilon_fix.tolist()),
                matlab.double([self.BW]), nargout=4)

            mec_powers = cal_render_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
                                          bs_num=self.bs_num, fov_num=self.fov_patch_num,
                                          fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub, cr=self.cr,
                                          computing_resources=self.total_computing_resources)
            print("mec power:", mec_powers)
            self.opt_bs_power = np.array(self.opt_bs_power, dtype=np.float).flatten()
            self.opt_bs_power_noRIS = np.array(self.opt_bs_power_noRIS, dtype=np.float).flatten()
            self.init_bs_power = np.array(self.init_bs_power, dtype=np.float).flatten()

            if (self.rs == 1 and self.opt_noRIS_rs and self.train == False):
                for bs in range(self.bs_num):
                    if (self.opt_bs_power[bs] > self.transmit_p_max):
                        return -50

                total_powers = (1 - rho) * self.opt_bs_power + rho * mec_powers
                total_powers_NoRIS = (1 - rho) * self.opt_bs_power_noRIS + rho * mec_powers
                total_init_powers = (1 - rho) * self.init_bs_power + rho * mec_powers
                diff = np.sum(self.opt_bs_power - self.opt_bs_power_noRIS)
                if (self.mec_rule == "default" and diff > 0):
                    print("求解不成功_diff")
                    self.ue_avg_rates_record.append(-1)
                    self.ue_avg_rates_record_NoRIS.append(-1)
                    self.total_power_record_NoRIS.append(-1)
                    self.total_power_record.append(-1)
                    self.bs_power_record.append(-1)
                    self.bs_power_record_NoRIS.append(-1)
                    self.init_bs_power_record.append(-1)
                    self.total_init_power_record.append(-1)

                    self.opt_G_record.append(self.G2)
                    return -50
                self.total_diff += diff
                # if(np.sum(self.epsilon)>self.ue_num):
                #     print("CoMP")
                print("此次求解成功", "total_power_record:", total_powers, "diff=%.6f" % (self.total_diff),
                      "bs_power_record:", self.opt_bs_power, "bs_power_record_NoRIS", self.opt_bs_power_noRIS)

                self.ue_avg_rates_record.append(np.mean(self.opt_rates))
                self.ue_avg_rates_record_NoRIS.append(np.mean(self.opt_rates_noRIS))
                self.ue_avg_rates_record_noCoMP.append(np.mean(self.opt_rates_noCoMP))
                self.ue_avg_rates_record_noCoMP_NoRIS.append(np.mean(self.opt_rates_noCoMP_noRIS))
                self.total_power_record_NoRIS.append(np.mean(total_powers_NoRIS))
                self.total_power_record.append(np.mean(total_powers))
                self.bs_power_record.append(np.mean(self.opt_bs_power))
                self.bs_power_record_NoRIS.append(np.mean(self.opt_bs_power_noRIS))
                self.init_bs_power_record.append(np.mean(self.init_bs_power))
                self.total_init_power_record.append(np.mean(total_init_powers))
                self.opt_G_record.append(self.opt_G)
            else:
                print("求解不成功")
                self.ue_avg_rates_record_noCoMP.append(-1)
                self.ue_avg_rates_record_noCoMP_NoRIS.append(-1)
                self.ue_avg_rates_record.append(-1)
                self.ue_avg_rates_record_NoRIS.append(-1)
                self.total_power_record_NoRIS.append(-1)
                self.total_power_record.append(-1)
                self.bs_power_record.append(-1)
                self.bs_power_record_NoRIS.append(-1)
                self.init_bs_power_record.append(-1)
                self.total_init_power_record.append(-1)
                self.opt_G_record.append(self.G2)
                return -50
            # else:
            #     '''在满足最小速率的情况下，判断BS发射功率是否满足约束，先测试速率部分'''
            #     total_powers = 0.8*np.array(opt_power).reshape(self.bs_num)+0.2*cal_total_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
            #                                    omega=self.omegas, bs_num=self.bs_num, fov_num=self.fov_patch_num,
            #                                    fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub, cr=self.cr,
            #                                    computing_resources=self.total_computing_resources)
            reward = 0
            #     for bs in range(self.bs_num):
            #         if (total_powers[bs] > self.trabsmit_p_max):
            #             return -50
            #     reward = (self.bs_num * self.trabsmit_p_max - np.sum(total_powers)) * 2

        elif (self.train == False and self.open_matlab == False):
            self.action_record.append(actions)
            return -50

        else:
            '''在不超过存储容量和计算资源上限的前提下，计算UE上的速率，判断是否满足最小速率'''
            # scio.savemat(r'.\data.mat',
            #              {'G': self.G2.tolist(), 'E': self.epsilon.tolist(), 'gnr': self.g_ue_ris.tolist(),
            #               'gbr': self.g_bs_ris.tolist(), 'gbn': self.g_bs_ue.tolist()})

            self.omegas = (0.0001*action_real-0.5)+1j*(0.0001*action_imag-0.5)
            rates = []
            gap = 0
            for fov in range(self.fov_patch_num):
                rate = cal_transmit_rate(self.BW, self.G2, self.omegas, fov, self.epsilon, self.N_0)
                rates.append(rate)
                if (rate < self.r_min):
                    return -50
            #         gap+=(rate-self.r_min)
            # # #         # print("错误,速率")
            # if(gap<0):
            #         return gap
            # '''在满足最小速率的情况下，判断BS发射功率是否满足约束，先测试速率部分'''
            total_powers = cal_total_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
                                           omega=self.omegas, bs_num=self.bs_num, fov_num=self.fov_patch_num,
                                           fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub, cr=self.cr,
                                           computing_resources=self.total_computing_resources)

            gap = 0
            for bs in range(self.bs_num):
                if (total_powers[bs] > self.mec_p_max):
                    return -50
                #     gap += 2* (self.mec_p_max-total_powers[bs])
                # #     # print("错误,功率")
                # if (gap < 0):
                #     return gap
            reward = (self.bs_num * self.mec_p_max - np.sum(total_powers)) * 4
            # print(reward)
        return reward

    def cal_reward_2(self, actions, step,concern_all=True):
            self.epsilon = gen_epsilon(self.bs_num, self.fov_patch_num, actions, self.action_table)
            rates = []
            key = '%s%s%d'%(self.stored_dic_mainkey,str(actions),step)
            reward = self.stored_dic.get(key)

            if(reward==None or reward==-50):
                reward = 0
                total_powers = cal_total_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
                                               omega=self.omegas, bs_num=self.bs_num, fov_num=self.fov_patch_num,
                                               fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub, cr=self.cr,
                                               computing_resources=self.total_computing_resources)
                for bs in range(self.bs_num):
                    if (total_powers[bs] > self.mec_p_max):
                        return -50
                self.epsilon_fix = rechoose_epsilon_noCoMP(self.epsilon, self.cue_coord, self.bs_coord, step)
                scio.savemat(r'.\data.mat',
                             {'G': self.G2.tolist(), 'E': self.epsilon.tolist(), 'gnr': self.g_ue_ris.tolist(),
                              'gbr': self.g_bs_ris.tolist(), 'gbn': self.g_bs_ue.tolist(), 'omegas': self.omegas.tolist()})
                self.init_bs_power, self.init_bs_rates, self.opt_bs_power, self.opt_rates, self.opt_rates_noCoMP, self.opt_G, self.rs = self.engine.main_optmization(
                    matlab.double([self.bs_num]), matlab.double([self.fov_patch_num]),
                    matlab.double([self.antenna_num]),
                    matlab.double([self.irs_units_num]),
                    matlab.double([self.N_0]), matlab.double([self.transmit_p_max]),
                    matlab.double([self.r_min]), matlab.double(self.epsilon.tolist()),
                    matlab.double(self.epsilon_fix.tolist()), matlab.double([self.BW]),
                    nargout=7)
                if (self.rs == 0):
                        print("求解不成功")
                        return -50
                else:

                    reward = (self.bs_num * self.mec_p_max - np.sum(total_powers)) * 4
                self.stored_dic[key]=reward
                print(key,self.r_min,reward)
            return reward

    def cal_reward_validate(self, actions, step):
        '''
        考虑约束的适应度函数值计算
        :return:
        '''
        ''' 
        先满足所有用户需求的fov都能在MEC上找到缓存，且缓存内容所消耗的计算资源不超过MEC，缓存总大小不超过MEC的存储容量
        动作为每一时刻MEC选择缓存的内容
        首先要验证每个基站选择的缓存行动是否满足约束
        '''
        self.epsilon = gen_epsilon(self.bs_num, self.fov_patch_num, actions, self.action_table)
        '''在不超过存储容量和计算资源上限的前提下，计算UE上的速率，判断是否满足最小速率'''
        # scio.savemat(r'.\data.mat',
        #              {'G': self.G2.tolist(), 'E': self.epsilon.tolist(), 'gnr': self.g_ue_ris.tolist(),
        #               'gbr': self.g_bs_ris.tolist(), 'gbn': self.g_bs_ue.tolist()})
        total_powers = cal_total_power(static_power=10, pbd=5e-3, epsilon=self.epsilon,
                                           omega=self.omegas, bs_num=self.bs_num, fov_num=self.fov_patch_num,
                                           fov_sizes=self.fov_sizes, Kb=self.Kb, Ub=self.Ub, ub=self.ub, cr=self.cr,
                                           computing_resources=self.total_computing_resources)
        for bs in range(self.bs_num):
                if (total_powers[bs] > self.mec_p_max):
                    return -50
        # reward = (self.bs_num * self.mec_p_max - np.sum(total_powers)) * 4
        return 1

    def step2(self, step):


        self.G, self.G2, self.g_ue_ris, self.g_bs_ris, self.g_bs_ue = all_G_gain_cal_MISO_splitI(step, self.bs_num,
                                                                                                     self.ue_num,
                                                                                                     self.antenna_num,
                                                                                                     self.irs_coord,
                                                                                                     self.cue_coord,
                                                                                                     self.bs_coord,
                                                                                                     self.reflect,
                                                                                                     self.irs_units_num)

        # scio.savemat(r'.\data.mat',
        #                  {'G': self.G2.tolist(), 'E': self.epsilon.tolist(), 'gnr': self.g_ue_ris.tolist(),
        #                   'gbr': self.g_bs_ris.tolist(), 'gbn': self.g_bs_ue.tolist(),'omegas':self.omegas.tolist()})

        return self.G, self.G2, self.g_ue_ris, self.g_bs_ris, self.g_bs_ue

    # def step(self,actions,action_real,action_imag,step):
    #
    #     r = self.cal_reward(actions,action_real,action_imag, step)
    #     new_coord_lst = []
    #     if (self.load_H):
    #         # self.G = self.H_record[step, :]
    #         # G2 = np.zeros([ue + 1, antenna + 1, unit + 1, bs, ue, antenna, 1], dtype=np.complex)
    #
    #         # self.G2 = self.H2_record[step,self.ue_num, self.antenna_num, self.irs_units_num, :, :, :, :]
    #         # self.G2 = self.H2_record[step, :]
    #         self.g_ue_ris = self.Hrn_record[step, :]
    #         self.g_bs_ris = self.Hbr_record[step, :]
    #         self.g_bs_ue = self.Hbn_record[step, :]
    #         for b in range(self.bs_num):
    #             for u in range(self.ue_num):
    #                 self.G2[b, u, :, :] = G_gain_cal(self.g_bs_ue[b, u, :, :], self.g_bs_ris[b, :, :],
    #                                                  self.g_ue_ris[b, u, :, :], 1)
    #
    #         # self.G2 = self.g_bs_ue
    #         self.G = np.concatenate([self.G2.imag, self.G2.real], axis=-1).flatten()
    #         # self.omegas = gen_omega_ZF(self.fov_patch_num, self.bs_num, self.antenna_num, self.G2, 0.1)
    #     else:
    #         self.G, self.G2, self.g_ue_ris, self.g_bs_ris, self.g_bs_ue = all_G_gain_cal_MISO_splitI(step, self.bs_num,
    #                                                                                                  self.ue_num,
    #                                                                                                  self.antenna_num,
    #                                                                                                  self.irs_coord,
    #                                                                                                  self.cue_coord,
    #                                                                                                  self.bs_coord,
    #                                                                                                  self.reflect,
    #                                                                                                  self.irs_units_num)
    #         for b in range(self.bs_num):
    #             for u in range(self.ue_num):
    #                 self.G2[b, u, :, :] = G_gain_cal(self.g_bs_ue[b, u, :, :], self.g_bs_ris[b, :, :],
    #                                                  self.g_ue_ris[b, u, :, :], 1)
    #         # self.G2 = G_gain_cal(self.g_bs_ue, self.g_bs_ris, self.g_ue_ris, 1)
    #         self.G = np.concatenate([self.G2.imag, self.G2.real], axis=-1).flatten()
    #         # self.omegas = gen_omega_ZF(self.fov_patch_num, self.bs_num, self.antenna_num, self.G2, 0.1)
    #         # self.G2 = self.g_bs_ue
    #
    #     states_ = self.states = np.concatenate(
    #         [np.array(self.G).flatten(), self.epsilon.flatten(), self.cue_coord[step, :self.ue_num, :].flatten()],
    #         axis=0)
    #     return r, states_, self.epsilon
    def step(self, actions,step,concern_all=True):
        if(self.continue_cvx==False):
            r = self.cal_reward(actions,step,concern_all)
        else:
            r = self.cal_reward_2(actions,step,concern_all)
        new_coord_lst =[]
        if (self.load_H):
            # self.G = self.H_record[step, :]
            # G2 = np.zeros([ue + 1, antenna + 1, unit + 1, bs, ue, antenna, 1], dtype=np.complex)

            # self.G2 = self.H2_record[step,self.ue_num, self.antenna_num, self.irs_units_num, :, :, :, :]
            # self.G2 = self.H2_record[step, :]
            self.g_ue_ris = self.Hrn_record[step, :]
            self.g_bs_ris = self.Hbr_record[step, :]
            self.g_bs_ue = self.Hbn_record[step, :]
            for b in range(self.bs_num):
                for u in range(self.ue_num):
                    self.G2[b, u, :, :] = G_gain_cal(self.g_bs_ue[b, u, :, :], self.g_bs_ris[b, :, :],
                                                     self.g_ue_ris[b, u, :, :], 1)
                    # self.G2[b, u, :, :] = self.g_bs_ue[b, u, :, :]

            # self.G2 = self.g_bs_ue
            self.G = np.concatenate([self.g_bs_ue.imag, self.g_bs_ue.real], axis=-1).flatten()
            # self.omegas = gen_omega_ZF(self.fov_patch_num, self.bs_num, self.antenna_num, self.G2, 0.1)
        else:
            self.G, self.G2, self.g_ue_ris, self.g_bs_ris, self.g_bs_ue = all_G_gain_cal_MISO_splitI(step, self.bs_num,
                                                                                                     self.ue_num,
                                                                                                     self.antenna_num,
                                                                                                     self.irs_coord,
                                                                                                     self.cue_coord,
                                                                                                     self.bs_coord,
                                                                                                     self.reflect,
                                                                                                     self.irs_units_num)
            for b in range(self.bs_num):
                for u in range(self.ue_num):
                    self.G2[b,u,:,:]=G_gain_cal(self.g_bs_ue[b,u,:,:], self.g_bs_ris[b,:,:], self.g_ue_ris[b,u,:,:], 1)
            # self.G2 = G_gain_cal(self.g_bs_ue, self.g_bs_ris, self.g_ue_ris, 1)
            self.G = np.concatenate([self.G2.imag, self.G2.real], axis=-1).flatten()
            # self.omegas = gen_omega_ZF(self.fov_patch_num, self.bs_num, self.antenna_num, self.G2, 0.1)
            # self.G2 = self.g_bs_ue


        states_ = self.states = np.concatenate([ np.array(self.G).flatten(),self.epsilon.flatten(),self.cue_coord[step, :self.ue_num, :].flatten()], axis=0)
        return r,states_,self.epsilon
    def action_states(self):
        p = []
        for i in range(self.bs_num):
            p.append(self.action[i])
        return p
    def reflect_amp_add_states(self):
        reflect_amp= []
        for i in range(self.irs_units_num):
            reflect_amp.append(self.reflect[i][i])
        return reflect_amp
    def G_tau__add_states(self):
        g_tau= []
        for i in range(self.ue_num):
            for j in range(self.antenna_num):
                if self.G[i][j]**2>=self.tau:
                    g_tau.append(1)
                else:
                    g_tau.append(0)
        return g_tau

