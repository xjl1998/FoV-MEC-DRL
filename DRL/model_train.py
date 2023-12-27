
import time
import os
from os import makedirs
from os.path import dirname, abspath, exists
import matlab
import matlab.engine
import scipy.io as scio
from tqdm import tqdm
from pylab import  *
from DRL.TFAgent.DDPG_update import DDPGUP
from DRL.TFAgent.DQN import DqnAgent
import matplotlib.pyplot as plt
import numpy as np
import  os
from config import  FLAGS
from DRL.env import *
import tensorflow as tf
from functools import  reduce
from brokenaxes import brokenaxes
tf.disable_v2_behavior()
tf.set_random_seed(1)
np.random.seed(1)
BW = 40
N_0_dbm = -174 + 10 * np.log10(BW)

plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
plt.tight_layout()
import multiprocessing
# from pathos.multiprocessing import ProcessingPool as Pool
# engine.cd(r'E:\研究生\工作\毕业论文\VR
# \code\matlab\test')
# bs_fix_power_dict = {'Baseline_Bs3_Fov6_Antenna5': 2.32, 'Baseline_Bs3_Fov6_Antenna8': 3.84, 'Baseline_Bs3_Fov8_Antenna5':3.2,'Baseline_Bs3_Fov8_Antenna8':5.12}

bs_fix_power_dict = {'base fov:6 antenna:5': 2.32, 'base fov:6 antenna:8': 2.84, 'base fov:8 antenna:5':2.78,'base fov:8 antenna:8':3.41}
mec_fix_power_dict = {'u6':31,'u8':42,'u10':52}
# dictionary为参数名
# key1......keyn为键名，必须唯一且不可变，键名可以是字符串、数字或者元组
# value1......valuen表示元素的值，可以是任何数据类型，不一定唯一

marker_lst = ['s', 'd', 'o', '^', 'x', '<', '+', '>','^','*']
color_lst = ['red', 'green', 'black', 'blue', 'grey','m','c','purple']
markerfacecolor_lst = ['none', 'none', 'none', 'none', 'none', 'none','none','none','none']
dic ={2:'dual',3:'three'}
'''
plot_reward_graph
'''
def plot_reward_graph(x_labelName,y_labelName,start,disp_amount,disp_gap=10):
    Ys = []

    data_path = []
    data_path.append('./simulation_result/full/reward_record_ue8_bs3_u30_a5_r5_improved_concern_all_DDQN.npy')#DDQN+MAT
    data_path.append('./simulation_result/full/reward_record_ue8_bs3_u30_a5_r5_improved_concern_part_DDQN.npy')#DDQN+rand
    data_path.append('./simulation_result/full/reward_record_ue8_bs3_u30_a5_r5_improved_concern_allDQN.npy')#DQN+MAT
    data_path.append('./simulation_result/full/reward_record_ue8_bs3_u30_a5_r5_base_concern_all_DDQN_model_free.npy')  # 全DDQN

    label_lst = []
    label_lst.append('DDQN-AO')#DDQN and AO jointly optimize (proposed method)
    label_lst.append('DDQN-Cache')#Only optimize cache placement with DDQN
    label_lst.append('DQN-AO')#DQN and AO jointly optimize
    label_lst.append('DDQN-NAO')#DDQN optimize both cache and beamforming
    # data_path.append(reward_record_ue8_bs3_u30_a5_r5_base_concern_allDQN)

    for i in range(len(data_path)):
        Ys.append(np.load(data_path[i]))


    for j in range(len(label_lst)):
        y = np.array(Ys[j][start:disp_amount])
        x = np.arange(len(y))
        #
        if (len(y) > 5):
            x = np.arange(1, len(y)+1, disp_gap)
            # end_x  =x
            y = y[x]

        plt.plot(x, y, linewidth=1.5, color=color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                 marker=marker_lst[j], markersize=5, label=label_lst[j])
    plt.xlabel(x_labelName,labelpad=0, fontdict={'family' : 'Times New Roman', 'size': 15})
    plt.ylabel(y_labelName,labelpad=0, fontdict={'family' : 'Times New Roman', 'size': 15})
    plt.xticks(fontproperties="Times New Roman",size=15)
    plt.yticks(fontproperties="Times New Roman", size=15)
    plt.legend(loc=0,edgecolor='#000000',prop={'family': 'Times New Roman', 'size': 12})
    plt.xlim(0,6050)
    # plt.ylim(20,100)
    plt.grid(linestyle='-')


    # ax = plt.gca()
    # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))
    plt.savefig('.\仿真图\\'+'average_reward.png',bbox_inches="tight")
    plt.savefig('.\仿真图\\'+'average_reward.pdf',bbox_inches="tight")
    plt.close()

'''
plot_simulation_mec_power
'''
def plot_simulation_mec_power(ue_num_seq,bs_num_seq,units_seq,antenna_seq,r_min_seq,bw_seq,mec_rule_seq,with_CoMP_seq,x_labelName,y_labelName,fixed_ue_seq=[6,8,10],fixed_bs_seq=[3,3,3],fixed_power_seq=[31,42,52],start=0,disp_amount=60,disp_gap=10,loc=4):
    ue_seq_len = len(ue_num_seq)
    bs_seq_len = len(bs_num_seq)

    label_lst = []
    # label_lst = ['Ideal IRS','Non-ideal IRS with continuous phase shifts','Non-ideal IRS with discrete phase shifts'
    #     ,'No IRS']

    if(ue_seq_len!=bs_seq_len):
        print("输入的序列长度不对齐")
        return
    else:
        model_path_seq=[]
        bsp_npy_seq=[]
        tp_npy_seq=[]
        mec_power_seq = []
        avg_power_seq = []
        long_term_avg_power_seq=[]
        indices = []

        for i in range(ue_seq_len):
            if (mec_rule_seq == "max"):
                addition = "_mecRuleMax"
            elif (mec_rule_seq == "min"):
                addition = "_mecRuleMin"
            else:
                addition = ""
            if (with_CoMP_seq == False):
                addition = addition + "_noCoMP"
            head = '_ue%d_bs%d_u%d_a%d_r%d_%s' % (ue_num_seq[i], bs_num_seq[i], units_seq[i], antenna_seq[i], r_min_seq[i],  addition)

            model_path_seq.append("..\DRL\simulation_result\\full\model_ue" + str(ue_num_seq[i]) + "_bs" + str(
                bs_num_seq[i]) + "_u" + str(units_seq[i]) + "_a" + str(antenna_seq[i]) + '_r' + str(
                r_min_seq[i]) + '_bw' + str(bw_seq[i]) + ".ckpt")
            tp_npy_seq.append(
                "..\DRL\simulation_result\\full\\total_power_record" +head +".npy")
            bsp_npy_seq.append(
                "..\DRL\simulation_result\\full\\bs_power_record"+ head+".npy")

            label_lst.append("bs_num=" + str(bs_num_seq[i]) + ",fov_num=" + str(ue_num_seq[i])+addition)

        for i in range(ue_seq_len):
            collections = []
            total_power = np.load(tp_npy_seq[i])
            bs_power = np.load(bsp_npy_seq[i])
            for index in range(total_power.shape[0]):
                if (total_power[index] > 0):
                    collections.append(index)
            indices.append(np.array(collections))

        for i in range(ue_seq_len):
            total_power = np.load(tp_npy_seq[i])[indices[i]][start:disp_amount]
            bs_power = np.load(bsp_npy_seq[i])[indices[i]][start:disp_amount]
            mec_power = 10*(total_power-bs_power)
            long_term_avg_power = []
            sum_power= 0
            for time in range(len(mec_power)):
                sum_power=sum_power+mec_power[time]
                long_term_avg_power.append(sum_power/(time+1))
            long_term_avg_power_seq.append(long_term_avg_power)
            mec_power_seq.append(mec_power)
        plt.figure()


        for j in range(len(label_lst)):
            # times_sum_rate[:10000,j] = times_sum_rate[10000,j]
            y = long_term_avg_power_seq[j]
            x = np.arange(0,len(y))
            if(len(x)>5):
                x_slice=np.arange(0,len(x),disp_gap)
                y_slice=np.array(y)[x_slice]

            plt.plot(x_slice,y_slice,linewidth = 1.5, color =color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                     marker = marker_lst[j],markersize=8,label = label_lst[j])
        begin = len(label_lst)
        for j in range(begin,begin+len(fixed_ue_seq)):
            label_lst.append("bs_num="+str(bs_num_seq[j-begin])+",fov_num="+str(ue_num_seq[j-begin])+"Fixed")
            # times_sum_rate[:10000,j] = times_sum_rate[10000,j]
            y = np.ones(len(long_term_avg_power_seq[0]))*fixed_power_seq[j-begin]
            x = np.arange(0, len(y))
            if (len(x) > 5):
                x_slice = np.arange(0, len(x),disp_gap)
                y_slice = np.array(y)[x_slice]

            plt.plot(x_slice, y_slice, linewidth=1.5, color=color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                     marker=marker_lst[j], markersize=8, label=label_lst[j])



        plt.legend(loc=loc,edgecolor='#000000',prop={'family': 'Times New Roman', 'size': 11})
        # plt.xlim(10000,100000)
        # plt.ylim(20,100)
        plt.grid(linestyle='-')
        plt.xlabel(x_labelName,fontdict={'family' : 'Times New Roman', 'size': 12})
        plt.ylabel(y_labelName,fontdict={'family' : 'Times New Roman', 'size': 12})
        # ax = plt.gca()
        # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))

        for i in range(ue_seq_len):
            head = head + "_b" + str(bs_num_seq[i]) + "_f" + str(ue_num_seq[i]) + "_u" + str(units_seq[i]) + "_a" + str(
                antenna_seq[i]) + str(r_min_seq[i]) + str(bw_seq[i])
            if (mec_rule_seq[i] == "max"):
                head = head + "_a"
            elif (mec_rule_seq[i] == "min"):
                head = head + "_i"
            if (with_CoMP_seq[i] == True):
                head = head + "_t"
            else:
                head = head + "_f"
        plt.savefig('.\仿真图\\'+'avg_sum_power'+head+'.png')
        plt.savefig('.\仿真图\\'+'avg_sum_power'+head+'.pdf')
        plt.close()


'''
bs_fix_power = {'u6b3a5': 2.4, 'u6b3a8': 3.84, 'u8b3a5':3.2,'u8b3a8':5.12}
mec_fix_power = {'u6':31,'u8':42,'u10':52}
'''
'''
plot_simulation_results_total_power
'''
def plot_simulation_results_total_power(ue_num_seq,bs_num_seq,units_seq,antenna_seq,r_min_seq,bw_seq,mec_rule_seq,with_CoMP_seq,x_labelName,y_labelName,disp_amount,start,disp_gap,fix_ue_seq,fix_antenna_seq,disp_elements):
    ue_seq_len = len(ue_num_seq)
    bs_seq_len = len(bs_num_seq)


    label_lst = []
    end_x=0

    # label_lst = ['Ideal IRS','Non-ideal IRS with continuous phase shifts','Non-ideal IRS with discrete phase shifts'
    #     ,'No IRS']
    if(ue_seq_len!=bs_seq_len):
        print("输入的序列长度不对齐")
        return
    else:
        model_path_seq=[]
        result_npy_seq=[]
        result_bs_npy_seq=[]
        result_mec_npy_seq=[]
        records = []
        indices = []
        avg_power_seq = []
        long_term_avg_power_seq=[]
        for i in range(ue_seq_len):
            label = ""
            if (disp_elements[0] == True):
                label = label + "B=" + str(bs_num_seq[i]) + " "
            if (disp_elements[1] == True):
                label = label + "%s FoVs"%str(ue_num_seq[i]) + " "
            if (disp_elements[2] == True):
                label = label + "Q=" + str(units_seq[i]) + " "
            if (disp_elements[3] == True):
                label = label + "K=" + str(antenna_seq[i]) + " "
            if (disp_elements[4] == True):
                label = label + "R=" + str(r_min_seq[i]) + " "

            if (mec_rule_seq[i] == "max"):
                addition = "_mecRuleMax"
                label = label + "Max-AO"
            elif (mec_rule_seq[i] == "min"):
                addition = "_mecRuleMin"
                label = label+"Min-AO"
            elif (mec_rule_seq[i] == 'exhaustion'):
                addition = 'mecRuleExhaustion'
                label = label+'Exhaustion-AO'
            else:
                addition=""
                label = label + "DDQN-AO"
            if (with_CoMP_seq[i] == False):
                addition = addition + "_noCoMP"

            label_lst.append(label)
            head = '_ue%d_bs%d_u%d_a%d_r%d%s_improved_concern_all_DDQN' % ( ue_num_seq[i], bs_num_seq[i], units_seq[i], antenna_seq[i], r_min_seq[i],  addition)
            model_path_seq.append("..\DRL\simulation_result\\full\model_ue"+str(ue_num_seq[i])+"_bs"+str(bs_num_seq[i])+"_u"+str(units_seq[i])+"_a"+str(antenna_seq[i])+'_r'+str(r_min_seq[i])+".ckpt")
            result_npy_seq.append("..\DRL\simulation_result\\full\\total_power_record"+head+".npy")
            result_bs_npy_seq.append( "..\DRL\simulation_result\\full\\bs_power_record" + head + ".npy")
            result_mec_npy_seq.append( "..\DRL\simulation_result\\full\\mec_power_record" + head + ".npy")
            # label_lst.append(str(bs_num_seq[i])+'BS,'+str(ue_num_seq[i])+'Fov,'+str(antenna_seq[i])+"Antenna"+addition)




        for i in range(ue_seq_len):
            collections = []
            avg_power = np.load(result_npy_seq[i])
            for index in range(avg_power.shape[0]):
                if (avg_power[index] > 0):
                        collections.append(index)
            indices.append(np.array(collections))
        indices = reduce(np.intersect1d, indices)
        indices = np.tile(indices, [10])
            # common_indices = indices[0]
            # for i in range(1,ue_seq_len):
            #     common_indices = np.intersect1d(common_indices,indices[i])
        for i in range(ue_seq_len):
            avg_power = 10*np.log10(3*np.load(result_npy_seq[i])[indices]/0.001)
            # avg_bs_power = np.load(result_bs_npy_seq[i])[indices]
            # avg_mec_power = np.load(result_mec_npy_seq[i])[indices]
            # avg_power = 10*(avg_power-0.9*avg_bs_power)
            # records.append(avg_power)
            # avg_power =mec_power_seq[i]-0.5*avg_reward/bs_num_seq[i]
            long_term_avg_power = []
            sum_power= 0
            for time in range(len(avg_power)):
                sum_power=sum_power+avg_power[time]
                long_term_avg_power.append(sum_power/(time+1))
                # long_term_avg_power.append(avg_power[time])
            long_term_avg_power_seq.append(long_term_avg_power)
        long_term_avg_power_seq = np.array(long_term_avg_power_seq)
        plt.figure()

        for j in range(len(label_lst)):
            # times_sum_rate[:10000,j] = times_sum_rate[10000,j]
            y = np.array(long_term_avg_power_seq[j])[start:disp_amount]
            x=np.arange(len(y))
            #
            if(len(y)>5):
                x=np.arange(0,len(y),disp_gap)
                # end_x  =x
                y=y[x]

            plt.plot(x,y,linewidth = 1.5, color =color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                     marker = marker_lst[j],markersize=5,label = label_lst[j])

        # #画固定功率
        # for j in range(len(label_lst), len(label_lst) + len(fix_ue_seq)):
        #     # times_sum_rate[:10000,j] = times_sum_rate[10000,j]
        #     # seq = "base fov:" + str(fix_ue_seq[j - len(label_lst)])+" antenna:"+str(fix_antenna_seq[j - len(label_lst)])
        #     seq = "base fov:" + str(fix_ue_seq[j - len(label_lst)]) + " antenna:" + str(
        #         fix_antenna_seq[j - len(label_lst)])
        #     label = "base fov:" + str(fix_ue_seq[j - len(label_lst)])
        #     y = 10*np.log10(np.ones(disp_amount - start) * (bs_fix_power_dict[seq])/0.001)
        #     x = np.arange(len(y))
        #     #
        #     if (len(y) > 5):
        #         x = np.arange(0, len(y), disp_gap)
        #         # end_x  =x
        #         y = y[x]
        #
        #     plt.plot(x, y, linewidth=1.5, color=color_lst[j], markerfacecolor=markerfacecolor_lst[j],
        #              marker=marker_lst[j], markersize=8, label=label)

        plt.legend(loc=10,edgecolor='#000000',prop={'family': 'Times New Roman', 'size': 12})
        # plt.xlim(0,205)
        # plt.ylim(37.08,38.25)
        plt.grid(linestyle='-')
        plt.xticks(fontproperties="Times New Roman", size=15)
        plt.yticks(fontproperties="Times New Roman", size=15)
        plt.xlabel(x_labelName,labelpad=0, fontdict={'family' : 'Times New Roman', 'size': 15})
        plt.ylabel(y_labelName,labelpad=0,fontdict={'family' : 'Times New Roman', 'size': 15})

        # ax = plt.gca()
        # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))
        head = ""
        # for i in range(ue_seq_len):
        #     head = head + "_bs" + str(bs_num_seq[i]) + "_f" + str(ue_num_seq[i]) + "_u" + str(
        #         units_seq[i]) + "_a" + str(antenna_seq[i]) + str(r_min_seq[i]) + str(bw_seq[i]) +"_"+str(mec_rule_seq[i])+"_"+str(with_CoMP_seq[i])
        plt.savefig('.\仿真图\\'+'avg_total_power'+'.png',bbox_inches='tight')
        plt.savefig('.\仿真图\\'+'avg_total_power'+'.pdf',bbox_inches='tight')
        plt.close()
'''
plot_simulation_results_bs_power
'''
def plot_simulation_results_bs_power(ue_num_seq,bs_num_seq,units_seq,mec_rule_seq,with_CoMP_seq,antenna_seq,omega_seq,r_min_seq,bw_seq,x_labelName,y_labelName,sample_amount,start,disp_gap,fix_ue_seq,fix_antenna_seq,disp_elements):
    ue_seq_len = len(ue_num_seq)
    bs_seq_len = len(bs_num_seq)

    label_lst_noRIS = ["8 FoVs without RIS","6 FoVs without RIS"]
    label_lst = []
    indices = []
    end_x=0
    # label_lst = ['Ideal IRS','Non-ideal IRS with continuous phase shifts','Non-ideal IRS with discrete phase shifts'
    #     ,'No IRS']
    if(ue_seq_len!=bs_seq_len):
        print("输入的序列长度不对齐")
        return
    else:
        model_path_seq=[]
        result_npy_seq=[]
        result_npy_noRIS_seq=[]
        records = []
        avg_power_seq = []
        long_term_avg_power_seq=[]
        result_npy_noRIS_seq.append(
            "..\DRL\simulation_result\\full\\bs_power_record_NoRIS_ue8_bs3_u30_a5_r30_improved_concern_all_DDQN.npy")
        result_npy_noRIS_seq.append(
            "..\DRL\simulation_result\\full\\bs_power_record_NoRIS_ue6_bs3_u30_a5_r30_improved_concern_all_DDQN.npy")
        for i in range(ue_seq_len):
            addition = ""
            if (mec_rule_seq[i] == "max"):
                addition = "mecRuleMax"
            elif (mec_rule_seq[i] == "min"):
                addition = "mecRuleMin"
            if (with_CoMP_seq[i] == False):
                addition = addition + "_noCoMP"
            head = '_ue%d_bs%d_u%d_a%d_r%d_%simproved_concern_all_DDQN' % (ue_num_seq[i], bs_num_seq[i], units_seq[i], antenna_seq[i], r_min_seq[i],  addition)
            head_omega=  '_ue%d_bs%d_u%d_a%d' % (ue_num_seq[i], bs_num_seq[i], units_seq[i], antenna_seq[i])
            # model_path_seq.append("..\DRL\simulation_result\\full\model_ue"+str(ue_num_seq[i])+"_bs"+str(bs_num_seq[i])+"_u"+str(units_seq[i])+"_a"+str(antenna_seq[i])+'_r'+str(r_min_seq[i])+'_bw'+str(bw_seq[i])+".ckpt")
            if(omega_seq[i]==True):
                result_npy_seq.append(
                    "..\DRL\simulation_result\\full\\bs_power_randOmega_record" + head_omega + ".npy")
                addition = addition + "random"
            else:
                addition = addition + "with RIS"
                result_npy_seq.append(
                    "..\DRL\simulation_result\\full\\bs_power_record" + head + ".npy")

            # label_lst.append(str(bs_num_seq[i])+"b,"+str(ue_num_seq[i])+"f,"+str(units_seq[i])+"u,"+str(antenna_seq[i])+'a,'+str(r_min_seq[i])+'r'+addition)
            label = ""
            if (disp_elements[0] == True):
                label = label + "B=" + str(bs_num_seq[i]) + " "
            if (disp_elements[1] == True):
                label = label + "%s FoVs"%str(ue_num_seq[i]) + " "
            if (disp_elements[2] == True):
                label = label + "Q=" + str(units_seq[i]) + " "
            if (disp_elements[3] == True):
                label = label + "K=" + str(antenna_seq[i]) + " "
            if (disp_elements[4] == True):
                label = label + "R_min=" + str(r_min_seq[i]) + " "
            label_lst.append(label+addition)
        for i in range(ue_seq_len):
                collections = []
                avg_power = np.load(result_npy_seq[i])
                avg_power[np.isnan(avg_power)] = 0

                if(len(avg_power.shape)>1):
                    avg_power = np.mean(avg_power,axis=-1)
                for index in range(avg_power.shape[0]):
                    if (avg_power[index] > 0 ):
                        collections.append(index)
                indices.append(np.array(collections))
        indices = np.array(indices)
        indices  = reduce(np.intersect1d,indices)
        indices = np.tile(indices,[1000])
        # 画无RIS功率
        long_term_avg_power_noRIS_seq = []
        for i in range(2):
            avg_power = 10 * np.log10( 3 *np.load(result_npy_noRIS_seq[i])[indices] / 0.001)
            if (len(avg_power.shape) > 1):
                avg_power = np.mean(avg_power, axis=-1)
            # records.append(avg_power)
            # avg_power =mec_power_seq[i]-0.5*avg_reward/bs_num_seq[i]
            long_term_avg_power_noRIS = []
            avg_power_seq.append(avg_power)
            sum_power = 0
            for time in range(len(avg_power)):
                sum_power = sum_power + avg_power[time]
                long_term_avg_power_noRIS.append(sum_power / (time + 1))
                # long_term_avg_power_noRIS.append(avg_power[time])
            long_term_avg_power_noRIS_seq.append(long_term_avg_power_noRIS)
        long_term_avg_power_noRIS_seq = np.array(long_term_avg_power_noRIS_seq)

        for i in range(ue_seq_len):
            avg_power = 10*np.log10(3*np.load(result_npy_seq[i])/0.001)
            if (len(avg_power.shape) > 1):
                avg_power = np.mean(avg_power, axis=-1)
            avg_power = avg_power[indices]
            # records.append(avg_power)
            # avg_power =mec_power_seq[i]-0.5*avg_reward/bs_num_seq[i]
            long_term_avg_power = []
            avg_power_seq.append(avg_power)
            sum_power= 0
            for time in range(len(avg_power)):
                sum_power=sum_power+avg_power[time]
                long_term_avg_power.append(sum_power/(time+1))
                # long_term_avg_power.append(avg_power[time])
            long_term_avg_power_seq.append(long_term_avg_power)
        long_term_avg_power_seq = np.array(long_term_avg_power_seq)
        plt.figure()
        # ax = brokenaxes(xlims=[(0, sample_amount-start-1)], ylims=((6,6.45), (20, 20.4)), despine=False,
        #                 hspace=0.15, d=0.02)  # (6.27,6.30),(12.08,12.09),(25.225,25.24)
        # ax = brokenaxes(xlims=[(0, sample_amount - start -1+5)], ylims=((9.15,9.75),(10.8,11.3),(23,23.5),(24.7,25.0)), despine=False,
        #                 hspace=0.18, d=0.005)  # (6.27,6.30),(12.08,12.09),(25.225,25.24)
        ax = brokenaxes(xlims=[(0, sample_amount - start - 1 + 5)],
                        ylims=((9.2, 9.9), (11.0, 11.8), (23.3, 23.7), (25.0, 25.5)), despine=False,
                        hspace=0.18, d=0.005)  # (6.27,6.30),(12.08,12.09),(25.225,25.24)


        for j in range(0,2):
            # times_sum_rate[:10000,j] = times_sum_rate[10000,j]
            # y = avg_power_seq[j]
            y = np.array(long_term_avg_power_seq[j])[start:sample_amount]
            x= np.arange(len(y))
            if(len(y)>5):
                x=np.arange(0,len(y),disp_gap)
                # end_x  =x
                y=y[x]

            ax.plot(x,y,linewidth = 1, color =color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                     marker = marker_lst[j],markersize=5,label = label_lst[j])

        y = np.array(long_term_avg_power_noRIS_seq[0])[start:sample_amount]
        x = np.arange(len(y))
        if (len(y) > 5):
            x = np.arange(0, len(y), disp_gap)
            # end_x  =x
            y = y[x]
        ax.plot(x, y, linewidth=1, color=color_lst[2], markerfacecolor=markerfacecolor_lst[2],
                marker=marker_lst[2], markersize=5, label=label_lst_noRIS[0])

        for j in range(2,ue_seq_len):
            # times_sum_rate[:10000,j] = times_sum_rate[10000,j]
            # y = avg_power_seq[j]
            y = np.array(long_term_avg_power_seq[j])[start:sample_amount]
            x = np.arange(len(y))
            if (len(y) > 5):
                x = np.arange(0, len(y), disp_gap)
                # end_x  =x
                y = y[x]

            ax.plot(x, y, linewidth=1, color=color_lst[j+1], markerfacecolor=markerfacecolor_lst[j+1],
                    marker=marker_lst[j+1], markersize=5, label=label_lst[j])


        # for j in range(2):
        #     # times_sum_rate[:10000,j] = times_sum_rate[10000,j]
        #     # y = avg_power_seq[j]
        #     y = np.array(long_term_avg_power_noRIS_seq[j])[start:sample_amount]
        #     x = np.arange(len(y))
        #     if (len(y) > 5):
        #         x = np.arange(0, len(y), disp_gap)
        #         # end_x  =x
        #         y = y[x]
        #
        #     ax.plot(x, y, linewidth=1.5, color=color_lst[4+j], markerfacecolor=markerfacecolor_lst[4+j],
        #              marker=marker_lst[4+j], markersize=5, label=label_lst_noRIS[j])

        y = np.array(long_term_avg_power_noRIS_seq[1])[start:sample_amount]
        x = np.arange(len(y))
        if (len(y) > 5):
            x = np.arange(0, len(y), disp_gap)
            # end_x  =x
            y = y[x]
        ax.plot(x, y, linewidth=1, color=color_lst[5], markerfacecolor=markerfacecolor_lst[5],
                marker=marker_lst[5], markersize=5, label=label_lst_noRIS[1])
        #bs_power_record_NoRIS_ue8_bs3_u20_a5_r30
        #bs_power_record_NoRIS_ue6_bs3_u20_a5_r30




        ax.legend(loc=0,edgecolor='#000000',prop={'family': 'Times New Roman', 'size': 12})
        # plt.xlim(10000,100000)
        # plt.ylim(20,100)
        ax.grid(linestyle='-')
        # for sub_ax in ax.axs:
        #     for tick in sub_ax.get_major_ticks():
        #         tick.label.set_fontsize(15)

        for tick in ax.axs[0].yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax.axs[1].yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax.axs[2].yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax.axs[3].yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        for tick in ax.axs[3].xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
        ax.axs[3].set_xlabel(x_labelName, labelpad=0, fontdict={'family': 'Times New Roman', 'size': 15})
        ax.set_ylabel(y_labelName, labelpad=42, fontdict={'family': 'Times New Roman', 'size': 15})

        # ax.set_xlabel(x_labelName, fontdict={'family': 'Times New Roman', 'size': 15})
        # ax.set_ylabel(y_labelName, fontdict={'family': 'Times New Roman', 'size': 15})
        # grid_margin = MultipleLocator(disp_gap)
        # for axs in ax.axs:
        #     axs.xaxis.set_major_locator(grid_margin)
        # plt.xlabel(x_labelName,fontdict={'family' : 'Times New Roman', 'size': 12})
        # plt.ylabel(y_labelName,fontdict={'family' : 'Times New Roman', 'size': 12})
        # ax = plt.gca()
        # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))
        head = ""
        for i in range(ue_seq_len):
            head = head + "_b" + str(bs_num_seq[i]) + "_f" + str(ue_num_seq[i]) + "_u" + str(units_seq[i]) + "_a" + str(
                antenna_seq[i]) + str(r_min_seq[i]) + str(bw_seq[i])
            if (mec_rule_seq[i] == "max"):
                head = head + "_a"
            elif (mec_rule_seq[i] == "min"):
                head = head + "_i"
            if (with_CoMP_seq[i] == True):
                head = head + "_t"
            else:
                head = head + "_f"
            if(omega_seq[i]==True):
                head = head + "_t"
            else:
                head = head + "_f"
        plt.savefig('.\仿真图\\''avg_bs_power3'+'.png', bbox_inches='tight')
        plt.savefig('.\仿真图\\''avg_bs_power3'+'.pdf', bbox_inches='tight')
        plt.close()
'''
plot_simulation_results_ue_rates
'''
def plot_simulation_results_ue_rates(ue_num_seq,bs_num_seq,units_seq,mec_rule_seq,with_CoMP_seq,antenna_seq,r_min_seq,bw_seq,x_labelName,y_labelName,sample_amount,start,disp_gap,disp_elements):
    ue_seq_len = len(ue_num_seq)
    bs_seq_len = len(bs_num_seq)


    label_lst = []
    indices = []
    end_x=0
    # label_lst = ['Ideal IRS','Non-ideal IRS with continuous phase shifts','Non-ideal IRS with discrete phase shifts'
    #     ,'No IRS']
    if(ue_seq_len!=bs_seq_len and ue_seq_len!=mec_power_seq_len):
        print("输入的序列长度不对齐")
        return
    else:
        model_path_seq=[]
        result_npy_seq=[]
        records = []
        avg_power_seq = []
        long_term_avg_rates_seq=[]

        for i in range(ue_seq_len):

            base = ""
            addition = ""
            addition_label = ""
            if (mec_rule_seq[i] == "max"):
                addition = " mecRuleMax"
                addition_label = " max"
                base = "baseline "
            elif (mec_rule_seq[i] == "min"):
                addition = "_mecRuleMin"
                addition_label = " min"
                base = "baseline "
            if (with_CoMP_seq[i] == False):
                addition = addition + " noCoMP"
                addition_label = addition_label+" noCoMP"
            # np.save('.\simulation_result\\full\\ue_avg_rates_record' + head, np.array(scene.ue_avg_rates_record))
            # np.save('.\simulation_result\\full\\ue_avg_rates_record_NoRIS' + head,
            #         np.array(scene.ue_avg_rates_record_NoRIS))
            head = '_ue%d_bs%d_u%d_a%d_r%d_%s' % (ue_num_seq[i], bs_num_seq[i], units_seq[i], antenna_seq[i], r_min_seq[i], bw_seq[i], addition)
            model_path_seq.append("..\DRL\simulation_result\\full\model_ue"+str(ue_num_seq[i])+"_bs"+str(bs_num_seq[i])+"_u"+str(units_seq[i])+"_a"+str(antenna_seq[i])+'_r'+str(r_min_seq[i])+'_bw'+str(bw_seq[i])+".ckpt")
            result_npy_seq.append("..\DRL\simulation_result\\full\\ue_avg_rates_record"+head+".npy")
            # label_lst.append(str(bs_num_seq[i])+"BS,"+str(ue_num_seq[i])+"Fov,"+str(units_seq[i])+"RISUnits,"+str(antenna_seq[i])+'Antenna,'+addition_label)
            label = base
            if (disp_elements[0] == True):
                label = label + "bs:" + str(bs_num_seq[i]) + " "
            if (disp_elements[1] == True):
                label = label + "fov:" + str(ue_num_seq[i]) + " "
            if (disp_elements[2] == True):
                label = label + "units:" + str(units_seq[i]) + " "
            if (disp_elements[3] == True):
                label = label + "antenna:" + str(antenna_seq[i]) + " "
            if (disp_elements[4] == True):
                label = label + "min_rate:" + str(r_min_seq[i]) + " "
            label_lst.append(label)
        for i in range(ue_seq_len):
                collections = []
                avg_power = np.load(result_npy_seq[i])
                for index in range(avg_power.shape[0]):
                    if (avg_power[index] > 0):
                        collections.append(index)
                indices.append(np.array(collections))
        indices = reduce(np.intersect1d, indices)
        indices = np.tile(indices,[1000])
        for i in range(ue_seq_len):
            avg_power = np.load(result_npy_seq[i])[indices][start:sample_amount]
            # records.append(avg_power)
            # avg_power =mec_power_seq[i]-0.5*avg_reward/bs_num_seq[i]
            long_term_avg_rates = []
            avg_power_seq.append(avg_power)
            sum_power= 0
            for time in range(len(avg_power)):
                sum_power=sum_power+avg_power[time]
                long_term_avg_rates.append(sum_power/(time+1))
            long_term_avg_rates_seq.append(long_term_avg_rates)
        long_term_avg_rates_seq = np.array(long_term_avg_rates_seq)
        plt.figure()
        plt.grid(linestyle='-')
        plt.xticks(fontproperties="Times New Roman", size=15)
        plt.yticks(fontproperties="Times New Roman", size=15)
        plt.xlabel(x_labelName,fontdict={'family' : 'Times New Roman', 'size': 15})
        plt.ylabel(y_labelName,fontdict={'family' : 'Times New Roman', 'size': 15})
        for j in range(len(label_lst)):
            # times_sum_rate[:10000,j] = times_sum_rate[10000,j]
            # y = avg_power_seq[j]
            y = np.array(long_term_avg_rates_seq[j])
            x= np.arange(len(y))
            if(len(y)>5):
                x=np.arange(0,len(y),disp_gap)
                # end_x  =x
                y=y[x]

            plt.plot(x,y,linewidth = 1.5, color =color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                     marker = marker_lst[j],markersize=8,label = label_lst[j])
            # 画固定功率

        plt.legend(loc=0,edgecolor='#000000',prop={'family': 'Times New Roman', 'size': 12})
        # plt.xlim(10000,100000)
        # plt.ylim(20,100)


        # ax = plt.gca()
        # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))
        head = ""
        for i in range(ue_seq_len):
            head = head + "_b" + str(bs_num_seq[i]) + "_f" + str(ue_num_seq[i]) + "_u" + str(units_seq[i]) + "_a" + str(
                antenna_seq[i]) + str(r_min_seq[i]) + str(bw_seq[i])
            if (mec_rule_seq[i] == "max"):
                head = head + "_a"
            elif (mec_rule_seq[i] == "min"):
                head = head + "_i"
            if (with_CoMP_seq[i] == True):
                head = head + "_t"
            else:
                head = head + "_f"
        plt.savefig('.\仿真图\\'+'avg_sum_rates.png')
        plt.savefig('.\仿真图\\'+'avg_sum_rates.pdf')
        plt.close()

'''
plot_simulation_results_ue_rates
'''
def plot_simulation_results_ue_rates(ue_num_seq,bs_num_seq,units_seq,mec_rule_seq,with_CoMP_seq,antenna_seq,r_min_seq,bw_seq,x_labelName,y_labelName,sample_amount,start,disp_gap,disp_elements):
    ue_seq_len = len(ue_num_seq)
    bs_seq_len = len(bs_num_seq)
    # mec_power_seq_len = len(mec_power_seq)
    label_lst = []
    indices = []
    end_x=0
    # label_lst = ['Ideal IRS','Non-ideal IRS with continuous phase shifts','Non-ideal IRS with discrete phase shifts'
    #     ,'No IRS']
    if(ue_seq_len!=bs_seq_len):
        print("输入的序列长度不对齐")
        return
    else:
        model_path_seq=[]
        result_npy_seq=[]
        result_npy_seq_noCoMP = []
        records = []
        sum_rate_seq = []

        long_term_sum_rates_seq=[]

        for i in range(ue_seq_len):
            base = ""
            addition = ""
            prior = ""
            addition_label = ""
            if (mec_rule_seq[i] == "max"):
                addition = "_mecRuleMax"
                addition_label = " max"
                base = "baseline "
            elif (mec_rule_seq[i] == "min"):
                addition = "_mecRuleMin"
                addition_label = " min"
                base = "baseline "
            if (with_CoMP_seq[i] == False):
                prior = "_noCoMP"
                addition_label = " without CoMP"
            else:
                addition_label = " with CoMP"
            # np.save('.\simulation_result\\full\\ue_avg_rates_record' + head, np.array(scene.ue_avg_rates_record))
            # np.save('.\simulation_result\\full\\ue_avg_rates_record_NoRIS' + head,
            #         np.array(scene.ue_avg_rates_record_NoRIS))
            head = '_ue%d_bs%d_u%d_a%d_r%d%s_improved_concern_all_DDQN' % (ue_num_seq[i], bs_num_seq[i], units_seq[i], antenna_seq[i], r_min_seq[i],addition)
            # model_path_seq.append("..\DRL\simulation_result\\full\model_ue"+str(ue_num_seq[i])+"_bs"+str(bs_num_seq[i])+"_u"+str(units_seq[i])+"_a"+str(antenna_seq[i])+'_r'+str(r_min_seq[i])+'_bw'+str(bw_seq[i])+".ckpt")
            result_npy_seq.append("..\DRL\simulation_result\\full\\ue_avg_rates_record"+prior+head+".npy")
            # label_lst.append(str(bs_num_seq[i])+"BS,"+str(ue_num_seq[i])+"Fov,"+str(units_seq[i])+"RISUnits,"+str(antenna_seq[i])+'Antenna,'+addition_label)
            label = base
            if (disp_elements[0] == True):
                label = label + "bs=" + str(bs_num_seq[i]) + " "
            if (disp_elements[1] == True):
                label = label + "fov=" + str(ue_num_seq[i]) + " "
            if (disp_elements[2] == True):
                label = label + "units:" + str(units_seq[i]) + " "
            if (disp_elements[3] == True):
                label = label + "antenna:" + str(antenna_seq[i]) + " "
            if (disp_elements[4] == True):
                label = label + "%s Mbps"%str(r_min_seq[i]) + " "
            label_lst.append(label+addition_label)

        rates=[]
        for i in range(ue_seq_len):
                collections = []
                rate = np.load(result_npy_seq[i])
                for index in range(rate.shape[0]):
                    if (rate[index]!=-1):
                        r = np.array(rate[index])
                        collections.append(index)
                indices.append(np.array(collections))
        indices = reduce(np.intersect1d, indices)
        indices = np.tile(indices,[1000])
        for i in range(ue_seq_len):
            rate = np.load(result_npy_seq[i])[indices]*ue_num_seq[i]
            # avg_power = np.load(result_npy_seq[i])
            # records.append(avg_power)
            # avg_power =mec_power_seq[i]-0.5*avg_reward/bs_num_seq[i]
            long_term_sum_rates = []
            sum_rate= 0
            for time in range(len(rate)):
                sum_rate=sum_rate+rate[time]
                long_term_sum_rates.append(sum_rate/(time+1))
                # long_term_sum_rates.append(rate[time])
            long_term_sum_rates_seq.append(long_term_sum_rates)
        long_term_sum_rates_seq = np.array(long_term_sum_rates_seq)
        plt.figure()
        # ax = brokenaxes(xlims=[(0, 205)], ylims=((775, 880), (1125, 1275)), despine=False, hspace=0.1, d=0.005)
        # ax = brokenaxes(xlims=[(0, 205)], ylims=((900, 1300)), despine=False, hspace=0.1, d=0.005)
        ax = plt.subplots(1,1)
        for j in range(len(label_lst)):
            # times_sum_rate[:10000,j] = times_sum_rate[10000,j]
            # y = avg_power_seq[j]
            y = np.array(long_term_sum_rates_seq[j])[start:sample_amount]
            x= np.arange(len(y))
            if(len(y)>5):
                x=np.arange(0,len(y),disp_gap)
                # end_x  =x
                y=y[x]

            plt.plot(x,y,linewidth = 1.5, color =color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                     marker = marker_lst[j],markersize=5,label = label_lst[j])
            # 画固定功率
        # for tick in ax.axs[0].xaxis.get_major_ticks():
        #     tick.label.set_fontsize(15)
        # for tick in ax.axs[0].yaxis.get_major_ticks():
        #     tick.label.set_fontsize(15)
        # for tick in ax.axs[1].xaxis.get_major_ticks():
        #     tick.label.set_fontsize(15)
        # for tick in ax.axs[1].yaxis.get_major_ticks():
        #     tick.label.set_fontsize(15)
        plt.legend(loc=0,edgecolor='#000000',prop={'family': 'Times New Roman', 'size': 12})
        # plt.xlim(0,205)
        # plt.ylim(0,205)
        plt.grid(linestyle='-')
        # ax.axs[1].set_xlabel(x_labelName, labelpad=0, fontdict={'family': 'Times New Roman', 'size': 15})
        # ax.set_ylabel(y_labelName, labelpad=35, fontdict={'family': 'Times New Roman', 'size': 15})


        plt.grid(linestyle='-')
        plt.xticks(fontproperties="Times New Roman", size=15)
        plt.yticks(fontproperties="Times New Roman", size=15)
        plt.xlabel(x_labelName, fontdict={'family': 'Times New Roman', 'size': 15})
        plt.ylabel(y_labelName, fontdict={'family': 'Times New Roman', 'size': 15})
        # ax.set_xticks(fontfamily ="Times New Roman", fontsize=15)
        # ax.set_yticks(fontfamily="Times New Roman", fontsize=15)
        # ax = plt.gca()
        # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))
        head = ""
        # for i in range(ue_seq_len):
        #     head = head + "_b" + str(bs_num_seq[i]) + "_f" + str(ue_num_seq[i]) + "_u" + str(units_seq[i]) + "_a" + str(
        #         antenna_seq[i]) + str(r_min_seq[i]) + str(bw_seq[i])
        #     if (mec_rule_seq[i] == "max"):
        #         head = head + "_a"
        #     elif (mec_rule_seq[i] == "min"):
        #         head = head + "_i"
        #     if (with_CoMP_seq[i] == True):
        #         head = head + "_t"
        #     else:
        #         head = head + "_f"
        plt.savefig('.\仿真图\\'+'avg_sum_rates'+'.png',bbox_inches='tight')
        plt.savefig('.\仿真图\\'+'avg_sum_rates'+'.pdf',bbox_inches='tight')
        plt.close()

'''
plot_simulation_results_bs_power_diffRIS
'''
def plot_simulation_results_bs_power_diffRIS(ue_num_seq, bs_num_seq,units_seq,antenna_seq,r_min_seq,bw_seq, mec_rule_seq,with_CoMP_seq,x_labelName, y_labelName,disp_amount,start,disp_gap,disp_elements):
            ue_seq_len = len(ue_num_seq)
            bs_seq_len = len(bs_num_seq)
            irs_units_seq_len = len(units_seq)


            addition = ""
            addition_label=""
            label_lst = []
            end_x = 0
            # label_lst = ['Ideal IRS','Non-ideal IRS with continuous phase shifts','Non-ideal IRS with discrete phase shifts'
            #     ,'No IRS']
            if (ue_seq_len != bs_seq_len and ue_seq_len != mec_power_seq_len and ue_seq_len!=irs_units_seq_len):
                print("输入的序列长度不对齐")
                return
            else:
                model_path_seq = []
                result_npy_seq = []
                result_npy_seq_NoRIS = []
                records = []
                avg_power_seq = []
                avg_power_NoRIS_seq=[]
                long_term_avg_power_seq = []
                long_term_avg_power_NoRIS_seq = []
                diff_seq=[]
                indices = []
                for i in range(ue_seq_len):
                    base = ""
                    addition = ""
                    addition_label = ""
                    if (mec_rule_seq[i] == "max"):
                        addition = "_mecRuleMax"
                        addition_label= " max"
                        base = "baseline "
                    elif (mec_rule_seq[i] == "min"):
                        addition = "_mecRuleMin"
                        addition_label = " min"
                        base = "baseline "

                    if (with_CoMP_seq[i] == False):
                        addition = addition + "_noCoMP"

                    head = '_ue%d_bs%d_u%d_a%d_r%d%s' % (ue_num_seq[i], bs_num_seq[i], units_seq[i], antenna_seq[i], r_min_seq[i], addition)
                    model_path_seq.append(
                        "..\DRL\simulation_result\\full\\model_ue" + str(ue_num_seq[i]) + "_bs" + str(bs_num_seq[i]) +"_u"+str(units_seq[i])+"_a"+str(antenna_seq[i])+'_r'+str(r_min_seq[i])+'_bw'+str(bw_seq[i])+ ".ckpt")
                    result_npy_seq.append(
                        "..\DRL\simulation_result\\full\\bs_power_record" + head+ ".npy")
                    result_npy_seq_NoRIS.append(
                        "..\DRL\simulation_result\\full\\bs_power_record_NoRIS" +head+ ".npy")
                    # label_lst.append(base+str(bs_num_seq[i]) + "b,"+str(ue_num_seq[i]) + "f," + str(units_seq[i]) + "u," + str(antenna_seq[i]) + 'a,' + str(r_min_seq[i]) + 'r'+addition_label)
                    label=base
                    if(disp_elements[0]==True):
                        label = label+"bs:"+str(bs_num_seq[i]) +" "
                    if (disp_elements[1] == True):
                        label = label +"fov:"+str(ue_num_seq[i]) +" "
                    if (disp_elements[2] == True):
                        label =  label+ "units:"+ str(units_seq[i])+" "
                    if (disp_elements[3] == True):
                        label = label + "antenna:"+ str(antenna_seq[i]) + " "
                    if (disp_elements[4] == True):
                        label = label + "min_rate:"+ str(r_min_seq[i]) + " "
                    label_lst.append(label+addition_label)

                for i in range(ue_seq_len):
                    collections=[]
                    data = np.load(result_npy_seq[i])
                    for index in range(data.shape[0]):
                        if(data[index]>0):
                            data[index] = data[index]
                            collections.append(index)
                    collections = np.array(collections)
                    indices.append(collections)
                indices = reduce(np.intersect1d, indices)
                indices = np.tile(indices,[1000])
                # common_indices = indices[0]
                # for i in range(1,ue_seq_len):
                #     common_indices = np.intersect1d(common_indices,indices[i])
                for i in range(ue_seq_len):
                    # label_lst.append("bs_num=" + str(bs_num_seq[i]) + ",fov_num=" + str(ue_num_seq[i])+"units_num="+str(irs_units_seq[i])+"with RIS")
                    avg_power = 10*np.log10(np.load(result_npy_seq[i],allow_pickle=True)[indices]/0.001)
                    avg_power_NoRIS = 10*np.log10(np.load(result_npy_seq_NoRIS[i],allow_pickle=True)[indices]/0.001)
                    avg_power = avg_power[start:disp_amount]
                    avg_power_NoRIS = avg_power_NoRIS[start:disp_amount]
                    # records.append(avg_power)
                    # avg_power =mec_power_seq[i]-0.5*avg_reward/bs_num_seq[i]
                    long_term_avg_power = []
                    long_term_NoRIS_avg_power=[]
                    avg_power_seq.append(avg_power)
                    avg_power_NoRIS_seq.append(avg_power_NoRIS)
                    sum_power = 0
                    record_seq = []
                    sum_diff = 0
                    sum_power_NoRIS = 0
                    for time in range(len(avg_power)):
                        sum_power = sum_power + avg_power[time]
                        sum_diff =sum_diff+( avg_power_NoRIS[time]-avg_power[time])
                        sum_power_NoRIS= sum_power_NoRIS+avg_power_NoRIS[time]
                        # record_seq.append(sum_diff/ (time + 1))
                        record_seq.append(sum_diff/time+1)
                        # long_term_avg_power.append(sum_power / (time + 1))
                        # long_term_NoRIS_avg_power.append(sum_power_NoRIS/(time+1))
                    diff_seq.append(record_seq)
                    # long_term_avg_power_seq.append(long_term_avg_power)
                    # long_term_avg_power_NoRIS_seq.append(long_term_NoRIS_avg_power)
                long_term_avg_power_seq = np.array(long_term_avg_power_seq)

                plt.figure()

                for j in range(len(label_lst)):
                    # times_sum_rate[:10000,j] = times_sum_rate[10000,j]
                    # y=long_term_avg_power[j]-long_term_avg_power_NoRIS_seq[j]
                    y = diff_seq[j]
                    x =  np.arange(len(y))
                    if(len(y)>5):
                        x_slice=np.arange(0,len(y),disp_gap)
                        # end_x  =x
                        y_slice=np.array(y)[x_slice]
                    plt.plot(x_slice, y_slice, linewidth=1.5, color=color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                             marker=marker_lst[j], markersize=8, label=label_lst[j])
                    # y = avg_power_NoRIS_seq[j]
                    # x = np.arange(y.shape[0])
                    # if(len(y)>5):
                    #     x_slice=np.arange(0,len(y),5)
                    #     # end_x  =x
                    #     y_slice=y[x_slice]

                    # plt.plot(x, y, linewidth=1.5, color=color_lst[j+1], markerfacecolor=markerfacecolor_lst[j+1],
                    #          marker=marker_lst[j+1], markersize=8, label=label_lst[j+1])

                plt.legend(loc=0, edgecolor='#000000', prop={'family': 'Times New Roman', 'size': 11})
                # plt.xlim(10000,100000)
                # plt.ylim(20,100)
                plt.grid(linestyle='-')
                plt.xlabel(x_labelName, fontdict={'family': 'Times New Roman', 'size': 12})
                plt.ylabel(y_labelName, fontdict={'family': 'Times New Roman', 'size': 12})
                # ax = plt.gca()
                # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))
                head = ""

                for i in range(ue_seq_len):
                    head = head + "_b" + str(bs_num_seq[i]) + "_f" + str(ue_num_seq[i])+"_u"+str(units_seq[i])+"_a"+str(antenna_seq[i])+str(r_min_seq[i])+str(bw_seq[i])
                    if(mec_rule_seq[i]=="max"):
                        head = head+"_a"
                    elif(mec_rule_seq[i]=="min"):
                        head = head +"_i"
                    if(with_CoMP_seq[i]==True):
                        head = head + "_t"
                    else:
                        head = head + "_f"

                print('avg_sum_bs_power_diff_RIS'+  head + '.png')
                plt.savefig('.\仿真图\\'+'avg_sum_bs_power_diff_RIS'+head + '.png')
                plt.savefig('.\仿真图\\'+'avg_sum_bs_power_diff_RIS'+ head + '.pdf')
                plt.close()

'''
plot_simulation_results_bs_power_YNRIS
'''
def plot_simulation_results_bs_power_YNRIS(ue_num_seq, bs_num_seq, units_seq, antenna_seq, r_min_seq, bw_seq,mec_rule_seq, x_labelName, y_labelName, disp_amount, start,disp_gap, disp_elements):
    ue_seq_len = len(ue_num_seq)
    bs_seq_len = len(bs_num_seq)
    irs_units_seq_len = len(units_seq)

    # mec_power_seq_len = len(mec_power_seq)

    addition = ""
    addition_label = ""
    label_lst = []
    end_x = 0
    # label_lst = ['Ideal IRS','Non-ideal IRS with continuous phase shifts','Non-ideal IRS with discrete phase shifts'
    #     ,'No IRS']
    if (ue_seq_len != bs_seq_len  and ue_seq_len != irs_units_seq_len):
        print("输入的序列长度不对齐")
        return
    else:
        model_path_seq = []
        result_npy_seq = []
        result_npy_seq_NoCoMP=[]
        result_npy_seq_NoRIS = []
        result_npy_seq_NoRIS_NoCoMP=[]
        records = []
        avg_power_seq = []
        avg_power_NoRIS_seq = []
        long_term_avg_power_RIS_seq = []
        long_term_avg_power_NoRIS_seq = []
        diff_seq = []
        indices = []
        for i in range(ue_seq_len):
            base = ""
            addition = ""
            addition_label = ""
            if (mec_rule_seq[i] == "max"):
                addition = "_mecRuleMax"
                addition_label = " max"
                base = "baseline "
            elif (mec_rule_seq[i] == "min"):
                addition = "_mecRuleMin"
                addition_label = " min"
                base = "baseline "


            head = '_ue%d_bs%d_u%d_a%d_r%d%s' % (ue_num_seq[i], bs_num_seq[i], units_seq[i], antenna_seq[i], r_min_seq[i], addition)

            result_npy_seq.append(
                "..\DRL\simulation_result\\full\\bs_power_record" + head + ".npy")
            result_npy_seq_NoRIS.append(
                "..\DRL\simulation_result\\full\\bs_power_record_NoRIS" + head + ".npy")
            # result_npy_seq_NoCoMP.append(
            #     "..\DRL\simulation_result\\full\\bs_power_record" + head + ".npy"
            # )
            # result_npy_seq_NoRIS_NoCoMP.append(
            #     "..\DRL\simulation_result\\full\\bs_power_record_NoRIS" + head + ".npy")

            model_path_seq.append(
                "..\DRL\simulation_result\\full\\model_ue" + str(ue_num_seq[i]) + "_bs" + str(
                    bs_num_seq[i]) + "_u" + str(units_seq[i]) + "_a" + str(antenna_seq[i]) + '_r' + str(
                    r_min_seq[i]) + '_bw' + str(bw_seq[i]) + ".ckpt")

            # label_lst.append(base+str(bs_num_seq[i]) + "b,"+str(ue_num_seq[i]) + "f," + str(units_seq[i]) + "u," + str(antenna_seq[i]) + 'a,' + str(r_min_seq[i]) + 'r'+addition_label)
            label = base
            if (disp_elements[0] == True):
                label = label + "bs:" + str(bs_num_seq[i]) + " "
            if (disp_elements[1] == True):
                label = label + "fov:" + str(ue_num_seq[i]) + " "
            if (disp_elements[2] == True):
                label = label + "units:" + str(units_seq[i]) + " "
            if (disp_elements[3] == True):
                label = label + "antenna:" + str(antenna_seq[i]) + " "
            if (disp_elements[4] == True):
                label = label + "min_rate:" + str(r_min_seq[i]) + " "
            label_lst.append(label + addition_label)
        for i in range(ue_seq_len):
            collections = []
            data = np.load(result_npy_seq[i])
            avg_power_NoRIS = np.load(result_npy_seq_NoRIS[i])
            for index in range(data.shape[0]):
                if (data[index] > 0):
                    collections.append(index)
            collections = np.array(collections)
            indices.append(collections)
        indices = reduce(np.intersect1d, indices)
        indices = np.tile(indices, [100])
        # common_indices = indices[0]
        # for i in range(1,ue_seq_len):
        #     common_indices = np.intersect1d(common_indices,indices[i])
        for i in range(ue_seq_len):
            # label_lst.append("bs_num=" + str(bs_num_seq[i]) + ",fov_num=" + str(ue_num_seq[i])+"units_num="+str(irs_units_seq[i])+"with RIS")
            avg_power = 10*np.log10(np.load(result_npy_seq[i])[indices]/0.001)#转为DBM
            avg_power_NoRIS = 10*np.log10(np.load(result_npy_seq_NoRIS[i])[indices]/0.001)#转为DBM
            # rs = avg_power-avg_power_NoRIS
            # avg_power = 10*np.log10(avg_power/0.001)
            #
            # avg_power_NoRIS= 10*np.log10(avg_power_NoRIS/0.001)
            avg_power = avg_power
            avg_power_NoRIS = avg_power_NoRIS
            # records.append(avg_power)
            # avg_power =mec_power_seq[i]-0.5*avg_reward/bs_num_seq[i]
            long_term_avg_power = []
            long_term_NoRIS_avg_power = []
            avg_power_seq.append(avg_power)
            avg_power_NoRIS_seq.append(avg_power_NoRIS)
            sum_power_RIS = 0
            sum_power_NoRIS = 0
            record_seq_RIS = []
            record_seq_NoRIS = []
            sum_power_NoRIS = 0
            for time in range(len(avg_power)):
                sum_power_RIS = sum_power_RIS + avg_power[time]
                sum_power_NoRIS = sum_power_NoRIS + avg_power_NoRIS[time]
                # record_seq.append(sum_diff/ (time + 1))
                long_term_avg_power.append(float(sum_power_RIS)/float(time+1))
                long_term_NoRIS_avg_power.append(float(sum_power_NoRIS)/float(time+1))
            long_term_avg_power_RIS_seq.append(long_term_avg_power)
            long_term_avg_power_NoRIS_seq.append(long_term_NoRIS_avg_power)
        long_term_avg_power_RIS_seq = np.array(long_term_avg_power_RIS_seq)
        long_term_avg_power_NoRIS_seq = np.array(long_term_avg_power_NoRIS_seq)

        plt.figure()

        for j in range(len(label_lst)):
            y = long_term_avg_power_RIS_seq[j][start:disp_amount]
            if (len(y) > 5):
                x_slice = np.arange(0, len(y), disp_gap)
                y_slice = np.array(y)[x_slice]
            plt.plot(x_slice, y_slice, linewidth=1.5, color=color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                     marker=marker_lst[j], markersize=8, label=label_lst[j])
        for j in range(len(label_lst)):
            y = long_term_avg_power_NoRIS_seq[j][start:disp_amount]
            if (len(y) > 5):
                x_slice = np.arange(0, len(y), disp_gap)
                y_slice = np.array(y)[x_slice]
            plt.plot(x_slice, y_slice, linewidth=1.5, color=color_lst[len(label_lst)+j], markerfacecolor=markerfacecolor_lst[len(label_lst)+j],
                     marker=marker_lst[len(label_lst)+j], markersize=8, label=label_lst[j]+" NoRIS")


        plt.legend(loc=0, edgecolor='#000000', prop={'family': 'Times New Roman', 'size': 11})
        # plt.xlim(10000,100000)
        # plt.ylim(20,100)
        plt.grid(linestyle='-')
        plt.xlabel(x_labelName, fontdict={'family': 'Times New Roman', 'size': 12})
        plt.ylabel(y_labelName, fontdict={'family': 'Times New Roman', 'size': 12})
        # ax = plt.gca()
        # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))
        head = ""

        for i in range(ue_seq_len):
            head = head + "_b" + str(bs_num_seq[i]) + "_f" + str(ue_num_seq[i]) + "_u" + str(units_seq[i]) + "_a" + str(
                antenna_seq[i]) + str(r_min_seq[i]) + str(bw_seq[i])
            if (mec_rule_seq[i] == "max"):
                head = head + "_a"
            elif (mec_rule_seq[i] == "min"):
                head = head + "_i"
            else:
                head = head + "_f"

        print('avg_bs_power' + head + '.png')
        plt.savefig('.\仿真图\\' + 'avg_bs_power' + head + '.png')
        plt.savefig('.\仿真图\\' + 'avg_bs_power' + head + '.pdf')
        plt.close()

'''
plot_simulation_results_bs_power_rule
'''
def plot_simulation_results_bs_power_rule(ue_num_seq, bs_num_seq, units_seq, antenna_seq, r_min_seq, bw_seq,mec_rule_seq, x_labelName, y_labelName, disp_amount, start,disp_gap, disp_elements):
    ue_seq_len = len(ue_num_seq)
    bs_seq_len = len(bs_num_seq)
    irs_units_seq_len = len(units_seq)
    # mec_power_seq_len = len(mec_power_seq)
    addition = ""
    addition_label = ""
    label_lst = []
    end_x = 0
    # label_lst = ['Ideal IRS','Non-ideal IRS with continuous phase shifts','Non-ideal IRS with discrete phase shifts'
    #     ,'No IRS']
    if (ue_seq_len != bs_seq_len  and ue_seq_len != irs_units_seq_len):
        print("输入的序列长度不对齐")
        return
    else:
        model_path_seq = []
        result_npy_seq = []
        records = []
        avg_power_seq = []
        avg_power_NoRIS_seq = []
        long_term_avg_power_RIS_seq = []
        long_term_avg_power_NoRIS_seq = []
        diff_seq = []
        indices = []
        for i in range(ue_seq_len):
            base = ""
            addition = ""
            addition_label = ""
            if (mec_rule_seq[i] == "max"):
                addition = "_mecRuleMax"
                addition_label = "Max-AO"
                # base = "baseline "
            elif (mec_rule_seq[i] == "min"):
                addition = "_mecRuleMin"
                addition_label = "Min-AO"
                # base = "baseline "
            elif(mec_rule_seq[i]=='exhaustion'):
                addition='mecRuleExhaustion'
                addition_label='Exhaustion-AO'
            else:
                addition_label = "DDQN-AO"

            head = '_ue%d_bs%d_u%d_a%d_r%d%s_improved_concern_all_DDQN' % (ue_num_seq[i], bs_num_seq[i], units_seq[i], antenna_seq[i], r_min_seq[i],  addition)

            result_npy_seq.append(
                "..\DRL\simulation_result\\full\\bs_power_record" + head + ".npy")
            # result_npy_seq_NoCoMP.append(
            #     "..\DRL\simulation_result\\full\\bs_power_record" + head + ".npy"
            # )
            # result_npy_seq_NoRIS_NoCoMP.append(
            #     "..\DRL\simulation_result\\full\\bs_power_record_NoRIS" + head + ".npy")

            # model_path_seq.append(
            #     "..\DRL\simulation_result\\full\\model_ue" + str(ue_num_seq[i]) + "_bs" + str(
            #         bs_num_seq[i]) + "_u" + str(units_seq[i]) + "_a" + str(antenna_seq[i]) + '_r' + str(
            #         r_min_seq[i]) + '_bw' + str(bw_seq[i]) + ".ckpt")

            # label_lst.append(base+str(bs_num_seq[i]) + "b,"+str(ue_num_seq[i]) + "f," + str(units_seq[i]) + "u," + str(antenna_seq[i]) + 'a,' + str(r_min_seq[i]) + 'r'+addition_label)
            label = base
            if (disp_elements[0] == True):
                label = label + "B=" + str(bs_num_seq[i]) + " "
            if (disp_elements[1] == True):
                label = label + "%s FoVs"%str(ue_num_seq[i]) + " "
            if (disp_elements[2] == True):
                label = label + "Q=" + str(units_seq[i]) + " "
            if (disp_elements[3] == True):
                label = label + "K=" + str(antenna_seq[i]) + " "
            if (disp_elements[4] == True):
                label = label + "R_min=" + str(r_min_seq[i]) + " "
            label_lst.append(label + addition_label)
        for i in range(ue_seq_len):
            collections = []
            data = np.load(result_npy_seq[i])
            for index in range(data.shape[0]):
                if (data[index] > 0):
                    collections.append(index)
            collections = np.array(collections)
            indices.append(collections)
        indices = reduce(np.intersect1d, indices)
        indices = np.tile(indices, [5])
        # common_indices = indices[0]
        # for i in range(1,ue_seq_len):
        #     common_indices = np.intersect1d(common_indices,indices[i])
        for i in range(ue_seq_len):
            # label_lst.append("bs_num=" + str(bs_num_seq[i]) + ",fov_num=" + str(ue_num_seq[i])+"units_num="+str(irs_units_seq[i])+"with RIS")
            avg_power = 10*np.log10(3*np.load(result_npy_seq[i])[indices]/0.001)#转为DBM
            m = np.mean(avg_power)
            # rs = avg_power-avg_power_NoRIS
            # avg_power = 10*np.log10(avg_power/0.001)
            #
            # avg_power_NoRIS= 10*np.log10(avg_power_NoRIS/0.001)
            avg_power = avg_power
            # records.append(avg_power)
            # avg_power =mec_power_seq[i]-0.5*avg_reward/bs_num_seq[i]
            long_term_avg_power = []
            avg_power_seq.append(avg_power)
            sum_power_RIS = 0
            sum_power_NoRIS = 0
            record_seq_RIS = []
            for time in range(len(avg_power)):
                sum_power_RIS = sum_power_RIS + avg_power[time]
                # record_seq.append(sum_diff/ (time + 1))
                # long_term_avg_power.append(float(sum_power_RIS)/float(time+1))
                long_term_avg_power.append(float(sum_power_RIS) / float(time + 1))
            long_term_avg_power_RIS_seq.append(long_term_avg_power)
        long_term_avg_power_RIS_seq = np.array(long_term_avg_power_RIS_seq)

        plt.figure()
        # ax = brokenaxes(xlims=[(0, disp_amount-start)], ylims=((9,13.1), (15, 17)), despine=False,
        #                 hspace=0.08, d=0.005)  # (6.27,6.30),(12.08,12.09),(25.225,25.24)
        # plt.ylim([13,29])
        # plt.xlim([0,205])
        for j in range(len(label_lst)):
            y = long_term_avg_power_RIS_seq[j]
            y =y [start:disp_amount]
            if (len(y) > 5):
                x_slice = np.arange(0, len(y), disp_gap)
                y_slice = np.array(y)[x_slice]
            plt.plot(x_slice, y_slice, linewidth=1.5, color=color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                     marker=marker_lst[j], markersize=5, label=label_lst[j])

        # plt.xlim(0,205)
        plt.legend(loc=0, edgecolor='#000000', prop={'family': 'Times New Roman', 'size': 12})
        # plt.xlim(10000,100000)
        # plt.ylim(20,100)
        plt.grid(linestyle='-')
        # plt.xlabel(x_labelName, fontdict={'family': 'Times New Roman', 'size': 12})
        # plt.ylabel(y_labelName, fontdict={'family': 'Times New Roman', 'size': 12})

        # for tick in ax.axs[0].yaxis.get_major_ticks():
        #     tick.label.set_fontsize(15)
        # for tick in ax.axs[1].yaxis.get_major_ticks():
        #     tick.label.set_fontsize(15)
        # for tick in ax.axs[1].xaxis.get_major_ticks():
        #     tick.label.set_fontsize(15)
        # ax.axs[1].set_xlabel(x_labelName, labelpad=0, fontdict={'family': 'Times New Roman', 'size': 15})
        # ax.set_ylabel(y_labelName, labelpad=20, fontdict={'family': 'Times New Roman', 'size': 15})
        plt.xlabel(x_labelName, fontdict={'family': 'Times New Roman', 'size': 15})
        plt.ylabel(y_labelName, fontdict={'family': 'Times New Roman', 'size': 15})
        plt.xticks(fontproperties="Times New Roman", size=15)
        plt.yticks(fontproperties="Times New Roman", size=15)
        # ax = plt.gca()
        # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))
        head = ""

        # for i in range(ue_seq_len):
        #     head = head + "_b" + str(bs_num_seq[i]) + "_f" + str(ue_num_seq[i]) + "_u" + str(units_seq[i]) + "_a" + str(
        #         antenna_seq[i]) + str(r_min_seq[i])
        #     if (mec_rule_seq[i] == "max"):
        #         head = head + "_a"
        #     elif (mec_rule_seq[i] == "min"):
        #         head = head + "_i"
        #     else:
        #         head = head + "_f"

        # print('avg_bs_power' + head + '.png')
        plt.savefig('.\仿真图\\' + 'avg_bs_power1' + '.png',bbox_inches='tight')
        plt.savefig('.\仿真图\\' + 'avg_bs_power1' +'.pdf',bbox_inches='tight')
        plt.close()
'''
plot_simulation_results_bs_power_rule_Versus_RISunits
'''
def plot_simulation_results_bs_power_rule_Versus_RISunits(ue_num_seq, bs_num_seq, units_seq, antenna_seq, r_min_seq, bw_seq,mec_rule_seq, x_labelName, y_labelName, disp_elements):
    ue_seq_len = len(ue_num_seq)
    bs_seq_len = len(bs_num_seq)
    irs_units_seq_len = len(units_seq)

    # mec_power_seq_len = len(mec_power_seq)

    addition = ""
    addition_label = ""
    label_lst = []
    end_x = 0
    # label_lst = ['Ideal IRS','Non-ideal IRS with continuous phase shifts','Non-ideal IRS with discrete phase shifts'
    #     ,'No IRS']
    if (ue_seq_len != bs_seq_len  and ue_seq_len != irs_units_seq_len):
        print("输入的序列长度不对齐")
        return
    else:
        model_path_seq = []
        result_npy_seq = []
        records = []
        avg_power_seq = []
        avg_power_NoRIS_seq = []
        long_term_avg_power_RIS_seq = []
        long_term_avg_power_NoRIS_seq = []
        diff_seq = []

        for i in range(ue_seq_len):
            rs_collection = []
            label_collection = []
            rs_model_collection = []
            addition = ""
            addition_label = ""
            label = ""
            if (disp_elements[1] == True):
                label = label + "F=" + str(ue_num_seq[i]) + " "
            if (disp_elements[3] == True):
                label = label + "K=" + str(antenna_seq[i]) + " "
            if (disp_elements[4] == True):
                label = label + "R_min=" + str(r_min_seq[i]) + " "
            label_collection.append(label + addition_label)
            if (mec_rule_seq[i] == "max"):
                addition = "_mecRuleMax"
                addition_label = "Max-AO"
                # base = "baseline "
            elif (mec_rule_seq[i] == "min"):
                addition = "_mecRuleMin"
                addition_label = "Min-AO"
                # base = "baseline "
            else:
                addition_label = "DDQN-AO"
            for j in range(irs_units_seq_len):
                head = '_ue%d_bs%d_u%d_a%d_r%d%s' % (ue_num_seq[i], bs_num_seq[i], units_seq[j], antenna_seq[i], r_min_seq[i],  addition)
                rs_collection.append("..\DRL\simulation_result\\full\\bs_power_record" + head + ".npy")


            result_npy_seq.append(rs_collection)
            label_lst.append(label_collection)
        indices = []
        for i in range(ue_seq_len):
            for j in range(irs_units_seq_len):
                collections = []
                data = np.load(result_npy_seq[i][j])
                for index in range(data.shape[0]):
                    if (data[index] > 0):
                        collections.append(index)
                collections = np.array(collections)
                indices.append(collections)
        indices = reduce(np.intersect1d, indices)
        # indices = np.tile(indices, [10])
        # common_indices = indices[0]
        # for i in range(1,ue_seq_len):
        #     common_indices = np.intersect1d(common_indices,indices[i])
        plt.figure()
        # plt.ylim([13, 55])
        # ax = brokenaxes(xlims=[(0, 205)], ylims=((10, 27.5), (34, 55)), despine=False,
        #                 hspace=0.05, d=0.005)  # (6.27,6.30),(12.08,12.09),(25.225,25.24)
        plt.legend(loc=0, edgecolor='#000000', prop={'family': 'Times New Roman', 'size': 11})
        # plt.xlim(10000,100000)
        # plt.ylim(20,100)
        plt.grid(linestyle='-')
        plt.xlabel(x_labelName, fontdict={'family': 'Times New Roman', 'size': 12})
        plt.ylabel(y_labelName, fontdict={'family': 'Times New Roman', 'size': 12})
        # plt.xlim(0,205)
        # ax.set_xlabel(x_labelName, labelpad=35, fontdict={'family': 'Times New Roman', 'size': 12})
        # ax.set_ylabel(y_labelName, labelpad=35, fontdict={'family': 'Times New Roman', 'size': 12})
        # ax = plt.gca()
        # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))
        for i in range(ue_seq_len):
            long_term_avg_power_RIS_seq = []
            for j in range(irs_units_seq_len):
                # label_lst.append("bs_num=" + str(bs_num_seq[i]) + ",fov_num=" + str(ue_num_seq[i])+"units_num="+str(irs_units_seq[i])+"with RIS")
                avg_power = 10*np.log10(3*np.load(result_npy_seq[i][j])[indices]/0.001)#转为DBM
                avg_power = np.mean(avg_power)
                long_term_avg_power_RIS_seq.append(avg_power)
                avg_power_seq.append(avg_power)

            y = np.array(long_term_avg_power_RIS_seq)
            plt.plot(units_seq, y, linewidth=1.5, color=color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                     marker=marker_lst[j], markersize=5, label=label_lst[i])


        plt.savefig('.\仿真图\\' + 'tp_ris.png',bbox_inches='tight')
        plt.savefig('.\仿真图\\' + 'tp_ris.png',bbox_inches='tight')
        plt.close()
'''
plot_simulation_results_bs_power_rule_Versus_antenna
'''
def plot_simulation_results_bs_power_rule_Versus_antenna(ue_num_seq, bs_num_seq, units_seq, antenna_seq, r_min_seq, bw_seq,mec_rule_seq, x_labelName, y_labelName, disp_elements):
    ue_seq_len = len(ue_num_seq)
    bs_seq_len = len(bs_num_seq)
    irs_units_seq_len = len(units_seq)
    antenna_seq_len  = len(antenna_seq)

    # mec_power_seq_len = len(mec_power_seq)

    addition = ""
    addition_label = ""
    label_lst = []
    end_x = 0
    # label_lst = ['Ideal IRS','Non-ideal IRS with continuous phase shifts','Non-ideal IRS with discrete phase shifts'
    #     ,'No IRS']
    if (ue_seq_len != bs_seq_len  and ue_seq_len != irs_units_seq_len):
        print("输入的序列长度不对齐")
        return
    else:
        model_path_seq = []
        result_npy_seq = []
        records = []
        avg_power_seq = []
        avg_power_NoRIS_seq = []
        long_term_avg_power_RIS_seq = []
        long_term_avg_power_NoRIS_seq = []
        diff_seq = []

        for i in range(ue_seq_len):
            rs_collection = []
            label_collection = []
            rs_model_collection = []
            addition = ""
            addition_label = ""
            label = ""
            if (disp_elements[1] == True):
                label = label + "F=" + str(ue_num_seq[i]) + " "
            if (disp_elements[3] == True):
                label = label + "K=" + str(antenna_seq[i]) + " "
            if (disp_elements[4] == True):
                label = label + "R_min=" + str(r_min_seq[i]) + " "
            label_collection.append(label + addition_label)
            if (mec_rule_seq[i] == "max"):
                addition = "_mecRuleMax"
                addition_label = "Max-AO"
                # base = "baseline "
            elif (mec_rule_seq[i] == "min"):
                addition = "_mecRuleMin"
                addition_label = "Min-AO"
                # base = "baseline "
            else:
                addition_label = "DDQN-AO"
            for j in range(antenna_seq_len):
                head = '_ue%d_bs%d_u%d_a%d_r%d%s' % (ue_num_seq[i], bs_num_seq[i], units_seq[i], antenna_seq[j], r_min_seq[i],addition)
                rs_collection.append("..\DRL\simulation_result\\full\\bs_power_record" + head + ".npy")


            result_npy_seq.append(rs_collection)
            label_lst.append(label_collection)
        indices = []
        for i in range(ue_seq_len):
            for j in range(antenna_seq_len):
                collections = []
                data = np.load(result_npy_seq[i][j])
                for index in range(data.shape[0]):
                    if (data[index] > 0):
                        collections.append(index)
                collections = np.array(collections)
                indices.append(collections)
        indices = reduce(np.intersect1d, indices)
        # indices = np.tile(indices, [10])
        # common_indices = indices[0]
        # for i in range(1,ue_seq_len):
        #     common_indices = np.intersect1d(common_indices,indices[i])
        plt.figure()
        # plt.ylim([13, 55])
        # ax = brokenaxes(xlims=[(0, 205)], ylims=((10, 27.5), (34, 55)), despine=False,
        #                 hspace=0.05, d=0.005)  # (6.27,6.30),(12.08,12.09),(25.225,25.24)
        plt.legend(loc=0, edgecolor='#000000', prop={'family': 'Times New Roman', 'size': 11})
        # plt.xlim(10000,100000)
        # plt.ylim(20,100)
        plt.grid(linestyle='-')
        plt.xlabel(x_labelName, fontdict={'family': 'Times New Roman', 'size': 12})
        plt.ylabel(y_labelName, fontdict={'family': 'Times New Roman', 'size': 12})
        # plt.xlim(0,205)
        # ax.set_xlabel(x_labelName, labelpad=35, fontdict={'family': 'Times New Roman', 'size': 12})
        # ax.set_ylabel(y_labelName, labelpad=35, fontdict={'family': 'Times New Roman', 'size': 12})
        # ax = plt.gca()
        # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))
        for i in range(ue_seq_len):
            long_term_avg_power_RIS_seq = []
            for j in range(antenna_seq_len):
                # label_lst.append("bs_num=" + str(bs_num_seq[i]) + ",fov_num=" + str(ue_num_seq[i])+"units_num="+str(irs_units_seq[i])+"with RIS")
                avg_power = 10*np.log10(3*np.load(result_npy_seq[i][j])[indices]/0.001)#转为DBM
                avg_power = np.mean(avg_power)
                long_term_avg_power_RIS_seq.append(avg_power)
                avg_power_seq.append(avg_power)

            y = np.array(long_term_avg_power_RIS_seq)
            plt.plot(units_seq, y, linewidth=1.5, color=color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                     marker=marker_lst[j], markersize=5, label=label_lst[i])


        plt.savefig('.\仿真图\\' + 'tp_ris.png',bbox_inches='tight')
        plt.savefig('.\仿真图\\' + 'tp_ris.png',bbox_inches='tight')
        plt.close()

'''
plot_simulation_results_AO_algorithm
'''
def plot_simulation_results_AO_algorithm(x_labelName,y_labelName,label_lst,disp_gap=1):
        path = 'AO.mat'
        y = np.array(scio.loadmat(path)['data'])
        plt.figure()

        fig, ax = plt.subplots(1, 1)
        # ax.plot(x, y1, color='#f0bc94', label='trick-1', alpha=0.7)
        # ax.plot(x, y2, color='#7fe2b3', label='trick-2', alpha=0.7)
        # ax.plot(x, y3, color='#cba0e6', label='trick-3', alpha=0.7)
        # ax.legend(loc='right')


        # ax = brokenaxes(xlims=[(0, 20.5)],ylims=((6.27,6.30),(12.08,12.09),(25.225,25.24)), despine=False, hspace=0.15, d=0.005)#(6.27,6.30),(12.08,12.09),(25.225,25.24)
        # plt.xlim(0, disp_amount - start + 1)

        for j in range(len(label_lst)):
            y_slice= y[j,:]
            x_slice = np.arange(0, len(y_slice), 1)
            ax.plot(x_slice, y_slice, linewidth=1, color=color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                     marker=marker_lst[j], markersize=5, label=label_lst[j])


        ax.set_xlim(0,51)
        # plt.xlim(10000,100000)
        # plt.ylim(20,100)
        # ax.xlim(0,21)
        axins = ax.inset_axes((0.55, 0.3, 0.4, 0.3))
        ys = []
        for j in range(0,2):
            y_slice = y[j, :]
            x_slice = np.arange(0, len(y_slice), 1)
            ys.append(y_slice)
            axins.plot(x_slice, y_slice, linewidth=1, color=color_lst[j], markerfacecolor=markerfacecolor_lst[j],
                    marker=marker_lst[j], markersize=5, label=label_lst[j])

        zone_and_linked(ax, axins, 3, 20, x_slice, ys, 'right')

        ax.grid(linestyle='-')
        # xticksig = np.arange(0, 21, 1)
        # plt.xticks(xticksig)
        plt.xticks(fontproperties="Times New Roman", size=15)
        plt.yticks(fontproperties="Times New Roman", size=15)
        ax.set_xlabel(x_labelName,fontdict={'family': 'Times New Roman', 'size': 15})
        ax.set_ylabel(y_labelName,fontdict={'family': 'Times New Roman', 'size': 15},labelpad=0)
        ax.legend(loc=6, edgecolor='#000000', prop={'family': 'Times New Roman', 'size': 12})
        # plt.xlabel(x_labelName, fontdict={'family': 'Times New Roman', 'size': 12})
        # plt.ylabel(y_labelName, fontdict={'family': 'Times New Roman', 'size': 12})
        # ax.ticklabel_format(fontsize=5, axis='y')

        # ax.axs[0].get_yaxis().get_offset_text().set(size=30)
        # ax.axs[1].get_yaxis().get_offset_text().set_visible(False)  # 蔽掉下半图的指数（offset text）

        grid_margin = MultipleLocator(5)
        # for axs in ax.axes:
        ax.xaxis.set_major_locator(grid_margin)

        # ax = plt.gca()
        # ax.set_xticks(np.arange(0,len(end_x),int(len(end_x)/10)))
        plt.savefig('.\仿真图\\' + 'AO_iteration.png', bbox_inches='tight')
        plt.savefig('.\仿真图\\' + 'AO_iteration.pdf', bbox_inches='tight')
        # plt.show()
        plt.close()

def simulation(total_episode,detect_times,ue_num,bs_num,unit_num,antenna_num,target_r_min,mec_p_max,transmit_p_max,bw=100,r_min=5,test_mode=False,open_matlab=False,prioritized=False,improved_strategy=True,concern_all=False,double_q=True,mec_rule="default",test_step=60,load_H_path= None):
    # 用来控制什么时候学习
    step = 1
    result = []
    addition=""
    if(mec_rule=="max"):
        addition="_mecRuleMax"
    elif(mec_rule=="min"):
        addition = "_mecRuleMin"
    elif(mec_rule=="exhaustion"):
        addition = 'mecRuleExhaustion'
    # model_path = os.path.join('..\DRL', 'simulation_result', 'full',
    #                           'model_ue%d_bs%d_u%d_a%d_r%d.ckpt' % (ue_num, bs_num, unit_num,antenna_num, r_min))
    head2  =''
    if(improved_strategy):
        head = '_ue%d_bs%d_u%d_a%d_r%d%s_improved' % (ue_num, bs_num, unit_num, antenna_num, target_r_min, addition)
        head2 = '_ue%d_bs%d_u%d_a%d_r%d_improved' % (ue_num, bs_num, unit_num, antenna_num, 30)
    else:
        head = '_ue%d_bs%d_u%d_a%d_r%d%s_base' % (ue_num, bs_num, unit_num, antenna_num, target_r_min,addition)
        head2 = '_ue%d_bs%d_u%d_a%d_r%d_base' % (ue_num, bs_num, unit_num, antenna_num, 30)
    if(concern_all==True):
        head = head +'_concern_all'
        head2 = head2 +'_concern_all'
    else:
        head = head +'_concern_part'
        head2 = head2 + '_concern_part'
    if(double_q):
        head = head +"_DDQN"
        head2 = head2 + '_DDQN'
    else:
        head = head +'DQN'
        head2 = head2 + 'DQN'


    model_path= os.path.join('..\DRL', 'simulation_result', 'full',
                              'model%s.ckpt'%head2)
    train_mode = True
    if(test_mode==True):
        train_mode=False
        scene = IrsCompMISOEnv(bs_num=bs_num, ue_num=ue_num, transmit_p_max=transmit_p_max, mec_p_max=mec_p_max,
                               irs_units_num=unit_num,
                               antenna_num=antenna_num, fov_patch_num=ue_num, reflect_max=2 * np.pi,
                               r_min=target_r_min, load_H_path=load_H_path,
                               BW=bw, open_matlab=open_matlab, train=train_mode, mec_rule=mec_rule)
    else:
        scene = IrsCompMISOEnv(bs_num=bs_num, ue_num=ue_num, transmit_p_max=transmit_p_max,mec_p_max=mec_p_max,
                               irs_units_num=unit_num,
                               antenna_num=antenna_num, fov_patch_num=ue_num, reflect_max=2 * np.pi,
                               r_min=r_min,load_H_path=load_H_path,
                               BW=bw,open_matlab=open_matlab,train=train_mode,mec_rule=mec_rule)
    action_nums= []
    for table in scene.action_table:
        action_nums.append(len(table))
    if(test_mode==False):
            fov_agents = DqnAgent(n_actions=action_nums, n_features=len(scene.states),
                                  agent_num=int(scene.fov_patch_num),
                                  learning_rate=FLAGS.dqn_lrc, reward_decay=0.9, e_greedy=0.9,
                                  batch_size=FLAGS.batch_size, replace_target_iter=FLAGS.replace_target_iter,
                                  # 每 200 步替换一次 target_net 的参数
                                  memory_size=FLAGS.memory_size,  # 记忆上限
                                  e_greedy_increment=FLAGS.e_greedy_increment_c, output_graph=False,
                                  # 是否输出 tensorboard 文件
                                  name_str='dqn', double_q=double_q,
                                  prioritized_replay=prioritized)
        # try:
        #     fov_agents.load_model(model_path)
        # except:
        #     print("无已有存档")
        # fov_agents.load_model(model_path)
    else:
        fov_agents = DqnAgent(n_actions=action_nums, n_features=len(scene.states), agent_num=int(scene.fov_patch_num),
                                 learning_rate=FLAGS.dqn_lrc, reward_decay=0.9, e_greedy=1,
                                 batch_size=FLAGS.batch_size, replace_target_iter=FLAGS.replace_target_iter,
                                 # 每 200 步替换一次 target_net 的参数
                                 memory_size=FLAGS.memory_size,  # 记忆上限
                                 e_greedy_increment=None, output_graph=False,  # 是否输出 tensorboard 文件
                                 name_str='dqn', double_q=double_q,
                                 prioritized_replay=prioritized)
        fov_agents.load_model(model_path)

        # print('loading model')
        # fov_agents.load_model("bs")
        # print('model ' + 'loaded')

        print("%s加载成功"%model_path)
    # ax = []
    # ay = []
    # plt.ion()

    # if (test_mode == True):
    #     total_episode=999
    # loader = tqdm(range(total_episode))
    tqdm_loader =tqdm(range(total_episode))
    best_avg_r = -999
    best_ep = -999
    for episode in tqdm_loader:
            character = episode%ue_num
            observation = scene.reset()
            T_old = 0
            T_fov_deployment_num = 0

            for detect in range(detect_times):
                action_c_reflect = []
                if (improved_strategy):
                    chosen_actions = fov_agents.choose_action(character, observation, scene.fov_patch_num,
                                                              scene.epsilon)
                else:
                    chosen_actions = fov_agents.choose_action_base(observation, scene.fov_patch_num,
                                                                   scene.epsilon)
                chosen_action_real = omega_real_agent.choose_action(0, observation, scene.fov_patch_num,
                                                                    scene.epsilon)
                chosen_action_imag = omega_imag_agent.choose_action(0, observation, scene.fov_patch_num,
                                                                    scene.epsilon)
                for i in range(scene.irs_units_num):
                    action_c_reflect_i = 1
                    if action_c_reflect_i == 1:
                        action_c_reflect_i = np.pi
                    action_c_reflect.append(action_c_reflect_i)

                scene.reflect = reflect_calculate(action_c_reflect, np.ones((scene.irs_units_num)),
                                                  scene.irs_units_num)

                # print("chosen action",chosen_actions)
                current_reward, observation_, epsilon = scene.step(chosen_actions, chosen_action_real,
                                                                   chosen_action_imag, detect)

                # savePathG = r'.\dataG.mat'
                # savePathE = r'.\epsilon.mat'
                # reward = current_reward-last_reward
                # # G= np.array(G).reshape([scene.cue,scene.ch_k]).tolist()
                # last_reward=current_reward
                # scio.savemat(savePathG, {'G': scene.G2.tolist()})
                # scio.savemat(savePathE, {'Epsilon': epsilon.tolist()})

                T_old += current_reward
                if (test_mode == True and open_matlab == True):
                    print("bs:%d,ue:%d,antenna:%d,unit:%d,mec_rule:%s step:[%d|%d]" % (
                        bs_num, ue_num, antenna_num, unit_num, mec_rule, len(scene.total_power_record), test_step))
                    if (len(scene.total_power_record) >= test_step):
                        np.save('.\simulation_result\\full\\ue_avg_rates_record' + head,
                                np.array(scene.ue_avg_rates_record))
                        np.save('.\simulation_result\\full\\ue_avg_rates_record_NoRIS' + head,
                                np.array(scene.ue_avg_rates_record_NoRIS))
                        np.save('.\simulation_result\\full\\ue_avg_rates_record_noCoMP' + head,
                                np.array(scene.ue_avg_rates_record_noCoMP))
                        np.save('.\simulation_result\\full\\ue_avg_rates_record_noCoMP_NoRIS' + head,
                                np.array(scene.ue_avg_rates_record_noCoMP_NoRIS))
                        np.save('.\simulation_result\\full\\total_power_record' + head,
                                np.array(scene.total_power_record))
                        np.save('.\simulation_result\\full\\total_power_record' + head,
                                np.array(scene.total_power_record))
                        np.save('.\simulation_result\\full\\total_power_record' + head,
                                np.array(scene.total_power_record))
                        np.save('.\simulation_result\\full\\bs_power_record' + head,
                                np.array(scene.bs_power_record))
                        np.save('.\simulation_result\\full\\total_power_record_NoRIS' + head,
                                np.array(scene.total_power_record_NoRIS))
                        np.save('.\simulation_result\\full\\bs_power_record_NoRIS' + head,
                                np.array(scene.bs_power_record_NoRIS))
                        np.save('.\simulation_result\\full\\total_init_power_record' + head,
                                np.array(scene.total_init_power_record))
                        np.save('.\simulation_result\\full\\init_bs_power_record' + head,
                                np.array(scene.init_bs_power_record))
                        # np.save('.\simulation_result\\full\\opt_G_record' + head, np.array(scene.opt_G_record))
                        print("记录结束")
                        return

                # print('第%depisode的第%d次结果为：%f' % (episode,detect, current_reward))
                else:
                    for i in range(scene.ue_num):
                        fov_agents.store_transition(i, observation, chosen_actions[i], current_reward,
                                                    observation_)

                    sample_index = []
                    if (improved_strategy):
                        if step > FLAGS.batch_size and step % FLAGS.training_interval == 0:
                            # sample_index = ddpg_p.learn(sample_index)
                            # for i in range(scene.bs_num):
                            sample_index = fov_agents.learn(character, sample_index)
                            # var_ramp *= 0.9996
                            # var_p *= 0.9996
                    else:
                        if step > FLAGS.batch_size and step % FLAGS.training_interval == 0:
                            for i in range(scene.bs_num):
                                sample_index = fov_agents.learn(i, sample_index)
                observation = observation_
                step += 1  # 总步数
            avg_r = T_old / detect_times
            if (avg_r >= best_avg_r):
                best_avg_r = avg_r
                if (episode > 200):
                    fov_agents.save_model(head)
            info = 'Cha:%d BS:%d, UE:%d, Antenna:%d,Unit:%d, p:%f,Epoch[%d|%d] r:%f best_r:%f' % (
                character, bs_num, ue_num, antenna_num, unit_num, fov_agents.epsilons[character], episode,
                total_episode, avg_r, best_avg_r)
            tqdm_loader.set_description_str(info)
            # ax.append(episode)
            # ay.append(avg_r)
            # plt.clf()
            # plt.plot(ax, ay)
            # plt.pause(0.1)
            if (concern_all == False):
                result.append(np.maximum(T_old / detect_times - 25, -50))
            else:
                result.append(T_old / detect_times)




    # np.save("available_action_record.npy",scene.available_action)

    np.save('.\simulation_result\\full\\reward_record'+head+'.npy',np.array(result))
    # fov_agents.save_model(head)
    # disp_result(result)


def simulation_stage2(total_episode,detect_times,ue_num,bs_num,unit_num,antenna_num,target_r_min,mec_p_max,transmit_p_max,bw=100,r_min=5,open_matlab=False,prioritized=False,improved_strategy=True,concern_all=False,double_q=True,mec_rule="default",load_H_path= None):
    # 用来控制什么时候学习
    step = 1
    result = []
    addition=""
    if(mec_rule=="max"):
        addition="_mecRuleMax"
    elif(mec_rule=="min"):
        addition = "_mecRuleMin"
    # model_path = os.path.join('..\DRL', 'simulation_result', 'full',
    #                           'model_ue%d_bs%d_u%d_a%d_r%d.ckpt' % (ue_num, bs_num, unit_num,antenna_num, r_min, bw))
    if(improved_strategy):
        head = '_ue%d_bs%d_u%d_a%d_r%d%s_improved' % (ue_num, bs_num, unit_num, antenna_num, target_r_min, addition)
    else:
        head = '_ue_ue%d_bs%d_u%d_a%d_r%d%s_base' % (ue_num, bs_num, unit_num, antenna_num, target_r_min,  addition)
    if(concern_all==True):
        head = head +'_concern_all'
    else:
        head = head +'_concern_part'
    if(double_q):
        head = head +"_DDQN"
    else:
        head = head +'DQN'
    model_path= os.path.join('..\DRL', 'simulation_result', 'full',
                              'model%s.ckpt'%head)



    scene = IrsCompMISOEnv(bs_num=bs_num, ue_num=ue_num, transmit_p_max=transmit_p_max,mec_p_max=mec_p_max,
                               irs_units_num=unit_num,
                               antenna_num=antenna_num, fov_patch_num=ue_num, reflect_max=2 * np.pi,
                               r_min=target_r_min,load_H_path=load_H_path,
                               BW=bw,open_matlab=open_matlab,train=True,continue_cvx=True,mec_rule=mec_rule)
    action_nums= []
    for table in scene.action_table:
        action_nums.append(len(table))
    fov_agents = DqnAgent(n_actions=action_nums, n_features=len(scene.states), agent_num=int(scene.fov_patch_num),
                          learning_rate=FLAGS.dqn_lrc, reward_decay=0.9, e_greedy=0.99,
                          batch_size=FLAGS.batch_size, replace_target_iter=FLAGS.replace_target_iter,
                          # 每 200 步替换一次 target_net 的参数
                          memory_size=FLAGS.memory_size,  # 记忆上限
                          e_greedy_increment=None, output_graph=False,  # 是否输出 tensorboard 文件
                          name_str='dqn', double_q=double_q,
                          prioritized_replay=prioritized)
    fov_agents.load_model(model_path)
        # print('loading model')
        # fov_agents.load_model("bs")
        # print('model ' + 'loaded')
    print("加载成功")
    # ax = []
    # ay = []
    # plt.ion()

    # if (test_mode == True):
    #     total_episode=999
    # loader = tqdm(range(total_episode))
    tqdm_loader =tqdm(range(total_episode))
    best_avg_r = -999
    for episode in tqdm_loader:
            character = episode%ue_num
            observation = scene.reset()
            T_old = 0
            for detect in range(detect_times):
                action_c_reflect = []
                if (improved_strategy):
                    chosen_actions = fov_agents.choose_action(character,observation,scene.fov_patch_num,scene.epsilon)
                else:
                    chosen_actions = fov_agents.choose_action_base( observation, scene.fov_patch_num,
                                                              scene.epsilon)

                for i in range(scene.irs_units_num):
                    action_c_reflect_i = 1
                    if action_c_reflect_i == 1:
                        action_c_reflect_i = np.pi
                    action_c_reflect.append(action_c_reflect_i)

                scene.reflect = reflect_calculate(action_c_reflect, np.ones((scene.irs_units_num)), scene.irs_units_num)

                # print("chosen action",chosen_actions)
                current_reward, observation_,epsilon = scene.step(chosen_actions,detect)


                # savePathG = r'.\dataG.mat'
                # savePathE = r'.\epsilon.mat'
                # reward = current_reward-last_reward
                # # G= np.array(G).reshape([scene.cue,scene.ch_k]).tolist()
                # last_reward=current_reward
                # scio.savemat(savePathG, {'G': scene.G2.tolist()})
                # scio.savemat(savePathE, {'Epsilon': epsilon.tolist()})



                T_old += current_reward
                # print('第%depisode的第%d次结果为：%f' % (episode,detect, current_reward))
                for i in range(scene.ue_num):
                    fov_agents.store_transition(i, observation, chosen_actions[i], current_reward,
                                                observation_)

                sample_index = []
                if (improved_strategy):
                    if step > FLAGS.batch_size and step % FLAGS.training_interval == 0:
                        # sample_index = ddpg_p.learn(sample_index)
                        # for i in range(scene.bs_num):
                        sample_index = fov_agents.learn(character, sample_index)
                        # var_ramp *= 0.9996
                        # var_p *= 0.9996
                else:
                    if step > FLAGS.batch_size and step % FLAGS.training_interval == 0:
                        for i in range(scene.bs_num):
                            sample_index = fov_agents.learn(i, sample_index)

                observation = observation_
                step += 1  # 总步数
            avg_r=T_old/detect_times
            if(avg_r>=best_avg_r):
                best_avg_r = avg_r
                if(episode>0):
                   fov_agents.save_model(head)
            info ='Cha:%d BS:%d, UE:%d, Antenna:%d,Unit:%d, p:%f,Epoch[%d|%d] r:%f best_r:%f' % (character,bs_num,ue_num,antenna_num,unit_num,fov_agents.epsilons[character],episode,total_episode, avg_r,best_avg_r)
            tqdm_loader.set_description_str(info)
            # ax.append(episode)
            # ay.append(avg_r)
            # plt.clf()
            # plt.plot(ax, ay)
            # plt.pause(0.1)

            result.append(T_old / detect_times)





    # np.save("available_action_record.npy",scene.available_action)

    np.save('.\simulation_result\\full\\reward_record_model'+head+'.npy',np.array(result))
    # fov_agents.save_model(head)
    # disp_result(result)





def pre(ue_num,bs_num,unit_num,antenna_num,mec_p_max,transmit_p_max,bw=100,r_min=5,open_matlab=True,mec_rule="default",load_H_path= None):
    # 用来控制什么时候学习
    scene = IrsCompMISOEnv(bs_num=bs_num, ue_num=ue_num, transmit_p_max=transmit_p_max,mec_p_max=mec_p_max,
                               irs_units_num=unit_num,
                               antenna_num=antenna_num, fov_patch_num=ue_num, reflect_max=2 * np.pi,
                               r_min=r_min,load_H_path=load_H_path,
                               BW=bw,open_matlab=open_matlab,train=True,mec_rule=mec_rule)
    # action_nums= []
    # for table in scene.action_table:
    #     action_nums.append(len(table))
    action_space = len(scene.action_table[0])
    tqdm_loader =tqdm(range(action_space))
    # print("chosen action",chosen_actions)
    cnt = 0
    total_cnt = 0
    filename = '%d%d%d'%(ue_num,bs_num,unit_num) +'.npy'
    cur_dict  =dict()
    if not (os.path.exists(filename)):
        np.save(filename , cur_dict)
    for a1 in tqdm_loader:
        actions = np.zeros([ue_num],dtype=np.int)
        actions[0]=a1
        for a2 in range(action_space):
            actions[1] = a2
            for a3 in range(action_space):
                actions[2] = a3
                for a4 in range(action_space):
                    actions[3] = a4
                    for a5 in range(action_space):
                        actions[4] = a5
                        for a6 in range(action_space):
                            actions[5] = a6
                            for a7 in range(action_space):
                                actions[6] = a7
                                for a8 in range(action_space):

                                    actions[7] = a8
                                    current_reward = scene.cal_reward_validate(actions, 0)
                                    key = str(ue_num) + str(bs_num) + str(unit_num) + str(actions)

                                    if(current_reward<0):
                                        cur_dict[key]=50
                                    else:
                                        cur_dict[key]=-50
    print(cnt)
    save_dict_by_numpy('%d%d%d'%(ue_num,bs_num,unit_num) +'.npy', cur_dict)
    # savePathG = r'.\dataG.mat'
    # savePathE = r'.\epsilon.mat'step
    # reward = current_reward-last_reward
    # # G= np.array(G).reshape([scene.cue,scene.ch_k]).tolist()
    # last_reward=current_reward
    # scio.savemat(savePathG, {'G': scene.G2.tolist()})
    # scio.savemat(savePathE, {'Epsilon': epsilon.tolist()})
    # info ='Cha:%d BS:%d, UE:%d, Antenna:%d,Unit:%d, p:%f,Epoch[%d|%d] r:%f best_r:%f' % (character,bs_num,ue_num,antenna_num,unit_num,fov_agents.epsilons[character],episode,total_episode, avg_r,best_avg_r)
    # tqdm_loader.set_description_str(info)

    # np.save("available_action_record.npy",scene.available_action)

    # fov_agents.save_model(head)
    # disp_result(result)



def simulation_only_MEC(total_episode,detect_times,ue_num,bs_num,unit_num,antenna_num,r_min,p_max,bw,test_mode=False,test_step=60):
    # 用来控制什么时候学习
    step = 1
    result = []
    model_path = os.path.join('..\DRL', 'simulation_result', 'full', 'model_ue%d_bs%d_u%d_a%d_r%d.ckpt'%(ue_num,bs_num,unit_num,r_min))
    head = '_ue%d_bs%d_u%d_a%d_r%d'%(ue_num,bs_num,unit_num,antenna_num,r_min)


    scene = IrsCompMISOEnv(bs_num=bs_num, ue_num=ue_num, p_max=p_max,
                           irs_units_num=unit_num,
                           antenna_num=antenna_num, fov_patch_num=ue_num, reflect_max=2 * np.pi,
                           r_min=r_min,
                           BW=bw,open_matlab=False,action_main="fov",train_mode=False)
    action_nums= []
    for table in scene.action_table:
        action_nums.append(len(table))
    if(test_mode==False):
        fov_agents = DqnAgent(n_actions=action_nums, n_features=len(scene.states),agent_num=int(scene.fov_patch_num),
                            learning_rate=FLAGS.dqn_lrc, reward_decay=0.9, e_greedy=FLAGS.e_greedy,
                            batch_size=FLAGS.batch_size, replace_target_iter=FLAGS.replace_target_iter,
                            # 每 200 步替换一次 target_net 的参数
                            memory_size=FLAGS.memory_size,  # 记忆上限
                            e_greedy_increment=FLAGS.e_greedy_increment_c, output_graph=False,  # 是否输出 tensorboard 文件
                            name_str= 'dqn', double_q=True,
                            prioritized_replay=FLAGS.prioritized_r)
    else:


        fov_agents = DqnAgent(n_actions=action_nums, n_features=len(scene.states), agent_num=int(scene.fov_patch_num),
                             learning_rate=FLAGS.dqn_lrc, reward_decay=0.9, e_greedy=1,
                             batch_size=FLAGS.batch_size, replace_target_iter=FLAGS.replace_target_iter,
                             # 每 200 步替换一次 target_net 的参数
                             memory_size=FLAGS.memory_size,  # 记忆上限
                             e_greedy_increment=None, output_graph=False,  # 是否输出 tensorboard 文件
                             name_str='dqn', double_q=FLAGS.double_q,
                             prioritized_replay=FLAGS.prioritized_r)



        fov_agents.load_model(model_path)
        print("加载成功")
    ax = []
    ay = []
    plt.ion()

    if (test_mode == True):
        total_episode=999
    for episode in range(total_episode):
        observation = scene.reset()
        T_old = 0
        for detect in range(detect_times):
            action_c_reflect = []

            chosen_actions = fov_agents.choose_action(observation,scene.fov_patch_num,scene.epsilon)

            for i in range(scene.irs_units_num):
                action_c_reflect_i = 1
                if action_c_reflect_i == 1:
                    action_c_reflect_i = np.pi
                action_c_reflect.append(action_c_reflect_i)

            scene.reflect = reflect_calculate(action_c_reflect, np.ones((scene.irs_units_num)), scene.irs_units_num)

            print("chosen action",chosen_actions)
            current_reward, observation_,epsilon = scene.step(chosen_actions,detect)
            # savePathG = r'.\dataG.mat'
            # savePathE = r'.\epsilon.mat'
            # reward = current_reward-last_reward
            # # G= np.array(G).reshape([scene.cue,scene.ch_k]).tolist()
            # last_reward=current_reward
            # scio.savemat(savePathG, {'G': scene.G2.tolist()})
            # scio.savemat(savePathE, {'Epsilon': epsilon.tolist()})
            T_old += current_reward
            if(test_mode == True):
                print("记录进度", str(len(scene.total_power_record)), "|", str(test_step))
                if (len(scene.total_power_record)>=test_step):
                        np.save('.\simulation_result\\full\\mec_power_record'+head,np.array(scene.total_power_record))

                        print("记录结束")
                        return

            print('第%depisode的第%d次结果为：%f' % (episode,detect, current_reward))
            if(test_mode==False):
                for i in range(scene.bs_num):
                    fov_agents.store_transition(observation, chosen_actions[i], current_reward, observation_)
                sample_index = []
                if step > FLAGS.batch_size and step % FLAGS.training_interval == 0:
                    # sample_index = ddpg_p.learn(sample_index)
                    for i in range(scene.bs_num):
                        sample_index = fov_agents.learn(i,sample_index)
                    # var_ramp *= 0.9996
                    # var_p *= 0.9996
            observation = observation_
            step += 1  # 总步数
        avg_r=T_old/detect_times
        print('第%d次结果为：%f' % (episode, avg_r))
        ax.append(episode)
        ay.append(avg_r)
        plt.clf()
        plt.plot(ax, ay)
        plt.pause(0.1)
        result.append(T_old / detect_times)



    # np.save("available_action_record.npy",scene.available_action)
    np.save('reward_record'+head+'.npy',result)
    fov_agents.save_model(head)
    disp_result(result)



def simulation_only_MEC(total_episode,detect_times,ue_num,bs_num,unit_num,antenna_num,r_min,p_max,bw,test_mode=False,test_step=60):
    # 用来控制什么时候学习
    step = 1
    result = []
    model_path = os.path.join('..\DRL', 'simulation_result', 'full', 'model_ue' + str(ue_num)+ '_bs' + str(
        bs_num) + '_u' + str(unit_num)+'_a'+str(antenna_num)+'.ckpt')
    head = '_ue' + str(ue_num)+ '_bs' + str(
        bs_num) + '_u' + str(unit_num)+'_a'+str(antenna_num)+'_r'+str(r_min)+'_bw'+str(bw)

    scene = IrsCompMISOEnv(bs_num=bs_num, ue_num=ue_num, p_max=p_max,
                           irs_units_num=unit_num,
                           antenna_num=antenna_num, fov_patch_num=ue_num, reflect_max=2 * np.pi,
                           r_min=r_min,
                           BW=bw,open_matlab=False,action_main="fov",train_mode=False)
    action_nums= []
    for table in scene.action_table:
        action_nums.append(len(table))
    if(test_mode==False):
        fov_agents = DqnAgent(n_actions=action_nums, n_features=len(scene.states),agent_num=int(scene.fov_patch_num),
                            learning_rate=FLAGS.dqn_lrc, reward_decay=0.9, e_greedy=FLAGS.e_greedy,
                            batch_size=FLAGS.batch_size, replace_target_iter=FLAGS.replace_target_iter,
                            # 每 200 步替换一次 target_net 的参数
                            memory_size=FLAGS.memory_size,  # 记忆上限
                            e_greedy_increment=FLAGS.e_greedy_increment_c, output_graph=False,  # 是否输出 tensorboard 文件
                            name_str= 'dqn', double_q=True,
                            prioritized_replay=FLAGS.prioritized_r)
    else:


        fov_agents = DqnAgent(n_actions=action_nums, n_features=len(scene.states), agent_num=int(scene.fov_patch_num),
                             learning_rate=FLAGS.dqn_lrc, reward_decay=0.9, e_greedy=1,
                             batch_size=FLAGS.batch_size, replace_target_iter=FLAGS.replace_target_iter,
                             # 每 200 步替换一次 target_net 的参数
                             memory_size=FLAGS.memory_size,  # 记忆上限
                             e_greedy_increment=None, output_graph=False,  # 是否输出 tensorboard 文件
                             name_str='dqn', double_q=FLAGS.double_q,
                             prioritized_replay=FLAGS.prioritized_r)



        fov_agents.load_model(model_path)
        print("加载成功")
    ax = []
    ay = []
    plt.ion()

    if (test_mode == True):
        total_episode=999
    for episode in range(total_episode):
        observation = scene.reset()
        T_old = 0
        for detect in range(detect_times):
            action_c_reflect = []

            chosen_actions = fov_agents.choose_action(observation,scene.fov_patch_num,scene.epsilon)

            for i in range(scene.irs_units_num):
                action_c_reflect_i = 1
                if action_c_reflect_i == 1:
                    action_c_reflect_i = np.pi
                action_c_reflect.append(action_c_reflect_i)

            scene.reflect = reflect_calculate(action_c_reflect, np.ones((scene.irs_units_num)), scene.irs_units_num)

            print("chosen action",chosen_actions)
            current_reward, observation_,epsilon = scene.step(chosen_actions,detect)
            # savePathG = r'.\dataG.mat'
            # savePathE = r'.\epsilon.mat'
            # reward = current_reward-last_reward
            # # G= np.array(G).reshape([scene.cue,scene.ch_k]).tolist()
            # last_reward=current_reward
            # scio.savemat(savePathG, {'G': scene.G2.tolist()})
            # scio.savemat(savePathE, {'Epsilon': epsilon.tolist()})
            T_old += current_reward
            if(test_mode == True):
                print("记录进度", str(len(scene.total_power_record)), "|", str(test_step))
                if (len(scene.total_power_record)>=test_step):
                        np.save('.\simulation_result\\full\\mec_power_record'+head,np.array(scene.total_power_record))

                        print("记录结束")
                        return

            print('第%depisode的第%d次结果为：%f' % (episode,detect, current_reward))
            if(test_mode==False):
                for i in range(scene.bs_num):
                    fov_agents.store_transition(observation, chosen_actions[i], current_reward, observation_)
                sample_index = []
                if step > FLAGS.batch_size and step % FLAGS.training_interval == 0:
                    # sample_index = ddpg_p.learn(sample_index)
                    for i in range(scene.bs_num):
                        sample_index = fov_agents.learn(i,sample_index)
                    # var_ramp *= 0.9996
                    # var_p *= 0.9996
            observation = observation_
            step += 1  # 总步数
        avg_r=T_old/detect_times
        print('第%d次结果为：%f' % (episode, avg_r))
        ax.append(episode)
        ay.append(avg_r)
        plt.clf()
        plt.plot(ax, ay)
        plt.pause(0.1)
        result.append(T_old / detect_times)



    # np.save("available_action_record.npy",scene.available_action)
    np.save('reward_record'+head+'.npy',result)
    fov_agents.save_model(head)
    disp_result(result)

def simulation_random_omega(ue_num,bs_num,unit_num,antenna_num,r_min,bw,test_step=60,detect_times=128):
    # 用来控制什么时候学习
    step = 1


    model_path = os.path.join('..\DRL', 'simulation_result', 'full',
                              'model_ue%d_bs%d_u%d_a%d_r%d_improved_concern_all_DDQN.ckpt' % (
                              ue_num, bs_num, unit_num, antenna_num, r_min))
    # head = '_ue' + str(ue_num)+ '_bs' + str(
    #     bs_num) + '_u' + str(unit_num)+'_a'+str(antenna_num)+'_r'+str(r_min)+'_bw'+str(bw)
    head = '_ue' + str(ue_num) + '_bs' + str(
        bs_num) + '_u' + str(unit_num) + '_a' + str(antenna_num)

    scene = IrsCompMISOEnv(bs_num=bs_num, ue_num=ue_num, transmit_p_max=1, mec_p_max=35,
                           irs_units_num=unit_num,
                           antenna_num=antenna_num, fov_patch_num=ue_num, reflect_max=2 * np.pi,
                           r_min=r_min,train=False,
                           BW=bw, open_matlab=False,
                           mec_rule="default",rand_omega=True,
                           )
    action_nums= []
    for table in scene.action_table:
        action_nums.append(len(table))

    fov_agents = DqnAgent(n_actions=action_nums, n_features=len(scene.states), agent_num=int(scene.fov_patch_num),
                          learning_rate=FLAGS.dqn_lrc, reward_decay=0.9, e_greedy=1,
                          batch_size=FLAGS.batch_size, replace_target_iter=FLAGS.replace_target_iter,
                          # 每 200 步替换一次 target_net 的参数
                          memory_size=FLAGS.memory_size,  # 记忆上限
                          e_greedy_increment=None, output_graph=False,  # 是否输出 tensorboard 文件
                          name_str='dqn', double_q=FLAGS.double_q,
                          prioritized_replay=True)
    fov_agents.load_model(model_path)
    # print('loading model')
    # fov_agents.load_model("bs")
    # print('model ' + 'loaded')
    print("加载成功")
    ax = []
    ay = []
    plt.ion()
    total_episode=999
    for episode in range(total_episode):
        character = episode % ue_num
        observation = scene.reset()
        T_old = 0
        for detect in range(detect_times):
            action_c_reflect = []
            chosen_actions = fov_agents.choose_action(character, observation, scene.fov_patch_num, scene.epsilon)
            current_reward, observation_, epsilon = scene.step(chosen_actions, detect)
            T_old += current_reward
            print("记录进度", str(len(scene.bs_power_randOmega_record)), "|", str(test_step))
            if (len(scene.bs_power_randOmega_record) >= test_step):
                    np.save('.\simulation_result\\full\\bs_power_randOmega_record' + head,
                            np.array(scene.bs_power_randOmega_record))
                    np.save('.\simulation_result\\full\\total_power_randOmega_record' + head,
                            np.array(scene.total_power_randOmega_record))
                    print("记录结束")
                    return
            observation = observation_
            step += 1  # 总步数


        # ax.append(episode)
        # ay.append(avg_r)
        # plt.clf()
        # plt.plot(ax, ay)
        # plt.pause(0.1)




    # for episode in range(total_episode):
    #     observation = scene.reset()
    #     T_old = 0
    #     for detect in range(detect_times):
    #         action_c_reflect = []
    #         chosen_actions = fov_agents.choose_action(observation, scene.fov_patch_num, scene.epsilon)
    #         for i in range(scene.irs_units_num):
    #             action_c_reflect_i = 1
    #             if action_c_reflect_i == 1:
    #                 action_c_reflect_i = np.pi
    #             action_c_reflect.append(action_c_reflect_i)
    #         scene.reflect = reflect_calculate(action_c_reflect, np.ones((scene.irs_units_num)), scene.irs_units_num)
    #         print("chosen action",chosen_actions)
    #         current_reward, observation_,epsilon = scene.step(chosen_actions,detect)
    #         T_old += current_reward
    #         progress = len(scene.bs_power_randOmega_record)
    #         print("记录进度", str(progress), "|", str(test_step))
    #         if (progress >= test_step):
    #             np.save('.\simulation_result\\full\\bs_power_randOmega_record' + head, np.array(scene.bs_power_randOmega_record))
    #             print("记录结束")
    #             return
    #
    #         observation = observation_
    #         step += 1  # 总步数

def disp_result(result):
    plt.figure()
    plt.plot(np.arange(len(result)), result)
    plt.ylabel('average reward(ms)')
    plt.xlabel('episode')
    plt.savefig(r'reward_record.png')
    plt.show()


'''
1、plt.legend

plt.legend(loc=0)#显示图例的位置，自适应方式

说明：

'best' : 0, (only implemented for axes legends)(自适应方式)

'upper right' : 1,

'upper left' : 2,

'lower left' : 3,

'lower right' : 4,

'right' : 5,

'center left' : 6,

'center right' : 7,

'lower center' : 8,

'upper center' : 9,

'center' : 10,
'''
from matplotlib.patches import  ConnectionPatch


def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)



def record_H(ue, bs, antenna, unit, sample_num):
    MISO = IrsCompMISOEnv(bs_num=bs, ue_num=ue, transmit_p_max=5, mec_p_max=35,
                           irs_units_num=unit,
                           antenna_num=antenna, fov_patch_num=ue, reflect_max=2 * np.pi,
                           r_min=5, load_H_path=None,
                           BW=100, open_matlab=False, train=True, mec_rule="default")

    epoch_H_record = []
    MISO.reset()
    G1_record = []
    G2_record = []
    G_bsris_record = []
    G_ueris_record = []
    G_bsue_record = []

    for i in tqdm(range(0, sample_num+1)):
        G_, G2_, g_ue_ris, g_bs_ris, g_bs_ue = MISO.step2(i)
        G_bsris_record.append(g_bs_ris)
        G_ueris_record.append(g_ue_ris)
        G_bsue_record.append(g_bs_ue)

    G_bsris_record = np.array(G_bsris_record)
    G_ueris_record = np.array(G_ueris_record)
    G_bsue_record = np.array(G_bsue_record)
    # G2 = np.zeros([sample_num+1,ue + 1, antenna + 1, unit + 1, bs, ue, antenna, 1],dtype=np.complex)
    # # G1 = np.zeros([sample_num,ue + 1, antenna + 1, unit + 1, bs * ue * antenna * 2],dtype=np.complex)
    # for i in tqdm(range(0, sample_num)):
    #     for n in [6,7,8]:
    #         for a in [8,9,10]:
    #             for u in [20,30,40]:
    #                 for bs in range(3):
    #                     for ue in range(n):
    #                         g_bs_ue = G_bsue_record[i,bs, ue, :a, :]
    #                         g_bs_ris =G_bsris_record[i,bs, :u, :a]
    #                         g_ue_ris =G_ueris_record[i,bs, ue, :u, :]
    #                         g =G_gain_cal(g_bs_ue, g_bs_ris, g_ue_ris, 1)
    #                         G2[i,n,a,u,bs,ue, :a, :] = g
    #                         # for antenna in range(a):
    #                         #     G1[i,n,a,u,count]=(G2[i,n,a,u,bs, ue, antenna, 0].real)
    #                         #     count+=1
    #                         #     G1[i,n,a,u,count]=(G2[i,n,a,u,bs, ue, antenna, 0].imag)
    #                         #     count+=1



    # np.savez("./H_record/record_H_b%du%da%dru%d" % (bs, ue, antenna, unit), H=G1_record, H2=G2_record,
    #          Hbr=G_bsris_record,
    #          Hrn=G_ueris_record, Hbn=G_bsue_record)
    np.savez("./H_record/record_H" , H=[], H2=[],
             Hbr=G_bsris_record,
             Hrn=G_ueris_record, Hbn=G_bsue_record)

def test(ue,antenna,unit,tr):
    if (ue == 6):
        mec_p_max = 25
    else:
        mec_p_max = 30
    simulation(10, 1000, ue, 3, unit_num=unit, antenna_num=antenna, target_r_min=tr,
               transmit_p_max=1,
               mec_p_max=mec_p_max, test_mode=True, open_matlab=True, mec_rule=rule,
               test_step=251, double_q=True, improved_strategy=True, concern_all=True, load_H_path="./H_record")

if __name__ == "__main__":

    # ue_arr = [10]
    # antenna_arr = [10]
    # unit_arr = [40]
    # record_H(10, 3, 16, 80,1500)
    # for ue in ue_arr:
    #     for antenna in antenna_arr:
    #         for unit in unit_arr:
    #             record_H(10, 3, 10, 40,2000)

    # ue_arr = [8,6]
    # antenna_arr = [16]
    # unit_arr = [80]
    #
    # # simulation(6000, 256, 6, 3, unit_num=30, antenna_num=5, r_min=r, target_r_min=r, bw=bw, transmit_p_max=1,
    # # mec_p_max=25, test_mode=False, open_matlab=False, prioritized=True, double_q=False,improved_strategy=False,
    # # test_step=100, mec_rule="default", load_H_path="./H_record")  # "./H_record"
    # for ue in ue_arr:
    #     for antenna in antenna_arr:
    #         for unit in unit_arr:
    #             if(ue==8 and antenna==5 and unit ==20):continue
    #             if(ue==6):
    #                 mec_p_max=25
    #             else:
    #                 mec_p_max=30
    #             simulation(1500, 255, ue, 3, unit_num=unit, antenna_num=antenna, target_r_min=30, transmit_p_max=1,
    #                                 mec_p_max=mec_p_max, test_mode=False, open_matlab=False, prioritized=True,double_q=True,improved_strategy=True,concern_all=True,
    #                                    test_step=100,mec_rule="default", load_H_path="./H_record")#"./H_record"
    #             # simulation_stage2(100, 256, ue, 3, unit_num=unit, antenna_num=antenna, target_r_min=30, transmit_p_max=1,
    #             #            mec_p_max=mec_p_max, open_matlab=True, prioritized=True, double_q=True,
    #             #            improved_strategy=True, concern_all=True,
    #             #            mec_rule="default", load_H_path="./H_record")  # "./H_record"
    #
    # exit(0)

    # # 测试，只记录rand Omega
    # ue_arr = [6,8]
    # antenna_arr = [16]
    # unit_arr = [80]
    # for ue in ue_arr:
    #     for antenna in antenna_arr:
    #         for unit in unit_arr:
    #                 simulation_random_omega(ue, 3, unit,antenna, 30, 100, test_step=1500, detect_times=1500)
    # # plot_simulation_results_ue_rates([8,8,8,8], [3,3,3,3], [30,30,30,30],
    # #                                  ["default","default","default","default","default","default"],[True,False,True,False,True,False],[5,5,5,5],
    # #                                  [20,20,30,30], [100,100,100,100],'Number of time slots','Average transmit rates (Mbps)',
    # #                                251,50,10,[False,False,False,False,True])
    # exit(0)

    #测试
    # ue_arr = [6]
    # antenna_arr = [16]
    # unit_arr = [80]#[20,30]
    # # 6 8 30 default
    # target_r_min = [40]
    # mec_rule_arr = ['exhaustion']
    # # mec_rule_arr = ['default','exhaustion']
    # for ue in ue_arr:
    #     for antenna in antenna_arr:
    #         for unit in unit_arr:
    #             for rule in mec_rule_arr:
    #                 for tr in target_r_min:
    #                     if (ue == 6):
    #                         mec_p_max = 25
    #                     else:
    #                         mec_p_max = 30
    #                     # t = threading.Thread(target= test,args=(ue, antenna, unit, tr))
    #                     # t.start()
    #                     # print("%d%d%d%d"%(ue, antenna, unit, tr))
    #                     simulation(10, 1000, ue, 3, unit_num=unit, antenna_num=antenna, target_r_min=tr,
    #                                transmit_p_max=1,
    #                                mec_p_max=mec_p_max, test_mode=True, open_matlab=True, mec_rule=rule,
    #                                test_step=251,double_q=True,improved_strategy=True,concern_all=True, load_H_path="./H_record")
    # exit(0)
    # plot_simulation_results_bs_power_rule([6, 6, 6, 8, 8, 8], [3, 3, 3, 3, 3, 3], [30, 30, 30, 30, 30, 30],
    #                                       [5, 5, 5, 5, 5, 5], [30, 30, 30, 30, 30, 30], [100, 100, 100, 100, 100, 100],
    #                                       ["default", "min", "max", "default", "min", "max"], 'Number of time slots',
    #                                       'Average transmit power (dBm)', 251, 50, 10,
    #                                       [False, True, False, False, False])
    # plot_simulation_results_AO_algorithm("Iterations", 'Average trasnmit power(dBm)',
    #                                      ['AO', 'AO-random-passive', 'AO-random-active'], disp_gap=1)
    # plot_simulation_results_bs_power_rule([6, 6, 6,6, 8, 8, 8,8], [3, 3, 3,3,3, 3, 3, 3], [80, 80,80,80, 80, 80, 80, 80],
    #                                       [16,16, 16, 16, 16, 16,16,16], [40, 40, 40, 40, 40, 40,40,40], [],
    #                                       ["default", "min", "max",'exhaustion', "default", "min", "max",'exhaustion'], 'Number of time slots',
    #                                       'Average transmit power (dBm)', 301, 50, 20,
    #                                       [False, True, False, False, False])
    # plot_simulation_results_total_power([6, 6, 6, 6,8, 8, 8, 8], [ 3, 3,3,3,3, 3,3,3], [ 80, 80,80,80,80, 80,80,80],
    #                                     [ 16, 16,16,16, 16, 16,16,16], [ 40, 40,40,40,40, 40,40,40],
    #                                     [],
    #                                     [ "default", "min", "max",'exhaustion', "default", "min", "max",'exhaustion'],
    #                                     [ True, True,True, True,True, True,True, True], 'Number of time slots',
    #                                     'Average total weighted power(dBm)', 301, 50, 20, [6, 8], [5, 5],
    #                                     [False, True, False, False, False])

    # plot_simulation_results_ue_rates([8,8,8,8], [3,3,3,3], [30,30,30,30],
    #                                  ["default","default","default","default","default","default"],[True,False,True,False,True,False],[5,5,5,5],
    #                                  [40,40,80,80], [],'Number of time slots','Average transmit rates (Mbps)',
    #                                231, 30,10,[False,False,False,False,True])
    # plot_simulation_results_bs_power([8,8,6, 6], [3, 3,3,3],
    #                                  [80, 80,80,80],
    #                                  ["default", "default", "default", "default"], [True, True, True, True],
    #                                  [16, 16, 16, 16],
    #                                  [True, False, True, False],
    #                                  [30, 30, 30, 30], [100, 100, 100, 100],
    #                                  'Number of time slots', 'Average transmit power(dBm)', 251, 50, 10,
    #                                  [], [], [False, True, False, False, False])
    exit(0)

