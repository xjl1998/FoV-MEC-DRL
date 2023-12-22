import  argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--inputSize', type=int, default=1,
    help='inputSize')
parser.add_argument(
    '--outputSize', type=int, default=1,
    help='outputSize')
parser.add_argument(
    '--timeStep', type=int, default=5,
    help='timeStep')

parser.add_argument(
    '--dimension', type=int, default=3,
    help='dimension')

parser.add_argument(
    '--interval', type=int, default=1,
    help='interval')

parser.add_argument(
    '--fOV_2DShape', type=int, default=[1024,1024],
    help='fOV_2DShape')

parser.add_argument(
    '--eyesight', type=int, default=2,
    help='eyesight')

parser.add_argument(
    '--e_greedy', type=int, default=0.90,
    help='e_greedy')
parser.add_argument(
    '--e_greedy_increment_c', type=int, default=0.0002,
    help='e_greedy_increment_c  0.0005')
#0.0001
parser.add_argument(
    '--gfu_bs_a', type=float, default=3.5,
    help='gfu_bs_a')
parser.add_argument(
    '--irs_bs_a', type=float, default=2.5,
    help='f_mec')
parser.add_argument(
    '--ue_irs_a', type=float, default= 2.5,
    help='ue_irs_a')
parser.add_argument(
    '--ue_bs_a', type=float, default=  3.5,
    help='ue_bs_a')

parser.add_argument(
    '--batch_size', type=float, default=64,
    help='batch_size')
parser.add_argument(
    '--dqn_lrc', type=float, default= 1e-4,
    help='dqn_lrc')
parser.add_argument(
    '--e_vr', type=float, default=10**(15),
    help='e_vr')
parser.add_argument(
    '--r_min', type=int, default=1,
    help='r_min')
parser.add_argument(
    '--fov_patch_num', type=int, default=64,
    help='fov_patch_num')
parser.add_argument(
    '--BW', type=int, default=400,
    help='BW')
parser.add_argument(
    '--training_interval', type=int, default=2,
    help='training_interval')
parser.add_argument(
    '--double_q', type=bool, default=True,
    help='double_q')
parser.add_argument(
    '--prioritized_r', type=bool, default=False,
    help='prioritized_r')

parser.add_argument(
    '--replace_target_iter', type=float, default=50,
    help='replace_target_iter')
parser.add_argument(
    '--antenna_num', type=int, default=8,
    help='antenna_num')
parser.add_argument(
    '--bs_num', type=int, default=3,
    help='bs_num')
parser.add_argument(
    '--ue_num', type=int, default=14,
    help='ue_num')
parser.add_argument(
    '--irs_num', type=float, default=2,
    help='irs_num')
parser.add_argument(
    '--p_max', type=float, default=80,
    help='p_max')
parser.add_argument(
    '--irs_units_num', type=int, default=40,
    help='irs_units_num')
parser.add_argument(
    '--memory_size', type=int, default=30000,
    help='irs_units_num')


# ue_bs_a = 3.5
# ue_irs_a = 2.5
# irs_bs_a = 2.5
# gfu_bs_a = 3.5#2.5
# F_VR = 3 * 10**9
# F_MEC = 10 * 10**9

# f_VR = 15
# f_MEC = 15

# k_m = 10**(-9)
# k_v = 10**(-9)

# E_MEC = 10**(20)
# E_VR = 10**(15)

# np.random.seed(1)
# BW = 40
# N_0_dbm = -174 + 10 * log10(BW)

# N_0 = np.power(10,(N_0_dbm - 30) / 10)
# N_0 = 10 ** ((N_0_dbm - 30) / 10)
# N_0 =0.00001
# ue_bs_a = 3
# ue_irs_a = 2.2
# irs_bs_a = 2.2
FLAGS, _ = parser.parse_known_args()


