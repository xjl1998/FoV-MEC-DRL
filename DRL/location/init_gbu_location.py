import os
from os.path import dirname, abspath
import numpy as np
from DRL.function_all import dist_calc_x, dist_calc_y

np.random.seed(223)

# from env import IrsNoma
dir_path = dirname(abspath(__file__))
if __name__ == "__main__":
    max_step = 200010
    max_gfu = 10
    coord_cue = np.zeros((max_step,max_gfu, 3))
    cue_generate_limit = 20
    max_speed = 3
    x_min_edge = -35
    y_min_edge = -10
    x_max_edge = -25
    y_max_edge = 10
    gfu_min_x = 26
    gfu_max_x = 46
    gbu_min_x = -35
    user_angle = "random"

    for i in range(max_gfu):
        coord_cue[0,i, 0] = -30 + (2*np.random.rand()-1)*5
        coord_cue[0, i,1] = (2*np.random.rand()-1)*10
        coord_cue[0, i,-1] = 0

    print("第",0,"个坐标生成成功：",coord_cue[0])
    for k in range(1,max_step):
        for i in range(max_gfu):
            speed_lst=np.random.uniform(0,max_speed)
            angle_lst=np.random.uniform(0,2*np.pi)
            coord_cue[k,i, 0] = dist_calc_x(0,coord_cue[k-1,i, 0],speed_lst,angle_lst,[],gbu_min_x,x_max_edge,max_speed)
            coord_cue[k, i,1] = dist_calc_y(0,coord_cue[k-1,i, 1],speed_lst,angle_lst,[],y_min_edge,y_max_edge,max_speed)
            coord_cue[k, i,-1] = 0
    print(dir_path)
    np.save(dir_path+'/gbu.npy',coord_cue)

