import os
import random

from os.path import dirname, abspath

import numpy as np
from  tqdm import  tqdm

np.random.seed(1)
if __name__ == "__main__":
    timeplot = 2000
    max_cue = 20
    dir_path = dirname(abspath(__file__))
    CUE_coord = np.load(dir_path + '\\cue.npy')
    coord_cue_random = np.zeros((timeplot,max_cue, 3))
    cue_coord_now = CUE_coord
    cue_generate_limit = 3
    loader = tqdm(range(timeplot))


    for t1 in loader:
        loader.set_description_str(("[timeslot %d| %d]" % (t1, timeplot)))
        if t1 ==0:
            coord_cue_random[t1,:,:] =  cue_coord_now[1,:,:]
        else:
            for k in range(max_cue):
                coord_cue_random[t1,k,0] = 8 + (2*np.random.rand()-1)*cue_generate_limit
                coord_cue_random[t1,k,1] = 10+(2*np.random.rand()-1)*cue_generate_limit
                coord_cue_random[t1,k, -1] = 0


    np.save(dir_path+'/cue.npy',coord_cue_random)

    # for t1 in range(timeplot):
    #     if(t1%2!=0):
    #         for k in range(max_cue):
    #             coord_cue_mix_random[t1, k, 0] = coord_cue_near_random[t1,k,0]
    #             coord_cue_mix_random[t1, k, 1] = coord_cue_near_random[t1, k, 1]
    #             coord_cue_mix_random[t1, k, -1] = coord_cue_near_random[t1, k, -1]
    #     else:
    #         for k in range(max_cue):
    #             coord_cue_mix_random[t1, k, 0] = coord_cue_far_random[t1, k, 0]
    #             coord_cue_mix_random[t1, k, 1] = coord_cue_far_random[t1, k, 1]
    #             coord_cue_mix_random[t1, k, -1] = coord_cue_far_random[t1, k, -1]
    # np.save(dir_path + '/cue_mix.npy', coord_cue_mix_random)
    print('保存完成')

