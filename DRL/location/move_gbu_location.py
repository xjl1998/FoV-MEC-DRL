import os
import random
from os.path import dirname, abspath

import numpy as np
np.random.seed(1)

if __name__ == "__main__":
    timeplot = 10000
    max_cue = 10
    dir_path = dirname(abspath(__file__))
    CUE_all_coord = np.load(dir_path + '\\gbu.npy')
    coord_cue_random = np.zeros((timeplot,max_cue, 3))


    cue_coord_now = CUE_all_coord
    cue_generate_limit = 5
    for t1 in range(timeplot):
        if t1 ==0:
            a= coord_cue_random[t1,:,:]
            coord_cue_random[t1,:,:] =  cue_coord_now
        else:
            for k in range(max_cue):
                coord_cue_random[t1,k,0] = -30 + (2*np.random.rand()-1)*cue_generate_limit
                coord_cue_random[t1,k,1] = (2*np.random.rand()-1)*cue_generate_limit
                coord_cue_random[t1,k, -1] = 0
    np.save(dir_path+'/gbu_100001.npy',coord_cue_random)
    print('保存完成')

