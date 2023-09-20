function [g]= G_gain_cal(h_cue_bs, h_irs_bs, h_cue_irs, reflect)
%     '''
%     计算综合信道增益G
%     h_cue_bs:用户到基站的信道增益
%     h_irs_bs：IRS到基站的信道增益，是个一行K列的矩阵
%     h_cue_irs：用户到IRS的信道增益，是一个K行一列的矩阵
%     reflect：反射矩阵，是一个K行K列的矩阵
%     :return:一个综合信道增益的值
%     '''
%     # print("h_irs_bs",h_irs_bs)
%     # print("reflect",reflect)
    temp = h_irs_bs'*reflect;
%     # print("temp",temp)
    h_cue_irs_bs = temp*h_cue_irs;
    g = h_cue_bs + h_cue_irs_bs;