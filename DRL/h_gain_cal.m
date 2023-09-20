function [h]=  h_gain_cal(coord_a, coord_b, a,small_fading_style,irs_m)
%     '''
%     :param coord_a:用户或者基站坐标
%     :param coord_b:用户或者基站坐标
%     :param a:路径损耗系数
%     :param small_fading_style:小尺度衰落
%     irs_m:irs元件个数
%     :return:增益
%     '''
    if small_fading_style == "Rayleigh"
        small = normrnd(0,1/2,1);
%         small = np.random.normal(0,1/2,1)+np.random.normal(0,1/2,1)*1j
        ad = np.array(np.array(coord_a) - np.array(coord_b)).reshape(1, 3)
        d = norm(ad);
        if d == 0
            d = 0.000001;
        end
        h = sqrt(0.001*  power(d,(-a))) * small;
    else
        h=zeros(irs_m);
        for i=0:irs_m
            small = sqrt(2/3)*(exp(0) * (cos(rand()*2*pi) + sin(rand()*2*pi) * 1j))
                    +sqrt(1/3)*(normrnd(0,1/2,1)+normrnd(0,1/2,1)*1j)
%             # small=1
            ad = coord_a - coord_b
            d = norm(ad);
            if d == 0
                d = 0.000001;
            end
            h(i)=sqrt(0.001* power(d,(-a)) * small);
        end
    end