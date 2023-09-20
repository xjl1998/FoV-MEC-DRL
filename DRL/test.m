clear; clc; close all;
%测试凸优化工具
NumberOfFov =8;
NumberOfUe=NumberOfFov;
NumberOfBS=3;
NumberofAntenna =5;
NumberofRISUnit= 30;
% epsilon = load("data.mat").E;
G = load("data.mat").G;
epsilon = load("data.mat").E;
h_ue_ris= load("data.mat").gnr;
h_bs_ris=load("data.mat").gbr;
h_bs_ue =load("data.mat").gbn;
%初始化一个theta
%计算固定u
prior_u=zeros(NumberofRISUnit,1);
for  i =1:NumberofRISUnit
%     init_theta = 2*rand()*pi;
init_theta = pi/2;
    prior_u(i)=sin(init_theta)+cos(init_theta)*j;
end
%     for  b=1:NumberOfBS
%         for  f=1:NumberOfFov
%             hbn=reshape(h_bs_ue(b,f,:,:),[NumberofAntenna,1]);
%             hrn=reshape(h_ue_ris(b,f,:,:),[NumberofRISUnit,1]);
%             hbr=reshape(h_bs_ris(b,:,:),[NumberofRISUnit,NumberofAntenna]);
%             G2(b,f,:)=(hbr'*diag(prior_u)*hrn+hbn);
%         end
%     end
    
% tables =load("dataActionTables.mat").table;

%这里对G做转换变为H
N_0_dbm = -174 + 10 * log10(1e7);
N_0= 10^((N_0_dbm - 30) / 10);
total_iter=2;
BW=100;


prior_u_temp=prior_u;
G = load("data.mat").G;
% gen_omega_ZF(NumberOfFov,NumberOfBS,NumberofAntenna,G);
% [omegas]= gen_omega_ZF(NumberOfFov,NumberOfBS,NumberofAntenna,G,1);
% H = reshape(G(1,:,:),[NumberOfFov,NumberofAntenna]);
% omega = reshape(omegas(1,:,:),[NumberOfFov,NumberofAntenna]);
% rate = SumRate(H,omega,N_0);
epsilon_fix = [1,1,1,1,0,0;0,0,0,0,1,1;0,0,0,0,0,0];


[init_power,init_rates,opt_power,opt_rates,opt_rates_noCoMP,opt_G,rs]=main_optmization(NumberOfBS,NumberOfFov,NumberofAntenna,NumberofRISUnit,N_0,1,20,epsilon,epsilon,100);

[power_noRIS,rates_noRIS,opt_rates_noCoMP,rs]  = main_optmization_NoRIS(NumberOfBS,NumberOfFov,NumberofAntenna,N_0,1,20,epsilon,epsilon,100);
k  = sum(opt_power-power_noRIS)




return

%收集收敛曲线
P_max=1;
R_min=20;
NumberOfFov=8;
NumberofAntenna=5;
NumberofRISUnit=30;
total_iters = 50;
xs = [0:10];
data =zeros(3,total_iters+1);
[record_all] = main_optmization_record(NumberOfBS,NumberOfFov,NumberofAntenna,NumberofRISUnit,N_0,P_max,R_min,epsilon,epsilon,BW,total_iters,0);
[record_onlyW] = main_optmization_record(NumberOfBS,NumberOfFov,NumberofAntenna,NumberofRISUnit,N_0,P_max,R_min,epsilon,epsilon,BW,total_iters,1);
% [record_onlyRIS] = main_optmization_record(NumberOfBS,NumberOfFov,NumberofAntenna,NumberofRISUnit,N_0,P_max,20,epsilon,epsilon,BW,total_iters,2);
record_onlyRIS = repmat(record_all(1),[1,total_iters+1]);
data(1,:)= 10*log10(record_all/0.001);
data(2,:)= 10*log10(record_onlyW/0.001);
data(3,:)= 10*log10(record_onlyRIS/0.001);
save('AO.mat','data');
% 纵轴截断后图像
% 作者：凯鲁嘎吉 - 博客园 http://www.cnblogs.com/kailugaji/
% 参数设置
% y_interval=0.5;  %纵坐标两个刻度间隔距离
% y_break_start=13; % 截断的开始值
% y_break_end=25; % 截断的结束值

% adjust_value=0.4*y_interval; %微调截断处y坐标
% uptate_num=y_break_end-y_break_start-y_interval; %最高处曲线向下平移大小
% 超过截断结束位置的那些曲线统统向下平移uptate_num个长度
% for i=1:length(data(:, 1))
%     for j=1:length(data(i,:))
%          if (data(i, j)>y_break_end )
%             data(i, j)=data(i, j)-uptate_num;
%          end
%     end
% end

% 根据曲线的个数进行修改，这里曲线是3条
h=plot(xs, data(1, :), 'k*-', xs, data(2, :), 'g^-', xs, data(3, :), 'r-s', 'MarkerFaceColor','y', 'MarkerSize',7);
breakyaxis([13 25]);  % 截断纵坐标
set(gcf,'color','w') %后面背景变白

xlim([0 total_iters]); %横坐标范围
xlabel('x'); 
string='y';
ylabel(string);
legend('Line-1', 'Line-2', 'Line-3',  'Location', 'east');  %图例 根据曲线个数修改

% % 纵坐标截断设置
% ylimit=get(gca,'ylim');
% location_Y=(y_break_start+adjust_value-ylimit(1))/diff(ylimit);
% t1=text(0, location_Y,'//','sc','BackgroundColor','w','margin',eps, 'fontsize',13);
% set(t1,'rotation',90);
% t2=text(1, location_Y,'//','sc','BackgroundColor','w','margin',eps, 'fontsize',13); 
% set(t2,'rotation',90);

% 重新定义纵坐标刻度
% ytick=6:y_interval:27;
% set(gca,'ytick',ytick);
% ytick(ytick>y_break_start+eps)=ytick(ytick>y_break_start+eps)+uptate_num;
% for i=1:length(ytick)
%    yticklabel{i}=sprintf('%d',ytick(i));
% end
% set(gca,'yTickLabel', yticklabel, 'FontSize', 12, 'FontName', 'Times New Roman'); %修改坐标名称、字体



clf;
    figure(1);
    plot(xs(2:11), 10*log10(record_all(2:11)/0.001),'-r','LineWidth',1);
     hold on;
     plot(xs,10*log10(record_onlyRIS/0.001),'-g','LineWidth',1);
     hold on;
    plot(xs(2:11),10*log10(record_onlyW(2:11)/0.001),'-black','LineWidth',1);
     %为背景添加网格
    grid on
    %添加图形名称
    title("迭代优化后功率对比");
    %添加坐标轴名称
    xlabel("迭代次数");
    ylabel("基站平均功率");
  
    %添加图例及文字说明
    legend('optimize both active and passtive beamforming','optimize only passive beamforming','optimize only active beamforming');


 
    
% [init_power,init_rates,init_rates_noComp]=rs_validate(1,omegas,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);

for epoch=1:total_epochs
%     N_0_seq=zeros(total_epochs);
%     disp(['current epoch',epoch,', ',total_epochs-epoch,'epochs remained']);
%     N_0=10^((-174 + 10 * log10(rd(epoch)*1e7) - 30) / 10);
%     N_0_seq(epoch)=N_0;
    R_min=rd(epoch);
    prior_omegas = 0.1*(ones([NumberOfBS,NumberOfFov,NumberofAntenna])+j); %初始化一个omega
    prior_omegas_noRIS = prior_omegas;
    init_theta = pi;  %初始化一个theta
    %计算固定u
    prior_u=prior_u_temp;
    G = G_temp;
for iter=1:total_iter
%   if(iter==1)
%       %初始化阶段：步骤0，根据初始参数计算发射功率和速率 With Ris
%         [a,init_power,init_rates]=rs_validate(BW,prior_omegas,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);
% %         rate_records(iter) = sum(init_rates);
% %         power_records(iter) = sum(init_power);
%         [a,init_power_noRIS,init_rates_noRIS]=rs_validate(BW,prior_omegas,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,h_bn,N_0);
% %         rate_records_noRis(iter) =sum(init_rates_noRIS);
% %         power_records_noRis(iter) = sum(init_power_noRIS);
%   else
       %优化noRIS下的情况
      [current_omegas_noRIS,sum_p_noRIS,Rates_noRIS,status_noRIS]=cvx_optimization_Taylor(BW,prior_omegas_noRIS,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,h_bn,N_0,P_max,R_min);
      [a_noRIS,power_noRIS,rates_noRIS]=rs_validate(BW,current_omegas_noRIS,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,h_bn,N_0);%验证omega得到真实功率
%       rate_records_noRis(iter) = sum(rates_noRIS);
%       power_records_noRis(iter) = sum(sum_p_noRIS);
      prior_omegas_noRIS=current_omegas_noRIS;
    %优化阶段：步骤一，根据当前增益，优化w从而得到最小发射功率
    [current_omegas,sum_p,Rates,status]=cvx_optimization_Taylor(BW,prior_omegas,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0,P_max,R_min);
     [a,power,rates]=rs_validate(BW,current_omegas,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);%验证新u得到的真实速率
    %优化阶段：步骤二，根据当前omega，优化RIS反射theta
    [u,G,a,status] = cvx_optimization_RIS(BW,prior_u,NumberofRISUnit,current_omegas,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,h_bn,h_ue_ris,h_bs_ris,N_0,R_min);
    [a,power,rates]=rs_validate(BW,current_omegas,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);%验证新u得到的真实速率
    prior_u = u;
%     rate_records(iter)=sum(rates);
%     power_records(iter)=sum(power);
    prior_omegas= current_omegas;
  end
    rate_records(epoch) = sum(rates);
    power_records(epoch) = sum(power);
    rate_records_noRis(epoch) = sum(rates_noRIS);
    power_records_noRis(epoch) =  sum(sum_p_noRIS);
end
% end


   
%    for f = 1:NumberOfFov
%                     sum_down= N_0;
%                     sum_up=0;
%                     for b = 1:NumberOfBS
%                         for f2 = 1:NumberOfFov
%                                    omega_r=reshape(current_omegas(b,f2,:),[NumberofAntenna,1]);
%                                    h_r=reshape(G(b,f2,:),[NumberofAntenna,1]);
%                                    prior_omega = reshape(prior_omegas(b,f2,:),[NumberofAntenna,1]);
%                                    if(f2~=f)
%                                            sum_down= sum_down+ epsilon(b,f2)*pow_abs(omega_r'*h_r,2);
%                                    else
%                                        sum_up = sum_up+epsilon(b,f2)*2*real(h_r'*prior_omega*omega_r'*h_r)+pow_abs(prior_omega'*h_r,2);
%                                    end
%                         end
%                       
%                     end
%                 disp(2^R_min*sum_down-sum_up);
%             end
% [result,result1,result2,status] = cvx_optimization(W,epsilon,NumberOfFov,NumberOfUe,NumberOfBS,NumberofAntenna,H,N_0,P_max,R_min);
% [result,result1,result2,status] = cvx_optimization_Taylor(W,epsilon,NumberOfFov,NumberOfUe,NumberOfBS,NumberofAntenna,G,N_0,P_max,R_min);