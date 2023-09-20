function [power_noRIS,rates_noRIS,opt_rates_noCoMP,rs] = main_optmization_NoRIS(NumberOfBS,NumberOfFov,NumberofAntenna,N_0,P_max,R_min,epsilon,epsilon_fix,BW)
% epsilon = load("epsilon").Epsilon;
rs=1;
init_power=0;
init_rates=0;
opt_power=0;
opt_rates=0;
rates_noRIS= 0;
opt_rates_noCoMP=0;
power_noRIS = 0;
data = load("data.mat");

% h_ue_ris= data.gnr;
% h_bs_ris=data.gbr;
h_bs_ue =data.gbn;
G = data.G;
opt_G_real = 0;
opt_G_imag = 0;
% G = load("dataG.mat").G;
% G= zeros([NumberOfBS,NumberOfFov,NumberofAntenna]);
%No RIS

 %初始化一个omega
 prior_omegas =0.1*(ones([NumberOfBS,NumberOfFov,NumberofAntenna])-ones([NumberOfBS,NumberOfFov,NumberofAntenna])*j);
% prior_omegas =0.1*(ones([NumberOfBS,NumberOfFov,NumberofAntenna])+ones([NumberOfBS,NumberOfFov,NumberofAntenna])*j); %初始化一个omega
%初始化阶段：步骤0，根据初始参数计算发射功率和速率 With Ris
%    [init_power,init_rates]=rs_validate(BW,prior_omegas,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);
   %[a,init_power_noRIS,init_rates_noRIS]=rs_validate(BW,prior_omegas,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,h_bn,N_0);
    current_omegas_noRIS= prior_omegas;
    %优化noRIS下的情况
     for i =1:2
         prior_omegas = current_omegas_noRIS;
          [opt_omegas_noRIS,status]=cvx_optimization_Taylor(BW,current_omegas_noRIS,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0,P_max,R_min);
%           prior_omegas_noRIS=current_omegas_noRIS;
         current_omegas_noRIS=opt_omegas_noRIS;
         if(status ~="Solved")
            disp("No_RIS taylor pahse failed without RIS");
            rs=0;
            return;
         end
     end
     
[power_noRIS,rates_noRIS,opt_rates_noCoMP]=rs_validate(BW,current_omegas_noRIS,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);
% [power_noRIS,rates_noRIS,opt_rates_noCoMP]=rs_validate2(BW,prior_omegas,current_omegas_noRIS,epsilon,epsilon_fix,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);
%       [current_omegas_noRIS,status]=cvx_optimization_Taylor(BW,current_omegas_noRIS,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,h_bn,N_0,P_max,R_min);
%      if(status ~="Solved")
%         disp("taylor phase2 failed without RIS");
%         rs=0;
%         return;
%      end

%       
%     if(rates_noRIS<R_min)
%         rs=0;
%        return;
%     end
  end

