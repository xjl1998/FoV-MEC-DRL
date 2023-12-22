function [init_power,init_rates,opt_power,opt_rates,opt_rates_noCoMP,opt_G,rs] = main_optmization(NumberOfBS,NumberOfFov,NumberofAntenna,NumberofRISUnit,N_0,P_max,R_min,epsilon,epsilon_fix,BW)
init_power=0;
init_rates=0;
opt_power=0;
opt_rates=0;
opt_rates_noCoMP = 0;
data = load("data.mat");
h_ue_ris= data.gnr;
h_bs_ris=data.gbr;
h_bs_ue =data.gbn;
G = data.G;
opt_G = 0;
prior_u=zeros(NumberofRISUnit,1);
for  i =1:NumberofRISUnit
init_theta = pi/2;
    prior_u(i)=sin(init_theta)+cos(init_theta)*j;
end
rs = 1;
prior_omegas =0.1*(ones([NumberOfBS,NumberOfFov,NumberofAntenna])-ones([NumberOfBS,NumberOfFov,NumberofAntenna])*j);
current_omegas = prior_omegas;
[init_power,init_rates,init_rates_noComp]=rs_validate(BW,current_omegas,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);
   %[a,init_power_noRIS,init_rates_noRIS]=rs_validate(BW,prior_omegas,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,h_bn,N_0);
% [current_omegas]= gen_omega_ZF(NumberOfFov,NumberOfBS,NumberofAntenna,G,1);
%       [current_omegas_noRIS,sum_p_noRIS,Rates_noRIS,status_noRIS]=cvx_optimization_Taylor(BW,prior_omegas_noRIS,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,h_bn,N_0,P_max,R_min);
%       [a_noRIS,power_noRIS,rates_noRIS]=rs_validate(BW,current_omegas_noRIS,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,h_bn,N_0);%验证omega得到真实功率
%       prior_omegas_noRIS=current_omegas_noRIS;
target = 4;
u = prior_u;

for i=1:target
    prior_omegas = current_omegas;
    [opt_omegas,status]=cvx_optimization_Taylor(BW,current_omegas,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0,P_max,R_min);
     current_omegas=opt_omegas;
%      [opt_power0,opt_rates0]=rs_validate(BW,current_omegas,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);%验证新u得到的真实速率
% [opt_power,opt_rates,opt_rates_noCoMP]=rs_validate2(BW,prior_omegas,current_omegas,epsilon,epsilon_fix,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);%验证速率

%      [opt_power0,opt_rates0,opt_rates0_noComp]=rs_validate(BW,current_omegas,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);
    if(status  ~="Solved" )%&& status ~= "Inaccurate/Solved"
        disp("omega failed with RIS");
        rs=0;
        return;
    end
    
%     [opt_power,opt_rates,opt_rates_noCoMP]=rs_validate2(BW,prior_omegas,current_omegas,epsilon,epsilon_fix,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);%验证速率
    if(i==target)
        break;
    end
     [u,G,a,status] = cvx_optimization_RIS(BW,u,NumberofRISUnit,current_omegas,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,h_bs_ue,h_ue_ris,h_bs_ris,N_0,R_min);
%     [opt_power,opt_rates]=rs_validate(BW,current_omegas,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);%验证新u得到的真实速率
    if(status ~="Solved")
        disp("RIS phase failed");
        rs=0;
        return;
    end
      


end
%     [current_omegas,status]=cvx_optimization_Taylor(BW,current_omegas,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0,P_max,R_min);
%     if(status~="Solved" )
%         disp("taylor phase2 failed with RIS");
%          rs=0;
%          return;
%     end
%   [p,r,r2]=rs_validate(BW,current_omegas,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);
% [opt_power,opt_rates,opt_rates_noCoMP]=rs_validate2(BW,prior_omegas,current_omegas,epsilon,epsilon_fix,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);%验证速率
[opt_power,opt_rates,opt_rates_noCoMP]=rs_validate(BW,current_omegas,epsilon,epsilon_fix,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);

  %     if(opt_rates <R_min )
%         disp("taylor phase2 rate failed with RIS");
%         rs=0;
%         return;
%     end

%     [opt_power,opt_rates,opt_rates_noCoMP]=rs_validate(BW,current_omegas,epsilon,epsilon_fix,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);%验证速率
%     if(opt_rates <R_min )
%         disp("taylor phase2 rate failed with RIS");
%         rs=0;
%         return;
%     end
    opt_G = G;
    %opt_G_real = real(G);
    %opt_G_imag = imag(G);
  end

