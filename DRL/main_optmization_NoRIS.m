function [power_noRIS,rates_noRIS,opt_rates_noCoMP,rs] = main_optmization_NoRIS(NumberOfBS,NumberOfFov,NumberofAntenna,N_0,P_max,R_min,epsilon,epsilon_fix,BW)
rs=1;
init_power=0;
init_rates=0;
opt_power=0;
opt_rates=0;
rates_noRIS= 0;
opt_rates_noCoMP=0;
power_noRIS = 0;
data = load("data.mat");
h_bs_ue =data.gbn;
G = data.G;
opt_G_real = 0;
opt_G_imag = 0;
prior_omegas =0.1*(ones([NumberOfBS,NumberOfFov,NumberofAntenna])-ones([NumberOfBS,NumberOfFov,NumberofAntenna])*j);
current_omegas_noRIS= prior_omegas;
     for i =1:2
         prior_omegas = current_omegas_noRIS;
          [opt_omegas_noRIS,status]=cvx_optimization_Taylor(BW,current_omegas_noRIS,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0,P_max,R_min);
         current_omegas_noRIS=opt_omegas_noRIS;
         if(status ~="Solved")
            disp("No_RIS taylor pahse failed without RIS");
            rs=0;
            return;
         end
     end
     
[power_noRIS,rates_noRIS,opt_rates_noCoMP]=rs_validate(BW,current_omegas_noRIS,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);
  end

