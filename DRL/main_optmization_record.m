function [record] = main_optmization_record(NumberOfBS,NumberOfFov,NumberofAntenna,NumberofRISUnit,N_0,P_max,R_min,epsilon,epsilon_fix,BW,total_iters,mode)
init_power=0;
init_rates=0;
opt_power=0;
opt_rates=0;
opt_rates_noCoMP = 0;
data = load("data.mat");
G = data.G;
h_ue_ris= data.gnr;
h_bs_ris=data.gbr;
h_bs_ue =data.gbn;
opt_G = 0;
prior_u=zeros(NumberofRISUnit,1);
for  i =1:NumberofRISUnit
init_theta = pi/2;
    prior_u(i)=sin(init_theta)+cos(init_theta)*j;
end
target = 4;
prior_omegas =0.1*(ones([NumberOfBS,NumberOfFov,NumberofAntenna])+ones([NumberOfBS,NumberOfFov,NumberofAntenna])*j);
current_omegas = prior_omegas;
[init_power,init_rates,init_rates_noComp]=rs_validate(BW,current_omegas,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);
u = prior_u;
record = zeros(1,total_iters+1);
record(1,1)=mean(init_power);
%record_all
rate=1e-4;
for i=1:total_iters
    prior_omegas = current_omegas;
    if(mode==0||mode==1)
        [opt_omegas,status]=cvx_optimization_Taylor(BW,current_omegas,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0,P_max,R_min);
        if(i>target)
             rate=rate*(0.6+rand()*0.1);
             current_omegas=(1-rate)*prior_omegas+rate*opt_omegas;
        else
             current_omegas=opt_omegas;
        end
         if(status  ~="Solved" )%&& status ~= "Inaccurate/Solved"
            disp("omega failed with RIS");
            rs=0;
            return;
         end
    end
    [opt_power,opt_rates,opt_rates_noCoMP]=rs_validate(BW,current_omegas,epsilon,epsilon_fix,NumberOfFov,NumberOfBS,NumberofAntenna,G,N_0);
    record(1,i+1)=mean(opt_power);

    if(i==total_iters)
        break;
    end
    if(mode==0||mode==2)
             [u,G,a,status] = cvx_optimization_RIS(BW,u,NumberofRISUnit,current_omegas,epsilon,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,h_bs_ue,h_ue_ris,h_bs_ris,N_0,R_min);
             if(status ~="Solved")
                disp("RIS phase failed");
                rs=0;
                return;
             end
    end
end

    opt_G = G;
  end

