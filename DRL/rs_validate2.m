function [opt_power,opt_rates,opt_rates_noCoMP] = rs_validate2(BW,prior_omegas,omegas,epsilon,epsilon_fix,NumberOfFov,NumberOfBS,NumberofAntenna,h,N_0)
%     W2 = 40;

sum_up = zeros([NumberOfFov,1]);
sum_down = zeros([NumberOfFov,1]);
sum_up2 = zeros([NumberOfFov,1]);
sum_down2 = zeros([NumberOfFov,1]);
p=0;

sum_P = zeros([NumberOfBS,1]);


                        %计算功率
       for  b = 1:NumberOfBS
            for  f = 1:NumberOfFov
                        if(epsilon(b,f)==1)
                                           omega_r=reshape(omegas(b,f,:),[NumberofAntenna,1]);
                          n2=sum(pow_abs(omega_r,2));
                         sum_P(b,1)=sum_P(b,1)+n2;
                          p = p +n2;
                        end
            end
       end

    opt_power = sum_P;


        %计算有CoMP速率
        
            for  f = 1:NumberOfFov
                    sum_down(f,1)=  sum_down(f,1)+ N_0;
                    for  b = 1:NumberOfBS
                        for  f2 = 1:NumberOfFov
                                   omega_r=reshape(omegas(b,f2,:),[NumberofAntenna,1]);
                                   h_r=reshape(h(b,f,:),[NumberofAntenna,1]);
                                   prior_omega = reshape(prior_omegas(b,f2,:),[NumberofAntenna,1]);
                                   if(epsilon(b,f2)==1)
                                       if (f2==f)
                                            sum_up(f,1) = sum_up(f,1)+2*real(h_r'*prior_omega*omega_r'*h_r)-pow_abs(prior_omega'*h_r,2);
                                       else 
                                            sum_down(f,1)= sum_down(f,1)+pow_abs(omega_r'*h_r,2);
                                        end
                                   end
                        end
                end
            end
            
%             for  f = 1:NumberOfFov
%                     sum_down(f)=  sum_down(f)+ N_0;
%                     for  b = 1:NumberOfBS
%                         for  f2 = 1:NumberOfFov
%                                    omega_r=reshape(omegas(b,f2,:),[NumberofAntenna,1]);
%                                    h_r=reshape(h(b,f,:),[NumberofAntenna,1]);
%                                    prior_omega = reshape(prior_omegas(b,f2,:),[NumberofAntenna,1]);
%                                    if(epsilon(b,f2)==1)
%                                        if (f2==f)
%                                             sum_up(f) = sum_up(f)+ 2*real(h_r'*prior_omega*omega_r'*h_r)-pow_abs(prior_omega'*h_r,2);
%                                        else 
%                                             sum_down(f)= sum_down(f)+pow_abs(omega_r'*h_r,2);
%                                         end
%                                    end
%                         end
%                 end
%             end



        Rates=zeros([NumberOfFov,1]);
        for  i=1:NumberOfFov
            Rates(i,1)=BW*log2(1+sum_up(i,1)/sum_down(i,1) );
        end
        opt_rates = Rates;



            %计算无CoMP速率
            for  f = 1:NumberOfFov
                    sum_down2(f,1)=  sum_down2(f,1)+ N_0;
                    for  b = 1:NumberOfBS
                        for  f2 = 1:NumberOfFov
                                   omega_r=reshape(omegas(b,f2,:),[NumberofAntenna,1]);
                                   h_r=reshape(h(b,f,:),[NumberofAntenna,1]);
                                   prior_omega = reshape(prior_omegas(b,f2,:),[NumberofAntenna,1]);
                                   if(epsilon(b,f2)==1)
                                       if (f2==f&&epsilon_fix(b,f2)==1)
                                            sum_up2(f,1) = sum_up2(f,1)+2*real(h_r'*prior_omega*omega_r'*h_r)-pow_abs(prior_omega'*h_r,2);
                                       else 
                                            sum_down2(f,1)= sum_down2(f,1)+pow_abs(omega_r'*h_r,2);
                                        end
                                   end
                        end
                end
            end
       
        Rates2=zeros([NumberOfFov,1]);
        for  i=1:NumberOfFov
            Rates2(i,1)=BW*log2(1+sum_up2(i,1)/sum_down2(i,1) );
        end
        opt_rates_noCoMP=Rates2;
                    



  



end