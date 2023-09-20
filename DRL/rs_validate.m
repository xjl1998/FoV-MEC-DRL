function [opt_power,opt_rates,opt_rates_noCoMP] = rs_validate(BW,omegas,epsilon,epsilon_fix,NumberOfFov,NumberOfBS,NumberofAntenna,h,N_0)
%     W2 = 40;

sum_up = zeros([NumberOfFov]);
sum_down = zeros([NumberOfFov]);
sum_up2 = zeros([NumberOfFov]);
sum_down2 = zeros([NumberOfFov]);
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


        
%         for idx1 = 1:1:K
%         p1 = reshape(P(idx1,:),[K,1]);
%         ds = abs(H(idx1,:)*p1)^2;
%         int = 0;
%         for idx2 = 1:1:K
%             if idx2 ~= idx1
%                 p2=reshape(P(idx2,:),[K,1]);
%                 int = int + abs(H(idx1,:)*p2)^2;
%             end
%         end
%         sinr_k = ds/(sigma2+int);
%         c(idx1) = log2(1+sinr_k);
%         end

    
        %计算有CoMP速率
            for  f = 1:NumberOfFov
                    sum_down(f,1)=  sum_down(f,1)+ N_0;
                    for  b = 1:NumberOfBS
                        for  f2 = 1:NumberOfFov
                                   omega_r=reshape(omegas(b,f2,:),[NumberofAntenna,1]);
                                   h_r=reshape(h(b,f,:),[NumberofAntenna,1]);
                                   if(epsilon(b,f2)==1)
                                       if (f2==f)
                                            sum_up(f,1) = sum_up(f)+pow_abs(omega_r'*h_r,2);
                                       else 
                                            sum_down(f,1)= sum_down(f)+pow_abs(omega_r'*h_r,2);
                                        end
                                   end
                        end
                end
            end



        Rates=zeros([NumberOfFov,1]);
        for  i=1:NumberOfFov
            Rates(i,1)=BW*log2(1+(sum_up(i,1)/sum_down(i,1)) );
        end
        opt_rates = Rates;



            %计算无CoMP速率
            for  f = 1:NumberOfFov
                    sum_down2(f)=  sum_down2(f)+ N_0;
                    for  b = 1:NumberOfBS
                        for  f2 = 1:NumberOfFov
                                   omega_r=reshape(omegas(b,f2,:),[NumberofAntenna,1]);
                                   h_r=reshape(h(b,f,:),[NumberofAntenna,1]);
                                   if(epsilon(b,f2)==1)
                                       if (f2==f&&epsilon_fix(b,f2)==1)
                                            sum_up2(f) = sum_up2(f)+pow_abs(omega_r'*h_r,2);
                                       else 
                                            sum_down2(f)= sum_down2(f)+pow_abs(omega_r'*h_r,2);
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