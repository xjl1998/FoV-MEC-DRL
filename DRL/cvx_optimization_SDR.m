function [result,status] = cvx_optimization_SDR(BW,epsilon,NumberOfFov,NumberOfBS,NumberofAntenna,H,N_0,P_max,R_min)

    cvx_begin quiet
    cvx_solver sedumi

        variable omegas(NumberOfBS,NumberOfFov,NumberofAntenna) complex;
        variable W_var(NumberOfBS,NumberOfFov,NumberofAntenna,NumberofAntenna) nonnegative;
        expression sum_up(NumberOfFov);
        expression sum_down(NumberOfFov);
        expression mr(NumberOfFov);
        expression target;
        expression sum_P(NumberOfBS);
  

                    for  f = 1:NumberOfFov
                    sum_down(f)=  sum_down(f)+ N_0;
                    for  b = 1:NumberOfBS
                        for  f2 = 1:NumberOfFov
                              
                                   o=reshape(omegas(b,f2,:),[NumberofAntenna,1]);
                                 
                                   k =  o*o;
                                   W = reshape(o*o',[NumberofAntenna,NumberofAntenna]);
                                   W_var(b,f2,:,:)= W;
                                   h=reshape(H(b,f2,:,:),[NumberofAntenna,NumberofAntenna]);
                                   if(epsilon(b,f2)==1)
                                       if (f2==f)
                                            sum_up(f) = sum_up(f)+trace(h*W);
                                       else 
                                            sum_down(f)= sum_down(f)+trace(h*W);
                                        end
                                   end
                        end
                end
            end

        
        
        
            for  f = 1:NumberOfFov
                    sum_down(f)=  sum_down(f)+ N_0;
                    for  b = 1:NumberOfBS
                        for  f2 = 1:NumberOfFov
                                   omega_r=reshape(omegas(b,f2,:),[NumberofAntenna,1]);
                                   h_r=reshape(H(b,f2,:),[NumberofAntenna,1]);
                                   prior_omega = reshape(prior_omegas(b,f2,:),[NumberofAntenna,1]);
                                   if(epsilon(b,f2)==1)
                                       if (f2==f)
                                            im(f,b,f2)=2*real(h_r'*prior_omega*omega_r'*h_r)-pow_abs(prior_omega'*h_r,2);
                                            sum_up(f) = sum_up(f)+im(f,b,f2);
                                       else 
                                            sum_down(f)= sum_down(f)+pow_abs(omega_r'*h_r,2);
                                        end
                                   end
                        end
                end
            end

       
        %计算功率相关的约束
       for  b = 1:NumberOfBS
            for  f = 1:NumberOfFov
                        if(epsilon(b,f)==1)
                          omega_r=reshape(omegas(b,f,:),[NumberofAntenna,1]);
                          n2=sum(pow_abs(omega_r,2));
                          sum_P(b)=sum_P(b)+n2;
                          target = target +n2;
%                           target = target +trace(W_var(b,f));
                        end
            end
       end
        

    
        minimize target;
        subject to
                  
                  for  i=1:NumberOfFov
                     100*(pow_p(2,R_min/BW)-1)*sum_down(i)-100*sum_up(i)<=0;   
                  end
                   for  i=1:NumberOfBS
                         sum_P(i)-P_max<=0;
                   end
%                    for f=1:NumberOfFov
%                        for b=1:NumberOfBS
%                            for f2=1:NumberOfFov
%                                 im(f,b,f2)>=0;
%                            end
%                        end
%                    end
                  
%                     
            
    cvx_end

%               
          Rates=zeros([NumberOfFov,1]);
    for  i=1:NumberOfFov
        Rates(i,1)=BW*log2(1+sum_up(f)/sum_down(f) );
    end
     
    result =  full(omegas);
%     result1 = sum(sum_P);
    status = cvx_status;


end