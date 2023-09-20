function [result,status] = cvx_optimization_Taylor(BW,prior_omegas,epsilon,epsilon_fix,NumberOfFov,NumberOfBS,NumberofAntenna,h,N_0,P_max,R_min)
%     W2 = 40;

    cvx_begin  quiet
    cvx_solver Mosek 

        variable omegas(NumberOfBS,NumberOfFov,NumberofAntenna,1) complex;
        expression sum_up(NumberOfFov);
        expression sum_down(NumberOfFov);
        expression mr(NumberOfFov);
        expression im(NumberOfFov);
        expression diff(NumberOfFov,NumberOfBS,NumberOfFov);
        expression target;
        expression p;
        expression sum_P(NumberOfBS);


        %计算速率相关的约束
            for  f = 1:NumberOfFov
                    sum_down(f)=  sum_down(f)+ N_0;
                    for  b = 1:NumberOfBS
                        for  f2 = 1:NumberOfFov
                                   omega_r=reshape(omegas(b,f2,:),[NumberofAntenna,1]);
                                   h_r=reshape(h(b,f,:),[NumberofAntenna,1]);
                                   prior_omega = reshape(prior_omegas(b,f2,:),[NumberofAntenna,1]);
                                   if(epsilon(b,f2)==1)
                                       if (f2==f&&epsilon_fix(b,f2)==1)
%                                             im(f)=2*real(h_r'*prior_omega*omega_r'*h_r)-pow_abs(prior_omega'*h_r,2);
                                            sum_up(f) = sum_up(f)+2*real(h_r'*prior_omega*omega_r'*h_r)-pow_abs(prior_omega'*h_r,2);
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
                          p = p +n2;
                        end
            end
       end

%                   for f1=1:NumberOfFov
%                            for b=1:NumberOfBS
%                             h_r=reshape(h(b,f1,:),[NumberofAntenna,1]);
%                                for f2=1:NumberOfFov
%                                      if(f2~=f1&&epsilon(b,f2)==1)
%                                          omega_r=reshape(omegas(b,f2,:),[NumberofAntenna,1]);
%                                          diff(f1,b,f2) = omega_r'*h_r;
%                                      end
%                                end
%                            end
%             end     
%        
       
        minimize p;
        subject to
                  
                  for  i=1:NumberOfFov
                      1e4*(pow_p(2,R_min/BW)-1)*(sum_down(i))<=1e4*(sum_up(i));
                  end
%                    1e4*(pow_p(2,R_min/BW)-1)*sum_down(i)<=1e4*sum_up(i);   
%                     for  i=1:NumberOfFov
%                          im(i)>=0;
%                    end
                   for  i=1:NumberOfBS
                         sum_P(i)<=P_max;
                   end
%                    
%             for f1=1:NumberOfFov
%                 for b=1:NumberOfBS
%                                for f2=1:NumberOfFov
%                                      if(f2~=f1)
%                                          diff(f1,b,f2) == 0;
%                                      end
%                                end
%                 end
%             end
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
%           Rates=zeros([NumberOfFov,1]);
%     for  i=1:NumberOfFov
%         Rates(i,1)=BW*log2(1+sum_up(f)/sum_down(f) );
%     end
     
    Rates=zeros([NumberOfFov,1]);
    for i=1:NumberOfFov
        Rates(i,1)=BW*log2(1+sum_up(i)/sum_down(i) );
    end
%     diffs = zeros([NumberOfFov,NumberOfBS,NumberOfFov]);
%        for f1=1:NumberOfFov
%                            for b=1:NumberOfBS
%                             h_r=reshape(h(b,f1,:),[NumberofAntenna,1]);
%                                for f2=1:NumberOfFov
%                                      if(f2~=f1&&epsilon(b,f2)==1)
%                                          omega_r=reshape(omegas(b,f2,:),[NumberofAntenna,1]);
%                                          diffs(f1,b,f2) = omega_r'*h_r;
%                                      end
%                                end
%                            end
%        end     
            
    
    result =  full(omegas);
%     result1 = sum(sum_P);
    status = cvx_status;


end