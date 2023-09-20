function [result,result1,result2,status] = cvx_optimization_RIS2(BW,old_u,unit_num,omegas,epsilon,epsilon_fix,NumberOfFov,NumberOfBS,NumberofAntenna,h_bn,h_rn,h_br,N_0,R_min)

    cvx_begin quiet
    cvx_solver Mosek
        expression a(NumberOfFov);
        expression rest;
        variable u(unit_num,1) complex;
        expression sum_up(NumberOfFov);
        expression sum_down(NumberOfFov);
        expression p;
        expression sum_P(NumberOfBS);
        %计算速率相关的约束
            for  f = 1:NumberOfFov
                    sum_down(f)= N_0;
                    for  b = 1:NumberOfBS
                        for  f2 = 1:NumberOfFov
                                   omega_r=reshape(omegas(b,f2,:),[NumberofAntenna,1]);
                                   hbn=reshape(h_bn(b,f,:),[NumberofAntenna,1]);
                                   hrn=reshape(h_rn(b,f,:),[unit_num,1]);
                                   hbr=reshape(h_br(b,:,:),[unit_num,NumberofAntenna]);
                                   h = hbr'*diag(old_u)*hrn+hbn;
                                   zeta = omega_r'*hbn;
                                   if(epsilon(b,f2)==1)
                                   
                                        beta= diag(hrn')*hbr*omega_r;
                                        t=pow_abs(zeta,2)+2*real(u'*beta*zeta');
                                       if (f2==f&&epsilon_fix(b,f2)==1)
%                                             compare(f,f2)=2*real(old_u'*beta*beta'*u)-pow_abs(old_u'*beta,2);
                                            sum_up(f) = sum_up(f)+2*real((old_u'*beta)*beta'*u)-pow_abs(old_u'*beta,2)+t;
                                       else 
                                            sum_down(f)= sum_down(f)+ pow_abs(omega_r'*h,2);
                                        end
                                   end
                        end
                end
            end
         for  i=1:NumberOfFov
             a(i)=sum_up(i)-(2^(R_min/BW)-1)*sum_down(i);
             rest = rest+a(i);
         end
                

        maximize 1;
        subject to
            
                  norm(u)<=1;
              
                  %for  i=1:NumberOfFov
                  %    a(i)>=0;
                  %end
                  for  i=1:NumberOfFov
%                       (sum_up(i)/N_0)/(sum_down(i)/N_0)>=2^(R_min/BW)-1
                         (pow_p(2,R_min/BW)-1)*(sum_down(i)/N_0)<=(sum_up(i)/N_0);   
%                              pow_p(2,R_min/BW)*(sum_down(i)/N_0+1)-sum_down(i)/N_0<=1+sum_up(i)/N_0;   
                  end
                     
%                     (pow_p(2,R_min/BW)-1)*sum_down<=1e2*sum_up(i);  
            
    cvx_end

    Rates=zeros([NumberOfFov]);
    for i=1:NumberOfFov
        Rates(i)=BW*log2(1+sum_up(f)/sum_down(f) );
    end
    
    h_new = zeros([NumberOfBS,NumberOfFov,NumberofAntenna]);
    u=u/norm(u);
    for  b=1:NumberOfBS
        for  f=1:NumberOfFov
%             omega_r=reshape(omegas(b,f,:),[NumberofAntenna,1]);
            hbn=reshape(h_bn(b,f,:),[NumberofAntenna,1]);
            hrn=reshape(h_rn(b,f,:),[unit_num,1]);
            hbr=reshape(h_br(b,:,:),[unit_num,NumberofAntenna]);
            h_new(b,f,:)=(hrn'*diag(u)*hbr+hbn')';
        end
    end

%     
    result = u;
    result1 = h_new;
    result2 = a;
    status = cvx_status;


end