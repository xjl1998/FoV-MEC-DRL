function sinr = computeSinrWithP(NumberOfUE,NumberOfBS,NumberOfChannel,G,N_0,P,v)

    sinr = zeros(NumberOfUE,NumberOfChannel);
    for n = 1:NumberOfUE
        for m = 1:NumberOfChannel
            sum_up=0;
            sum_down=0;
            for i=1:NumberOfBS
                if v(i,m,n)==1 
                    sum_up = sum_up+P(i,m)*abs(G(n,i))^2;
                elseif ismember(1,v(i,m,:))~=0
                    sum_down = sum_down+P(i,m)*abs(G(n,i))^2;
                end
            end
            sinr(n,m) = sum_up/(sum_down+N_0);
        end
    end
end