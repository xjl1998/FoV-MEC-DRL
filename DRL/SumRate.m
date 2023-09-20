function C = SumRate(H,P,sigma2)
[K,M] = size(H);
c = zeros(1,K);
for idx1 = 1:1:K
    p1 = reshape(P(idx1,:),[K,1]);
    ds = abs(H(idx1,:)*p1)^2;
    int = 0;
    for idx2 = 1:1:K
        if idx2 ~= idx1
            p2=reshape(P(idx2,:),[K,1]);
            int = int + abs(H(idx1,:)*p2)^2;
        end
    end
    sinr_k = ds/(sigma2+int);
    c(idx1) = log2(1+sinr_k);
end
C = sum(c);