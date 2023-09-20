function P = ZF(H,pow)
[K,M] = size(H);
pre = H'*inv(H*H');
P = (sqrt(pow/trace(pre*pre'))*pre);