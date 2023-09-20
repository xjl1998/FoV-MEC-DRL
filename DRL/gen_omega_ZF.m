function [omegas]= gen_omega_ZF(NumberOfFov,NumberOfBS,NumberofAntenna,G,power)%定义命名函数
omegas = zeros([NumberOfBS,NumberOfFov,NumberofAntenna])
for b=1:NumberOfBS
    H = reshape(G(b,:,:),[NumberOfFov,NumberofAntenna]);
    PZF = ZF(H,power);
    omegas(b,:,:) = PZF';
end