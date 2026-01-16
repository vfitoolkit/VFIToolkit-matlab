function Policy=UnKronPolicyIndexes_InfHorz_CPU(PolicyKron, n_d,n_a, n_z)
% Can use vfoptions OR simoptions
% Input: PolicyKron is (2,N_a,N_z) first dim indexes the optimal choice for d and aprime
%                      (N_a,N_z) if there is no d
% Output: Policy is (l_d+l_a,n_a,n_z);

N_a=prod(n_a);
N_z=prod(n_z);

% On CPU
if n_d(1)==0
    Policy=zeros(l_aprime,N_a,N_z);
    for i=1:N_a
        for j=1:N_z
            optaindexKron=PolicyKron(i,j);
            optA=ind2sub_homemade(n_aprime',optaindexKron);
            Policy(:,i,j)=optA';
        end
    end
    Policy=reshape(Policy,[l_aprime,n_a,n_z]);
else
    l_d=length(n_d);
    Policy=zeros(l_d+l_aprime,N_a,N_z);
    for i=1:N_a
        for j=1:N_z
            optdindexKron=PolicyKron(1,i,j);
            optaindexKron=PolicyKron(2,i,j);
            optD=ind2sub_homemade(n_d',optdindexKron);
            optA=ind2sub_homemade(n_aprime',optaindexKron);
            Policy(:,i,j)=[optD';optA'];
        end
    end
    Policy=reshape(Policy,[l_d+l_aprime,n_a,n_z]);
end

end