function Policy=UnKronPolicyIndexes_FHorz_CPU(PolicyKron, n_d,n_a, n_z,N_j)
% Input: PolicyKron is (2,N_a,N_z,N_j) first dim indexes the optimal choice for d and aprime
%                      (N_a,N_z,N_j) if there is no d
% Output: Policy is (l_d+l_a,n_a,n_z,N_j);

N_a=prod(n_a);
N_z=prod(n_z);

l_aprime=length(n_a);
n_aprime=n_a;

% On CPU
Policy=zeros(l_aprime,N_a,N_z,N_j);
if n_d(1)==0
    for a_c=1:N_a
        for z_c=1:N_z
            for jj=1:N_j
                optaindexKron=PolicyKron(a_c,z_c,jj);
                optA=ind2sub_homemade(n_aprime',optaindexKron);
                Policy(:,a_c,z_c,jj)=optA';
            end
        end
    end
    Policy=reshape(Policy,[l_aprime,n_a,n_z,N_j]);
else
    l_d=length(n_d);
    Policy=zeros(l_d+l_aprime,N_a,N_z,N_j);
    for a_c=1:N_a
        for z_c=1:N_z
            for jj=1:N_j
                optdindexKron=PolicyKron(1,a_c,z_c,jj);
                optaindexKron=PolicyKron(2,a_c,z_c,jj);
                optD=ind2sub_homemade(n_d',optdindexKron);
                optA=ind2sub_homemade(n_aprime',optaindexKron);
                Policy(:,a_c,z_c,jj)=[optD';optA'];
            end
        end
    end
    Policy=reshape(Policy,[l_d+l_aprime,n_a,n_z,N_j]);
end


end