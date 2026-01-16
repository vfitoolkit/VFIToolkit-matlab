function Policy=UnKronPolicyIndexes_FHorz_noz_CPU(PolicyKron, n_d,n_a, N_j)
% Can use vfoptions OR simoptions
% Input: PolicyKron=zeros(2,N_a,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_j) if there is no d
% Output: Policy (l_d+l_a,n_a,N_j);

N_a=prod(n_a);

% On CPU
if n_d(1)==0
    Policy=zeros(l_aprime,N_a,N_j);
    for a_c=1:N_a
        for jj=1:N_j
            optaindexKron=PolicyKron(a_c,jj);
            optA=ind2sub_homemade(n_aprime',optaindexKron);
            Policy(:,a_c,jj)=optA';
        end
    end
    Policy=reshape(Policy,[l_aprime,n_a,N_j]);
else
    l_d=length(n_d);
    Policy=zeros(l_d+l_aprime,N_a,N_j);
    for a_c=1:N_a
        for jj=1:N_j
            optdindexKron=PolicyKron(1,a_c,jj);
            optaindexKron=PolicyKron(2,a_c,jj);
            optD=ind2sub_homemade(n_d',optdindexKron);
            optA=ind2sub_homemade(n_aprime',optaindexKron);
            Policy(:,a_c,jj)=[optD';optA'];
        end
    end
    Policy=reshape(Policy,[l_d+l_aprime,n_a,N_j]);
end


end