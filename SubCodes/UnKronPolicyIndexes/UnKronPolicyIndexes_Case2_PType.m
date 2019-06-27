function PolicyIndexes=UnKronPolicyIndexes_Case2_PType(PolicyIndexesKron, n_d, n_a, n_z,n_i)

%PolicyIndexesKron=zeros(N_a,N_z); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
%PolicyIndexesIntermediate (number_d_vars,N_a,N_z)
%PolicyIndexes (number_d_vars,n_a,n_z);

N_a=prod(n_a);
N_z=prod(n_z);
N_i=prod(n_i);

num_d_vars=length(n_d);

PolicyIndexesIntermediate=zeros(num_d_vars,N_a,N_z,N_i);
for a_c=1:N_a
    for z_c=1:N_z
        for i=1:N_i
            optdindexKron=PolicyIndexesKron(a_c,z_c,i);
            optD=ind2sub_homemade(n_d',optdindexKron);
            PolicyIndexesIntermediate(:,a_c,z_c,i)=[optD'];
        end
    end
end
PolicyIndexes=reshape(PolicyIndexesIntermediate,[num_d_vars,n_a,n_z,n_i]);

end