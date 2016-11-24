function PolicyIndexes=UnKronPolicyIndexes_Case1_PType(PolicyIndexesKron, n_d, n_a, n_z,n_i)

%PolicyIndexesKron=zeros(2,N_a,N_z); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
%PolicyIndexesIntermediate (number_d_vars+number_a_vars,N_a,N_z)
%PolicyIndexes (number_d_vars+number_a_vars,n_a,n_z);

N_a=prod(n_a);
N_z=prod(n_z);
N_i=prod(n_i);

num_a_vars=length(n_a);

if length(n_d)==1 && n_d(1)==0
    PolicyIndexesIntermediate=zeros(num_a_vars,N_a,N_z,N_i);
    for a_c=1:N_a
        for z_c=1:N_z
            for i=1:N_i
                optaindexKron=PolicyIndexesKron(1,a_c,z_c,i);
                optA=ind2sub_homemade([n_a'],optaindexKron);
                PolicyIndexesIntermediate(:,a_c,z_c,i)=[optA'];
            end
        end
    end
    PolicyIndexes=reshape(PolicyIndexesIntermediate,[num_a_vars,n_a,n_z,n_i]);
    return
end

num_d_vars=length(n_d);

PolicyIndexesIntermediate=zeros(num_d_vars+num_a_vars,N_a,N_z,N_i);
for a_c=1:N_a
    for z_c=1:N_z
        for i=1:N_i
            optdindexKron=PolicyIndexesKron(1,a_c,z_c,i);
            optaindexKron=PolicyIndexesKron(2,a_c,z_c,i);
            optD=ind2sub_homemade(n_d',optdindexKron);
            optA=ind2sub_homemade(n_a',optaindexKron);
            PolicyIndexesIntermediate(:,a_c,z_c,i)=[optD';optA'];
        end
    end
end
PolicyIndexes=reshape(PolicyIndexesIntermediate,[num_d_vars+num_a_vars,n_a,n_z,n_i]);


end