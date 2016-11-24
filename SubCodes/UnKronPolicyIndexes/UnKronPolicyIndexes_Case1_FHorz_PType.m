function PolicyIndexes=UnKronPolicyIndexes_Case1_FHorz_PType(PolicyIndexesKron, n_d, n_a, n_z,n_i,N_j)

%PolicyIndexesKron=zeros(2,N_a,N_z); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
%PolicyIndexesIntermediate (number_d_vars+number_a_vars,N_a,N_z)
%PolicyIndexes (number_d_vars+number_a_vars,n_a,n_z);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_i=prod(n_i);

num_a_vars=length(n_a);

if N_d==0
    PolicyIndexesIntermediate=zeros(num_a_vars,N_a,N_z,N_i,N_j);
    for i1=1:N_a
        for i2=1:N_z
            for i3=1:N_i
                for i4=1:N_j
                    optaindexKron=PolicyIndexesKron(1,i1,i2,i3,i4);
                    optA=ind2sub_homemade([n_a'],optaindexKron);
                    PolicyIndexesIntermediate(:,i1,i2,i3,i4)=[optA'];
                end
            end
        end
    end
    PolicyIndexes=reshape(PolicyIndexesIntermediate,[num_a_vars,n_a,n_z,n_i,N_j]);
    return
end

num_d_vars=length(n_d);

PolicyIndexesIntermediate=zeros(num_d_vars+num_a_vars,N_a,N_z,N_i,N_j);
for i1=1:N_a
    for i2=1:N_z
        for i3=1:N_i
            for i4=1:N_j
                optdindexKron=PolicyIndexesKron(1,i1,i2,i3,i4);
                optaindexKron=PolicyIndexesKron(2,i1,i2,i3,i4);
                optD=ind2sub_homemade(n_d',optdindexKron);
                optA=ind2sub_homemade(n_a',optaindexKron);
                PolicyIndexesIntermediate(:,i1,i2,i3,i4)=[optD';optA'];
            end
        end
    end
end
PolicyIndexes=reshape(PolicyIndexesIntermediate,[num_d_vars+num_a_vars,n_a,n_z,n_i,N_j]);


end