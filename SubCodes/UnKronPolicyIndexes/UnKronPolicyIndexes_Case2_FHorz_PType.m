function PolicyIndexes=UnKronPolicyIndexes_Case2_FHorz_PType(PolicyIndexesKron, n_d, n_a, n_z,n_i,N_j)

%PolicyIndexesKron (N_a,N_z,N_j); 
%PolicyIndexesIntermediate (num_d_vars,N_a,N_z,N_j)
%PolicyIndexes (numb_d_vars,n_a,n_z,N_j);

N_a=prod(n_a);
N_z=prod(n_z);
N_i=prod(n_i);

num_d_vars=length(n_d);

PolicyIndexesIntermediate=zeros(num_d_vars,N_a,N_z,N_i,N_j);
for i1=1:N_a
    for i2=1:N_z
        for i3=1:N_i
            for i4=1:N_j
                optdindexKron=PolicyIndexesKron(i1,i2,i3,i4);
                optD=ind2sub_homemade(n_d',optdindexKron);
                PolicyIndexesIntermediate(:,i1,i2,i3,i4)=[optD'];
            end
        end
    end
end
PolicyIndexes=reshape(PolicyIndexesIntermediate,[num_d_vars,n_a,n_z,n_i,N_j]);

end