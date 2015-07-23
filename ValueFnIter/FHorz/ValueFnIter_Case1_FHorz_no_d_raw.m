function [VKron, PolicyIndexesKron]=ValueFnIter_Case1_FHorz_no_d_raw(N_a,N_z,N_j,pi_z,beta_j,FmatrixFn_j)

%disp('Starting Value Fn Iteration')
VKron=zeros(N_a,N_z,N_j);
PolicyIndexesKron=zeros(1,N_a,N_z,N_j); %first dim indexes the optimal choice for aprime rest of dimensions a,z

FmatrixKron_j=reshape(FmatrixFn_j(N_j),[N_a,N_a,N_z]);
for z_c=1:N_z
    for a_c=1:N_a
        [VKron(a_c,z_c,N_j),PolicyIndexesKron(1,a_c,z_c,N_j)]=max(FmatrixKron_j(:,a_c,z_c),[],1);
    end
end

for reverse_j=1:N_j-1
    j=N_j-reverse_j;
    VKronNext_j=VKron(:,:,j+1);
    FmatrixKron_j=reshape(FmatrixFn_j(j),[N_a,N_a,N_z]);
    for z_c=1:N_z
        RHSpart2=VKronNext_j.*kron(ones(N_a,1),pi_z(z_c,:));
        RHSpart2(isnan(RHSpart2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        RHSpart2=sum(RHSpart2,2);
        for a_c=1:N_a
            entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(j)*RHSpart2; %aprime by 1
            
            %calculate in order, the maximizing aprime indexes
            [VKron(a_c,z_c,j),PolicyIndexesKron(1,a_c,z_c,j)]=max(entireRHS,[],1);
        end
    end
end

end