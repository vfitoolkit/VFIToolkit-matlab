function [VKron,PolicyIndexesKron]=ValueFnIter_Case1_FHorz_raw(N_d,N_a,N_z,N_j, pi_z, beta_j, FmatrixFn_j)

if N_d==0
    [VKron,PolicyIndexesKron]=ValueFnIter_Case1_FHorz_no_d_raw(N_a, N_z, N_j, pi_z, beta_j, FmatrixFn_j);
    return
end

disp('Starting Value Fn Iteration')

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);

VKron=zeros(N_a,N_z,N_j);
PolicyIndexesKron=zeros(2,N_a,N_z,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

FmatrixKron_j=reshape(FmatrixFn_j(N_j),[N_d*N_a,N_a,N_z]);
for z_c=1:N_z
    for a_c=1:N_a
        [VKron(a_c,z_c,N_j),maxindex]=max(FmatrixKron_j(:,a_c,z_c),[],1);
        PolicyIndexesKron(:,a_c,z_c,N_j)=[rem(maxindex,N_d);floor(maxindex/N_d)+1]; %[d;aprime]
    end
end

for reverse_j=1:N_j-1
    j=N_j-reverse_j;
    VKronNext_j=VKron(:,:,j+1);
    FmatrixKron_j=reshape(FmatrixFn_j(j),[N_d*N_a,N_a,N_z]);
    for z_c=1:N_z
        RHSpart2=VKronNext_j.*kron(ones(N_a,1),pi_z(z_c,:));
        RHSpart2(isnan(RHSpart2))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        RHSpart2=sum(RHSpart2,2);
        RHSpart2full=kron(RHSpart2,ones(N_d,1));
        
%         for zprime_c=1:N_z
%             if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                 RHSpart2=RHSpart2+VKronold(:,zprime_c)*pi_z(z_c,zprime_c);
%             end
%         end
%         RHSpart2full=kron(RHSpart2,ones(N_d,1));
        
        for a_c=1:N_a
            entireRHS=FmatrixKron_j(:,a_c,z_c)+beta_j(j)*RHSpart2full; %aprime by 1
            
            %calculate in order, the maximizing aprime indexes
            [VKron(a_c,z_c,j),maxindex]=max(entireRHS,[],1);
            PolicyIndexesKron(:,a_c,z_c,N_j)=[rem(maxindex,N_d);floor(maxindex/N_d)+1]; %[d;aprime]
        end
    end
end

end