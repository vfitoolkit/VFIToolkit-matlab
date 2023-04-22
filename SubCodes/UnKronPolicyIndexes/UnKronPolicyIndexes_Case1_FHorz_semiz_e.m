function PolicyOut=UnKronPolicyIndexes_Case1_FHorz_semiz_e(Policy3, n_d1, n_a, n_z,n_e,N_j,vfoptions)
% n_z should be [n_z,n_semiz]

%Input: Policy3=zeros(3,N_a,N_z,N_e,N_j); %first dim indexes the optimal choice for d2 and (d1,aprime) rest of dimensions a,z 
%Output: Policy (l_d+l_a,n_a,n_z,n_e,N_j);

% l_d=l_d1+l_d2
% l_d2=1 is hardcoded

N_d1=prod(n_d1);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

l_d1=length(n_d1); % Just d1
l_a=length(n_a);

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    Policy3=round(Policy3);
end

if N_d1==0
    error('Semi-exogenous state is not supported without another decision variable')
end


if vfoptions.parallel~=2
    PolicyOut=zeros(l_d1+l_a+1,N_a,N_z,N_e,N_j);
    for a_c=1:N_a
        for z_c=1:N_z
            for e_c=1:N_e
                for jj=1:N_j
                    optdindexKron=Policy3(1,a_c,z_c,e_c,jj);
                    optaindexKron=Policy3(3,a_c,z_c,e_c,jj);
                    optD=ind2sub_homemade(n_d',optdindexKron);
                    optA=ind2sub_homemade(n_a',optaindexKron);
                    PolicyOut(:,a_c,z_c,e_c,jj)=[optD';Policy3(2,a_c,z_c,e_c,jj);optA'];
                end
            end
        end
    end
    PolicyOut=reshape(PolicyOut,[l_d+l_a+1,n_a,n_z,n_e,N_j]);
else
    PolicyOut=zeros(l_d+1+l_a,N_a,N_z,N_e,N_j,'gpuArray');

    for jj=1:N_j
        PolicyOut(1,:,:,:,jj)=rem(Policy3(1,:,:,:,jj)-1,n_d1(1))+1;
        if l_d>1
            if l_d>2
                for ii=2:l_d-1
                    PolicyOut(ii,:,:,:,jj)=rem(ceil(Policy3(1,:,:,:,jj)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
                end
            end
            PolicyOut(l_d,:,:,:,jj)=ceil(Policy3(1,:,:,:,jj)/prod(n_d(1:l_d-1)));
        end

        PolicyOut(l_d+1,:,:,:,jj)=Policy3(2,:,:,:,jj);

        PolicyOut(l_d+1+1,:,:,:,jj)=rem(Policy3(3,:,:,:,jj)-1,n_a(1))+1;
        if l_a>1
            if l_a>2
                for ii=2:l_a-1
                    PolicyOut(l_d+1+ii,:,:,:,jj)=rem(ceil(Policy3(3,:,:,:,jj)/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
                end
            end
            PolicyOut(l_d+1+l_a,:,:,:,jj)=ceil(Policy3(3,:,:,:,jj)/prod(n_a(1:l_a-1)));
        end
    end

    PolicyOut=reshape(PolicyOut,[l_d+1+l_a,n_a,n_z,n_e,N_j]);
end

if vfoptions.policy_forceintegertype==1
    PolicyOut=round(PolicyOut);
end

end