function Policy=UnKronPolicyIndexes_Case1_FHorz_semiz(Policy3, n_d1, n_a, n_z,N_j,vfoptions)
% n_z should be [n_z,n_semiz]

%Input: Policy3=zeros(3,N_a,N_z,N_j); %first dim indexes the optimal choice for d2 and (d1,aprime) rest of dimensions a,z 
%Output: Policy (l_d+l_a,n_a,n_z,N_j);

% l_d=l_d1+l_d2
% l_d2=1 is hardcoded

N_d1=prod(n_d1);
N_a=prod(n_a);
N_z=prod(n_z);

l_d1=length(n_d1); % Just d1
l_a=length(n_a);

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if ~isfield(vfoptions,'policy_forceintegertype')
    vfoptions.policy_forceintegertype=0;
end
if vfoptions.policy_forceintegertype==1
    Policy3=round(Policy3);
end

PolicyOut=zeros(l_d1+1+l_a,N_a,N_z,N_j);

if N_d1==0
    error('Semi-exogenous state is not supported without another decision variable')
end

if vfoptions.parallel~=2
    PolicyTemp=zeros(l_d1+l_a,N_a,N_z,N_j);
    for a_c=1:N_a
        for z_c=1:N_z
            for jj=1:N_j
                optdindexKron=Policy3(1,a_c,z_c,jj);
                optaindexKron=Policy3(3,a_c,z_c,jj);
                optD=ind2sub_homemade(n_d1',optdindexKron);
                optA=ind2sub_homemade(n_a',optaindexKron);
                PolicyTemp(:,a_c,z_c,jj)=[optD';optA'];
            end
        end
    end
    PolicyOut(1:l_d1,:,:,:)=PolicyTemp(1:l_d1,:,:,:); % d1
    PolicyOut(l_d1+1,:,:,:)=Policy3(2,:,:,:); % d2
    PolicyOut(l_d1+2:end,:,:,:)=PolicyTemp(l_d1+1:end,:,:,:); % aprime
    Policy=reshape(PolicyOut,[l_d1+l_a+1,n_a,n_z,N_j]);
else
    l_da=length(n_d1)+length(n_a);
    n_da=[n_d1,n_a];
    PolicyTemp=zeros(l_da,N_a,N_z,N_j,'gpuArray');

    for jj=1:N_j
        PolicyTemp(1,:,:,jj)=rem(Policy3(1,:,:,jj)-1,n_da(1))+1;
        if l_d1>1
            if l_d1>2
                for ii=2:l_d1-1
                    PolicyTemp(ii,:,:,jj)=rem(ceil(Policy3(1,:,:,jj)/prod(n_d1(1:ii-1)))-1,n_d1(ii))+1;
                end
            end
            PolicyTemp(l_d1,:,:,jj)=ceil(Policy3(1,:,:,jj)/prod(n_d1(1:l_d1-1)));
        end

        PolicyTemp(l_d1+1,:,:,jj)=rem(Policy3(3,:,:,jj)-1,n_a(1))+1;
        if l_a>1
            if l_a>2
                for ii=2:l_a-1
                    PolicyTemp(l_d1+ii,:,:,jj)=rem(ceil(Policy3(3,:,:,jj)/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
                end
            end
            PolicyTemp(l_da,:,:,jj)=ceil(Policy3(3,:,:,jj)/prod(n_a(1:l_a-1)));
        end
    end
    PolicyOut(1:l_d1,:,:,:)=PolicyTemp(1:l_d1,:,:,:);
    PolicyOut(l_d1+1,:,:,:)=Policy3(2,:,:,:);
    PolicyOut(l_d1+2:end,:,:,:)=PolicyTemp(l_d1+1:end,:,:,:);
    Policy=reshape(PolicyOut,[l_da+1,n_a,n_z,N_j]);

end


if vfoptions.policy_forceintegertype==1
    Policy=round(Policy);
end


end
