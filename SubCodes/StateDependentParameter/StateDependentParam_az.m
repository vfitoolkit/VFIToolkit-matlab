function Values=StateDependentParam_az(Parameters,ParamNameStr,DependenceVec,n_a,n_z,Kron,FullAZ, Parallel)
% (Using FullAZ==1) This will create a matrix ('Values') of size [n_a,n_z] (if Kron==1 will be size [N_a,N_z] from the parameter named
% 'ParamNameStr', which can then be used elsewhere. DependenceVec says on
% which actual dimensions of [n_a,n_z] the parameter that is being stored
% in Parameters (is allowed to) depend on. It is assumed to be constant
% across all other dimensions.
% When FullAZ==1 the matrix ('Values') will simply be singluar in all of the
% dimensions in which it is constant.

l_a=length(n_a);
l_z=length(n_z);

if Parallel==2
    Param_val=gpuArray(Parameters.(ParamNameStr));
else
    Param_val=gather(Parameters.(ParamNameStr));
end

% create the 'order' for permute() from DependenceVec
permuteorder=zeros(1,l_a+l_z); % l_a+l_z because we want output to be [n_a,n_z]
jj=1; kk=1;
for ii=1:length(DependenceVec)
    if DependenceVec(ii)==1
        permuteorder(jj)=ii;
        jj=jj+1;
    else % DependenceVec(ii)==0
        permuteorder(kk+sum(DependenceVec))=ii;
        kk=kk+1;
    end
end
Values=permute(Param_val,permuteorder);

if FullAZ==1
    Values=Values.*ones([n_a,n_z]);
end

if Kron==1
    if FullAZ==0
        fprintf('ERROR: Cannot have combination of Kron=1 and FullAZ=0 \n')
        dbstack
        return
    end
    Values=reshape(Values,[N_a,N_z]);
end

end