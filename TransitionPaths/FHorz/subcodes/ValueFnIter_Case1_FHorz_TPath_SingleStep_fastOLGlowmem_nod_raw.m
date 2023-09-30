function [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLGlowmem_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames)

N_a=prod(n_a);
N_z=prod(n_z);
l_z=length(n_z);

Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_d*N_aprime by N_a*N_z*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

% z_gridvals_J=permute(z_gridvals_J,[3,2,1]); % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(): N_j-by-l_z-by-N_z
% pi_z_J=permute(pi_z_J,[3,2,1]); % Give it the size best for the loop below, namely (j,z',z)


DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=DiscountFactorParamsVec';

VKronNext=zeros(N_a,N_j,N_z,'gpuArray');
VKronNext(:,1:N_j-1,:)=permute(V(:,:,2:end),[1 3 2]); % Swap j and z
VKronNext=reshape(VKronNext,[N_a*N_j,N_z]);

for z_c=1:N_z
    z_vals_AllAges=z_gridvals_J(:,:,z_c);
    ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(ReturnFn, 0, n_a, ones(l_z,1), N_j, [], a_grid, z_vals_AllAges, ReturnFnParamsAgeMatrix);
    % fastOLG: ReturnMatrix is [aprime,a,j,z]

    ReturnMatrix_z=reshape(ReturnMatrix_z,[N_a,N_a*N_j]);

    %Calc the condl expectation term (except beta), which depends on z but not on control variables
%     EV_z=VKronNext.*(ones(N_a*N_j,1,'gpuArray')*pi_z(z_c,:));
    EV_z=VKronNext.*kron(pi_z_J(jj,:,z_c),ones(N_a,1,'gpuArray'));
    EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV_z=sum(EV_z,2);

    discountedEV_z=DiscountFactorParamsVec.*reshape(EV_z,[N_a,N_j]); % (aprime)-by-(j) % Note: jprime and j are essentially the same thing.

    entirediscountedEV_z=kron(discountedEV_z,ones(1,N_a));
    
    entirediscountedEV_z=ReturnMatrix_z+entirediscountedEV_z; %(aprime)-by-(a,j)
    
    %Calc the max and it's index
    [Vtemp,maxindex]=max(entirediscountedEV_z,[],1);
    V(:,z_c,:)=reshape(Vtemp,[N_a,1,N_j]); % V is over (a,z,j)
    Policy(:,z_c,:)=reshape(maxindex,[N_a,1,N_j]); % Policy is over (a,z,j)

end

end