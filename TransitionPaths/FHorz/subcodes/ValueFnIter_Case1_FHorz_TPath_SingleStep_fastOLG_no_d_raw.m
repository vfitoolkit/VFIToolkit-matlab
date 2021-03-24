function [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_no_d_raw(V,n_a,n_z,N_j, a_grid, z_grid,pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%%

eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')

%% First, create the big 'next period (of transition path) expected value fn.

% VfastOLG will be N_d*N_aprime by N_a*N_z*N_j (note: N_aprime is just equal to N_a)

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

z_grid_AllAges=z_grid;
pi_z_AllAges=pi_z;
if fieldexists_ExogShockFn==1
    z_grid_AllAges=z_grid.*ones(1,N_j);
    pi_z_AllAges=pi_z.*ones(1,1,N_j);
    for jj=1:N_j
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsVec);
            z_grid_AllAges(:,jj)=gpuArray(z_grid); pi_z_AllAges(:,:,jj)=gpuArray(pi_z);
        else
            [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
            z_grid_AllAges(:,jj)=gpuArray(z_grid); pi_z_AllAges(:,:,jj)=gpuArray(pi_z);
        end
    end
    temp=1:1:(l_a+l_a+l_z+1);
    temp(2)=l_a+l_a+l_z+1; temp(end)=2;
    z_grid_AllAges=permute(z_grid_AllAges,temp); % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG()
end

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(ReturnFn, 0, n_a, n_z, N_j, [], a_grid, z_grid_AllAges, ReturnFnParamsAgeMatrix);

% ReturnMatrix=permute(ReturnMatrix,[1 2 4 3]); % Swap j and z (so that z
% is last) % Modified CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG to
% eliminate this step.
ReturnMatrix=reshape(ReturnMatrix,[N_a,N_a*N_j,N_z]);

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=DiscountFactorParamsVec';
% DiscountFactorParamsVec=kron(ones(N_a,1),DiscountFactorParamsVec);

VKronNext=zeros(N_a,N_j,N_z,'gpuArray');
VKronNext(:,1:N_j-1,:)=permute(V(:,:,2:end),[1 3 2]); % Swap j and z
VKronNext=reshape(VKronNext,[N_a*N_j,N_z]);

for z_c=1:N_z
    ReturnMatrix_z=ReturnMatrix(:,:,z_c);

    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV_z=VKronNext.*(ones(N_a*N_j,1,'gpuArray')*pi_z(z_c,:));
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