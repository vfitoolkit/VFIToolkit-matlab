function [V,Policy]=ValueFnIter_Case1_FHorz_ExpAsset_noa1_raw(n_d1,n_d2,n_a2,n_z,N_j, d1_grid, d2_grid, a2_grid,z_gridvals_J,pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a2_grid=gpuArray(a2_grid);

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, n_z, [d1_grid; d2_grid], a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, n_z, [d1_grid; d2_grid], a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
    % (d,a)

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]
    
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);
    % Using V_Jplus1
    EV1=V_Jplus1(a2primeIndex,:).*shiftdim(pi_z_J(:,:,N_j)',-1);    % (d2,a2,zprime), the lower aprime
    EV1(isnan(EV1))=0; % where V=-Inf but pi_z is zero
    EV1=sum(EV1,2);
    EV2=V_Jplus1(a2primeIndex+1,:).*shiftdim(pi_z_J(:,:,N_j)',-1);    % (d2,a2,zprime), the upper aprime
    EV2(isnan(EV2))=0; % where V=-Inf but pi_z is zero
    EV2=sum(EV2,2);

    % Apply the aprimeProbs
    entireEV=reshape(EV1,[N_d2,N_a2]).*a2primeProbs+reshape(EV2,[N_d2,N_a2]).*(1-a2primeProbs); % probability of lower grid point+ probability of upper grid point
    % entireEV is (d,a2)

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(entireEV,N_d1,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,:,N_j)=shiftdim(Vtemp,1);
    Policy(:,:,N_j)=shiftdim(maxindex,1);

end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]
    a2primeProbs=repelem(a2primeProbs,1,1,N_z);

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,[n_d1,n_d2], n_a2, n_z, [d1_grid; d2_grid], a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
    % (d,a)

    EV1=V(a2primeIndex,:,jj+1).*shiftdim(pi_z_J(:,:,jj)',-1);    % (d2,a2,zprime), the lower aprime
    EV1(isnan(EV1))=0; % where V=-Inf but pi_z is zero
    EV1=sum(EV1,2);
    EV2=V(a2primeIndex+1,:,jj+1).*shiftdim(pi_z_J(:,:,jj)',-1);    % (d2,a2,zprime), the upper aprime
    EV2(isnan(EV2))=0; % where V=-Inf but pi_z is zero
    EV2=sum(EV2,2);

    % Apply the aprimeProbs
    entireEV=reshape(EV1,[N_d2,N_a2,N_z]).*a2primeProbs+reshape(EV2,[N_d2,N_a2,N_z]).*(1-a2primeProbs); % probability of lower grid point+ probability of upper grid point
    % entireEV is (d, a2)

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(entireEV,N_d1,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,:,jj)=shiftdim(Vtemp,1);
    Policy(:,:,jj)=shiftdim(maxindex,1);

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron
% Policy2=zeros(2,N_a,N_z,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
% Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d2)+1,-1);
% Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d2),-1);

end