function [V,Policy]=ValueFnIter_Case1_FHorz_ExpAssetu_noa1_noz_raw(n_d1,n_d2,n_a2,n_u, N_j, d1_grid, d2_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
d1_grid=gpuArray(d1_grid);
d2_grid=gpuArray(d2_grid);
a2_grid=gpuArray(a2_grid);

pi_u=shiftdim(pi_u,-2); % put it into third dimension

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn,[n_d1,n_d2], n_a2, [d1_grid; d2_grid], a2_grid, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2*N_u,1], whereas aprimeProbs is [N_d2,N_a2,N_u]
    
    % Switch EV from being in terms of a2prime to being in terms of d2 and a2 (in expectation because of the u shocks)
    EV1=a2primeProbs.*reshape(vfoptions.V_Jplus1(a2primeIndex),[N_d2,N_a2,N_u]); % (d2,a2,u), the lower aprime
    EV2=(1-a2primeProbs).*reshape(vfoptions.V_Jplus1(a2primeIndex+1),[N_d2,N_a2,N_u]); % (d2,a2,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid

    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((EV1.*pi_u),3)+sum((EV2.*pi_u),3); % (d2,a2), sum over u
    % EV is over (d2,a2)
    
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn,[n_d1,n_d2], n_a2, [d1_grid; d2_grid], a2_grid, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
    % (d,a)

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,N_j)=shiftdim(Vtemp,1);
    Policy(:,N_j)=shiftdim(maxindex,1);

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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix_Case1(aprimeFn, n_d2, n_a2, n_u, d2_grid, a2_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)

    % Note: aprimeIndex is [N_d2*N_a2*N_u,1], whereas aprimeProbs is [N_d2,N_a2,N_u]
    
    % Switch EV from being in terms of a2prime to being in terms of d2 and a2 (in expectation because of the u shocks)
    EV1=a2primeProbs.*reshape(V(a2primeIndex,jj+1),[N_d2,N_a2,N_u]); % (d2,a2,u), the lower aprime
    EV2=(1-a2primeProbs).*reshape(V(a2primeIndex+1,jj+1),[N_d2,N_a2,N_u]); % (d2,a2,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid

    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((EV1.*pi_u),3)+sum((EV2.*pi_u),3); % (d2,a2), sum over u
    % EV is over (d2,a2)
    
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn,[n_d1,n_d2], n_a2, [d1_grid; d2_grid], a2_grid, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
    % (d,a)

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,N_d1,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,jj)=shiftdim(Vtemp,1);
    Policy(:,jj)=shiftdim(maxindex,1);

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron
% Policy2=zeros(2,N_a,N_z,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
% Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d2)+1,-1);
% Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d2),-1);

end