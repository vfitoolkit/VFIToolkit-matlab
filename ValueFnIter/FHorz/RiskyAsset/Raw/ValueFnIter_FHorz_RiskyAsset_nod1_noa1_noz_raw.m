function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_nod1_noa1_noz_raw(n_d2,n_d3,n_a,n_u, N_j, d2_grid, d3_grid, a_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a=prod(n_a);
N_u=prod(n_u);

% For ReturnFn
% n_d3
% N_d3
% d3_grid
% For aprimeFn
n_d23=[n_d2,n_d3];
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_j,'gpuArray');
Policy2=zeros(2,N_a,N_j,'gpuArray'); % two: d2, d3 

%%
u_grid=gpuArray(u_grid);

d3_gridvals=CreateGridvals(n_d3,d3_grid,1);
a_gridvals=CreateGridvals(n_a,a_grid,1);

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn, n_d3, n_a, d3_gridvals, a_gridvals, ReturnFnParamsVec);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy2(1,:,N_j)=1; % d2, is meaningless anyway
    Policy2(2,:,N_j)=shiftdim(maxindex,-1); % d3
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d23, n_a, n_u, d23_grid, a_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]
    
    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EV1=aprimeProbs.*reshape(V_Jplus1(aprimeIndex),[N_d,N_u]); % (d,u), the lower aprime
    EV2=(1-aprimeProbs).*reshape(V_Jplus1(aprimeIndex+1),[N_d,N_u]); % (d,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid
    
    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,z), sum over u
    % EV is over (d,1)
    
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn, n_d3, n_a, d3_gridvals, a_gridvals, ReturnFnParamsVec);
    % (d,aprime,a)

    % Time to refine
    % First: ReturnMatrix, we can refine out d1
    % no d1 here
    % Second: EV, we can refine out d2
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3,1]),[],1);
    % Now put together entireRHS, which just depends on d3
    entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV_onlyd3,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,N_j)=Vtemp;
    Policy2(2,:,N_j)=shiftdim(maxindex,1); % d3
    Policy2(1,:,N_j)=shiftdim(d2index(maxindex),1); % d2
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
    [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d23, n_a, n_u, d23_grid, a_grid, u_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d*N_u,1], whereas aprimeProbs is [N_d,N_u]

    VKronNext_j=V(:,jj+1);
    
    % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
    EV1=aprimeProbs.*reshape(VKronNext_j(aprimeIndex),[N_d,N_u]); % (d,u), the lower aprime
    EV2=(1-aprimeProbs).*reshape(VKronNext_j(aprimeIndex+1),[N_d,N_u]); % (d,u), the upper aprime
    % Already applied the probabilities from interpolating onto grid
    
    % Expectation over u (using pi_u), and then add the lower and upper
    EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d,1,z), sum over u
    % EV is over (d,1)
    
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn, n_d3, n_a, d3_gridvals, a_gridvals, ReturnFnParamsVec);
    % (d,aprime,a)
    
    % Time to refine
    % First: ReturnMatrix, we can refine out d1
    % no d1 here
    % Second: EV, we can refine out d2
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3,1]),[],1);
    % Now put together entireRHS, which just depends on d3
    entireRHS=ReturnMatrix+DiscountFactorParamsVec*shiftdim(EV_onlyd3,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,jj)=Vtemp;
    Policy2(2,:,N_j)=shiftdim(maxindex,1); % d3
    Policy2(1,:,N_j)=shiftdim(d2index(maxindex),1); % d2
end

Policy=Policy2(1,:,:)+N_d2*(Policy2(2,:,:)-1); % two: d2, d3


end
