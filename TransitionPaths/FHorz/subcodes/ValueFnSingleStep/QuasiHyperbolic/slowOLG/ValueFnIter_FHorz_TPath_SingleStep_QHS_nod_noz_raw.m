function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z
Vhat=zeros(N_a,N_j,'gpuArray'); % agent's-perspective value at QH-optimal policy under beta0beta

%% j=N_j

% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);
%Calc the max and it's index
[Vtemp,maxindex]=max(ReturnMatrix,[],1);
V(:,N_j)=Vtemp;
Policy(:,N_j)=maxindex;
Vhat(:,N_j)=V(:,N_j); % terminal: no continuation, Vhat=V


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    VKronNext_j=Vtemp_j; % Has been presaved before it was
    Vtemp_j=V(:,jj); % Grab this before it is replaced/updated

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);

    % First Policy
    entireRHS=ReturnMatrix+beta0beta*VKronNext_j*ones(1,N_a,1); % Use the today-to-tomorrow discount factor
    [Vtilde_tmp,maxindex]=max(entireRHS,[],1);
    Vhat(:,jj)=Vtilde_tmp;
    Policy(:,jj)=maxindex;
    % Now Vunderbar
    entireRHS=ReturnMatrix+beta*VKronNext_j*ones(1,N_a,1); % Use the two-future-periods discount factor
    maxindexfull=maxindex+N_a*(0:1:N_a-1);
    V(:,jj)=entireRHS(maxindexfull);
end

%%
Policy=shiftdim(Policy,-1);

end
