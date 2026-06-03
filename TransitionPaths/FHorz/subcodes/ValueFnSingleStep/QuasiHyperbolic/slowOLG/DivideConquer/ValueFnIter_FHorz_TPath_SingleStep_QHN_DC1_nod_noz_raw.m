function [V, Policy, Policyalt, Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_DC1_nod_noz_raw(V,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Naive quasi-hyperbolic: V carries Valt (exp-discounter value); Vtilde is the agent's-perspective value (beta0*beta).

N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a
Policyalt=zeros(N_a,N_j,'gpuArray'); % exponential discounter optimal choice (Valt is computed at this)
Vtilde=zeros(N_a,N_j,'gpuArray'); % agent's-perspective value (beta0*beta-discounted)

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j: terminal age has no continuation in TPath
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

%Calc the max and it's index
[Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);

V(level1ii,N_j)=shiftdim(Vtempii,1);
Policy(level1ii,N_j)=shiftdim(maxindex,1);

for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(Policy(level1ii(ii),N_j):Policy(level1ii(ii+1),N_j)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
    V(curraindex,N_j)=shiftdim(Vtempii,1);
    Policy(curraindex,N_j)=shiftdim(maxindex,1)+Policy(level1ii(ii),N_j)-1;
end
Policyalt(:,N_j)=Policy(:,N_j); % terminal: QH and exp discounter coincide
Vtilde(:,N_j)=V(:,N_j); % terminal: Vtilde coincides with Valt


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,jj); % Grab this before it is replaced/updated

    EV=VKronNext_j;

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

    %% Valt (beta)
    entireRHS_ii=ReturnMatrix_ii+beta*EV;
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
    V(level1ii,jj)=shiftdim(Vtempii,1);
    Policyalt(level1ii,jj)=shiftdim(maxindex1,1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(Policyalt(level1ii(ii),jj):Policyalt(level1ii(ii+1),jj)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii_dc+beta*EV(Policyalt(level1ii(ii),jj):Policyalt(level1ii(ii+1),jj));
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        V(curraindex,jj)=shiftdim(Vtempii,1);
        Policyalt(curraindex,jj)=shiftdim(maxindex,1)+Policyalt(level1ii(ii),jj)-1;
    end

    %% Policy (beta0*beta)
    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
    Vtilde(level1ii,jj)=shiftdim(Vtempii,1);
    Policy(level1ii,jj)=shiftdim(maxindex1,1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(Policy(level1ii(ii),jj):Policy(level1ii(ii+1),jj)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*EV(Policy(level1ii(ii),jj):Policy(level1ii(ii+1),jj));
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        Vtilde(curraindex,jj)=shiftdim(Vtempii,1);
        Policy(curraindex,jj)=shiftdim(maxindex,1)+Policy(level1ii(ii),jj)-1;
    end

end

%% Output shape for policy
Policy=shiftdim(Policy,-1);
Policyalt=shiftdim(Policyalt,-1);

end
