function V=ValueFnFromPolicy_InfHorz_GI(Policy,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Value fn from policy, infinite horizon, with the grid interpolation layer.
% Handles both one endogenous state (GI) and two-or-more endogenous states (GI2A).
%
% Note on GI2A: the grid interpolation layer is on the FIRST endogenous state only, the
% remaining endogenous state(s) sit on the standard grid. KronPolicyIndexes_forValueFnFromPolicy
% returns a1 and the remaining a as separate rows (it does NOT Kron them into a single index),
% so the linear index into V has to be built here as a1+n_a(1)*(arest-1).
% [This mirrors ValueFnFromPolicy_FHorz_GI, which does the same via lower_lin=alower+n_a1*(a2prime-1)]

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
l_a=length(n_a);

l_daprime=size(Policy,1)-2; % -2 for the L2index and L2flag

a_gridvals=CreateGridvals(n_a,a_grid,1);
% Switch to z_gridvals
[z_gridvals, pi_z, vfoptions]=ExogShockSetup_InfHorz(n_z,z_grid,pi_z,Parameters,vfoptions,3);

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

%% Calculate FofPolicy (the return fn evaluated at the Policy)
PolicyValues=PolicyInd2Val_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid, vfoptions);
if N_z==0
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a]),[2,1]); %[N_a,l_d+l_a]
else
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]
end

ReturnFnParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames);
FofPolicy=EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);

%% Now that we have FofPolicy, calculate V.
DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames));

currdist=Inf;
itercount=1;
VKron=FofPolicy/(1-DiscountFactorParamsVec); % rough guess

% Row of a1 in the Kron'd policy (comes after d, when there is a d)
if N_d==0
    index_a1=1;
else
    index_a1=2;
end

if N_z==0
    Policy=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, 1, 0, vfoptions);

    alowerindex=reshape(ceil(Policy(index_a1,:,:)),[1,N_a]); % a1 lower grid point, 1 to n_a(1)-1
    if l_a>=2
        % GI2A: add the offset for the remaining endogenous state(s), which Kron returns separately.
        % Interpolation is on a1 only, so the upper point (alowerindex+1) steps one point in a1.
        alowerindex=alowerindex+n_a(1)*(reshape(ceil(Policy(index_a1+1,:,:)),[1,N_a])-1);
    end
    aprimeindex=[alowerindex; alowerindex+1]; % [2,N_a]
    PolicyProbs=reshape(ceil(Policy(end,:,:)),[1,N_a]); % L2 (Kron drops L2flag)
    PolicyProbs=(PolicyProbs-1)/(vfoptions.ngridinterp+1); % prob of upper point
    PolicyProbs=[1-PolicyProbs; PolicyProbs]; % [2,N_a]

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        EVKrontemp=reshape(VKron(aprimeindex,:),[2,N_a]);
        EVKrontemp=PolicyProbs.*EVKrontemp;
        % A zero interpolation weight on a grid point where V is -Inf gives 0*(-Inf)=NaN, which then
        % contaminates the sum. The value fn iteration has the same guard on its own expectation, for
        % the same reason: a zero weight is meant to contribute nothing. Note the value fn iteration
        % weights by pi_z and guards before it interpolates, whereas here the interpolation comes
        % first, so the guard has to be at the product rather than after the pi_z multiply.
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=shiftdim(sum(EVKrontemp,1),1);

        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(max(abs(VKron-VKronold)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,1]);
else % N_z>0
    Policy=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, 0, vfoptions);

    pi_z_howards=repelem(pi_z,N_a,1);

    alowerindex=reshape(ceil(Policy(index_a1,:,:)),[1,N_a*N_z]); % a1 lower grid point, 1 to n_a(1)-1
    if l_a>=2
        % GI2A: add the offset for the remaining endogenous state(s), which Kron returns separately.
        % Interpolation is on a1 only, so the upper point (alowerindex+1) steps one point in a1.
        alowerindex=alowerindex+n_a(1)*(reshape(ceil(Policy(index_a1+1,:,:)),[1,N_a*N_z])-1);
    end
    aprimeindex=[alowerindex; alowerindex+1]; % [2,N_a*N_z]
    PolicyProbs=reshape(ceil(Policy(end,:,:)),[1,N_a*N_z]); % L2 (Kron drops L2flag)
    PolicyProbs=(PolicyProbs-1)/(vfoptions.ngridinterp+1); % prob of upper point
    PolicyProbs=[1-PolicyProbs; PolicyProbs]; % [2,N_a*N_z]

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        EVKrontemp=reshape(VKron(aprimeindex,:),[2,N_a*N_z,N_z]); % last dimension is zprime
        EVKrontemp=PolicyProbs.*EVKrontemp;
        % A zero interpolation weight on a grid point where V is -Inf gives 0*(-Inf)=NaN, which then
        % contaminates the sum. The value fn iteration has the same guard on its own expectation, for
        % the same reason: a zero weight is meant to contribute nothing. Note the value fn iteration
        % weights by pi_z and guards before it interpolates, whereas here the interpolation comes
        % first, so the guard has to be at the product rather than after the pi_z multiply.
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=shiftdim(sum(EVKrontemp,1),1); % [N_a*N_z,N_z]

        EVKrontemp=EVKrontemp.*pi_z_howards;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(max(abs(VKron-VKronold)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,n_z]);
end

end
