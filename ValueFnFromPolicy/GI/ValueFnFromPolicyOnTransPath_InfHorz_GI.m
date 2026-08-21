function VPath=ValueFnFromPolicyOnTransPath_InfHorz_GI(PolicyPath,V_final,ParamPath,PricePath,T,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Value fn from policy, infinite horizon transition path, with the grid interpolation layer.
% Handles both one endogenous state (GI) and two-or-more endogenous states (GI2A).
%
% This is the transition-path counterpart of ValueFnFromPolicy_InfHorz_GI: same treatment of the
% interpolated policy, but stepping backwards along the path instead of iterating to convergence.
%
% Note on GI2A: the grid interpolation layer is on the FIRST endogenous state only, the
% remaining endogenous state(s) sit on the standard grid. KronPolicyIndexes_forValueFnFromPolicy
% returns a1 and the remaining a as separate rows (it does NOT Kron them into a single index),
% so the linear index into V has to be built here as a1+n_a(1)*(arest-1).

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
l_a=length(n_a);

PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);

l_daprime=size(PolicyPath,1)-2; % -2 for the L2index and L2flag
a_gridvals=CreateGridvals(n_a,a_grid,1);

%% Switch to z_gridvals
l_z=length(n_z);
if all(size(z_grid)==[sum(n_z),1])
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
elseif all(size(z_grid)==[prod(n_z),l_z])
    z_gridvals=z_grid;
end

%%
% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
pi_z=gpuArray(pi_z);
PricePath=gpuArray(PricePath);

pi_z_howards=repelem(pi_z,N_a,1);

% Row of a1 in the Kron'd policy (comes after d, when there is a d)
if N_d==0
    index_a1=1;
else
    index_a1=2;
end

%%
VPath=zeros(N_a,N_z,T,'gpuArray');
VPath(:,:,T)=reshape(V_final,[N_a,N_z]);

for ttr=1:T-1
    tt=T-ttr; % T-1 to 1

    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(T-ttr,kk);
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,kk);
    end

    %% Calculate FofPolicy (the return fn evaluated at the Policy)
    PolicyValues=PolicyInd2Val_InfHorz(PolicyPath(:,:,:,tt),n_d,n_a,n_z,d_grid,a_grid, vfoptions);
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]

    ReturnFnParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames);
    FofPolicy=EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);

    %% Now that we have FofPolicy, calculate V.
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames));

    Policy=KronPolicyIndexes_forValueFnFromPolicy(PolicyPath(:,:,:,tt), n_d, n_a, n_z, 0, vfoptions);

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

    EVKrontemp=reshape(VPath(aprimeindex,:,tt+1),[2,N_a*N_z,N_z]); % last dimension is zprime
    EVKrontemp=shiftdim(sum(PolicyProbs.*EVKrontemp,1),1); % [N_a*N_z,N_z]

    EVKrontemp=EVKrontemp.*pi_z_howards;
    EVKrontemp(isnan(EVKrontemp))=0;
    EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);

    VPath(:,:,tt)=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

end

VPath=reshape(VPath,[n_a,n_z,T]);

end
