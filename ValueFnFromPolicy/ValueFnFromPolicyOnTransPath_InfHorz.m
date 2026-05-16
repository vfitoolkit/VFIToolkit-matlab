function VPath=ValueFnFromPolicyOnTransPath_InfHorz(PolicyPath,V_final,ParamPath,PricePath,T,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)

if ~exist('vfoptions','var')
    vfoptions.gridinterplayer=0;
    % divide-and-conquer is not relevant for ValueFnFromPolicy
else
    if gpuDeviceCount==0
        error('ValueFnFromPolicyOnTransPath_InfHorz is only available on GPU')
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    end
    if isfield(vfoptions,'exoticpreferences')
        error('ValueFnFromPolicyOnTransPath_InfHorz() does not yet work with exotic preferences. Please ask on forum if you want/need this feature. \n');
    end
    if isfield(vfoptions,'experienceasset')
        if vfoptions.experienceasset==1
            error('ValueFnFromPolicyOnTransPath_InfHorz() does not yet work with experienceasset. Please ask on forum if you want/need this feature. \n');
        end
    end
    % divide-and-conquer is not relevant for ValueFnFromPolicy
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);

if N_d==0 && isscalar(n_a) && vfoptions.gridinterplayer==0
    l_daprime=1;
else
    l_daprime=size(PolicyPath,1);
    if vfoptions.gridinterplayer==1
        l_daprime=l_daprime-1;
    end
end
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

% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes)

% PricePath is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

% AgeWeightsParamNames are not actually needed as an input, but require
% them anyway to make it easier to 'copy-paste' input lists from other
% similar functions the user is likely to be using.

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
pi_z=gpuArray(pi_z);
PricePath=gpuArray(PricePath);

pi_z_howards=repelem(pi_z,N_a,1);

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

    Policy=KronPolicyIndexes_Case1(PolicyPath(:,:,:,tt), n_d, n_a, n_z, vfoptions);
    if N_d==0
        Policy_a=Policy;
    else
        Policy_a=shiftdim(ceil(Policy(2,:,:)),1);
    end

    EVKrontemp=VPath(Policy_a,:,tt+1);

    EVKrontemp=EVKrontemp.*pi_z_howards;
    EVKrontemp(isnan(EVKrontemp))=0;
    EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);

    VPath(:,:,tt)=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

end

VPath=reshape(VPath,[n_a,n_z,T]);

end
