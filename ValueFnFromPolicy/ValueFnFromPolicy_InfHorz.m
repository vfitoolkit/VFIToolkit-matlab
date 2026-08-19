function V=ValueFnFromPolicy_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)

if ~exist('vfoptions','var')
    vfoptions.gridinterplayer=0;
    vfoptions.tolerance=10^(-9);
    vfoptions.maxiter=10^4; % Can be used to stop the VFI after a finite number of iterations
    % divide-and-conquer is not relevant for ValueFnFromPolicy
else
    if gpuDeviceCount==0
        error('ValueFnFromPolicy_InfHorz is only available on GPU')
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    end
    if ~isfield(vfoptions,'tolerance')
        vfoptions.tolerance=10^(-9);
    end
    if ~isfield(vfoptions,'maxiter')
        vfoptions.maxiter=10^4; % Can be used to stop the VFI after a finite number of iterations
    end
    if isfield(vfoptions,'exoticpreferences')
        error('ValueFnFromPolicy_InfHorz() does not yet work with exotic preferences. Please ask on forum if you want/need this feature. \n');
    end
    % divide-and-conquer is not relevant for ValueFnFromPolicy
end

%% Grid interpolation layer is handled by its own command
if vfoptions.gridinterplayer==1
    V=ValueFnFromPolicy_InfHorz_GI(Policy,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% Note: gridinterplayer=1 has already been sent to ValueFnFromPolicy_InfHorz_GI above
if N_d==0 && isscalar(n_a)
    l_daprime=1;
else
    l_daprime=size(Policy,1);
end
a_gridvals=CreateGridvals(n_a,a_grid,1);
% Switch to z_gridvals
[z_gridvals, pi_z, vfoptions]=ExogShockSetup_InfHorz(n_z,z_grid,pi_z,Parameters,vfoptions,3);

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);
% Basic setup: the first inputs of ReturnFn will be (d,aprime,a,z,..) and everything after this is a parameter, so we get the names of all these parameters.
% But this changes if you have e, semiz, or just multiple d, and if you use riskyasset, expasset, etc.
% So figure out which setup we have, and get the relevant ReturnFnParamNames

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
if N_z==0
    Policy=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, 1, 0, vfoptions);

    if N_d==0
        Policy_a=shiftdim(Policy(1,:),1);
    else
        Policy_a=shiftdim(ceil(Policy(2,:)),1);
    end

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        EVKrontemp=VKron(Policy_a,:);

        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp;

        currdist=max(max(abs(VKron-VKronold)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,1]);
else % N_z>0
    Policy=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, 0, vfoptions);

    pi_z_howards=repelem(pi_z,N_a,1);

    if N_d==0
        Policy_a=shiftdim(Policy(1,:,:),1);
    else
        Policy_a=shiftdim(ceil(Policy(2,:,:)),1);
    end

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        EVKrontemp=VKron(Policy_a,:);

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
