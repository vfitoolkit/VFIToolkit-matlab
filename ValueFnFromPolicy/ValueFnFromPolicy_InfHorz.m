function varargout=ValueFnFromPolicy_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Typically, varargout={V}.
% Under quasi-hyperbolic discounting it is varargout={V,Valt}; see
% ValueFnFromPolicy_InfHorz_QuasiHyperbolic for what the two mean in the Naive and
% Sophisticated cases. Declared as varargout (rather than a fixed output list) so that a
% two-output call is legal, matching ValueFnFromPolicy_FHorz: MATLAB checks the requested
% output count against the declared list before any line of the body runs, so a fixed
% 'function V=' would reject [V,Valt]=... before the dispatch below could ever be reached.

if ~exist('vfoptions','var')
    vfoptions.gridinterplayer=0;
    vfoptions.tolerance=10^(-9);
    vfoptions.maxiter=10^4; % Can be used to stop the VFI after a finite number of iterations
    % divide-and-conquer is not relevant for ValueFnFromPolicy
    vfoptions.n_e=0;
    vfoptions.n_semiz=0;
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
    % divide-and-conquer is not relevant for ValueFnFromPolicy
    if ~isfield(vfoptions,'n_e')
        vfoptions.n_e=0;
    end
    if ~isfield(vfoptions,'n_semiz')
        vfoptions.n_semiz=0;
    end
end

N_e=prod(vfoptions.n_e);
if N_e>0
    if isfield(vfoptions,'exoticpreferences') && strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        error('vfoptions.n_e (iid e) is not yet implemented for ValueFnFromPolicy_InfHorz with vfoptions.exoticpreferences=QuasiHyperbolic')
    end
end
N_semiz=prod(vfoptions.n_semiz);
if N_semiz>0
    if vfoptions.gridinterplayer==1 && ~isscalar(n_a)
        error('vfoptions.n_semiz (semi-exogenous states) is not yet implemented for ValueFnFromPolicy_InfHorz with the grid interpolation layer and more than one endogenous state')
    elseif isfield(vfoptions,'exoticpreferences') && ~(isequal(vfoptions.exoticpreferences,0) || strcmp(vfoptions.exoticpreferences,'None'))
        error('vfoptions.n_semiz (semi-exogenous states) is not yet implemented for ValueFnFromPolicy_InfHorz with vfoptions.exoticpreferences')
    end
end

%% Dispatch to QuasiHyperbolic subfn if exoticpreferences=='QuasiHyperbolic'
% This sits BEFORE the grid-interpolation hand-off, matching ValueFnFromPolicy_FHorz, where the
% quasi-hyperbolic dispatch likewise precedes every other dispatch. The subfn handles
% gridinterplayer itself; sending grid-interpolation models to ValueFnFromPolicy_InfHorz_GI first
% would strand them in a command that knows nothing about quasi-hyperbolic discounting.
if isfield(vfoptions,'exoticpreferences') && strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    [V,Valt]=ValueFnFromPolicy_InfHorz_QuasiHyperbolic(Policy,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V,Valt};
    return
end
if isfield(vfoptions,'exoticpreferences') && ~strcmp(vfoptions.exoticpreferences,'None')
    error('ValueFnFromPolicy_InfHorz() does not yet work with this exotic preference type (QuasiHyperbolic is implemented; the others are not). Please ask on forum if you want/need this feature.')
end

%% Grid interpolation layer is handled by its own command (except with semiz, which ValueFnFromPolicy_InfHorz_SemiExo handles, see below)
if vfoptions.gridinterplayer==1 && N_semiz==0
    V=ValueFnFromPolicy_InfHorz_GI(Policy,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
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
[z_gridvals, pi_z, vfoptions]=ExogShockSetup_InfHorz(n_z,z_grid,pi_z,Parameters,vfoptions,3,0);

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

%% Semi-exogenous states are handled by their own command
if N_semiz>0
    V=ValueFnFromPolicy_InfHorz_SemiExo(Policy,n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,ReturnFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,vfoptions);
    varargout={V};
    return
end
% Basic setup: the first inputs of ReturnFn will be (d,aprime,a,z,..) and everything after this is a parameter, so we get the names of all these parameters.
% But this changes if you have e, semiz, or just multiple d, and if you use riskyasset, expasset, etc.
% So figure out which setup we have, and get the relevant ReturnFnParamNames

%% Calculate FofPolicy (the return fn evaluated at the Policy)
PolicyValues=PolicyInd2Val_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid, vfoptions); % (PolicyInd2Val_InfHorz treats e as part of z)
ReturnFnParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames);
if N_e==0
    if N_z==0
        PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a]),[2,1]); %[N_a,l_d+l_a]
    else
        PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]
    end
    FofPolicy=EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
else
    % To evaluate the return fn, just treat e as part of z
    if N_z==0
        n_ze=vfoptions.n_e;
        ze_gridvals=vfoptions.e_gridvals;
    else
        n_ze=[n_z,vfoptions.n_e];
        ze_gridvals=[repmat(z_gridvals,N_e,1),repelem(vfoptions.e_gridvals,N_z,1)];
    end
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,prod(n_ze)]),[2,3,1]); %[N_a,N_z*N_e,l_d+l_a]
    FofPolicy=EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_ze,a_gridvals,ze_gridvals);
    if N_z==0
        FofPolicy=reshape(FofPolicy,[N_a,N_e]);
    else
        FofPolicy=reshape(FofPolicy,[N_a,N_z,N_e]);
    end
end

%% Now that we have FofPolicy, calculate V.
DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames));

currdist=Inf;
itercount=1;
VKron=FofPolicy/(1-DiscountFactorParamsVec); % rough guess
if N_z==0 && N_e==0
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
elseif N_z>0 && N_e==0
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
elseif N_z==0 && N_e>0
    Policy=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, vfoptions.n_e, 0, vfoptions); % no z, so can treat e like z

    pi_e_alt=shiftdim(vfoptions.pi_e,-1); % 1-by-e

    if N_d==0
        Policy_a=shiftdim(Policy(1,:,:),1);
    else
        Policy_a=shiftdim(ceil(Policy(2,:,:)),1);
    end
    Policy_a=reshape(Policy_a,[N_a,N_e]);

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        % e is iid, so take the expectation over e' first
        EVKrontemp=VKron.*pi_e_alt;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=sum(EVKrontemp,2); % aprime-by-1
        VKron=FofPolicy+DiscountFactorParamsVec*reshape(EVKrontemp(Policy_a),[N_a,N_e]);

        currdist=max(abs(VKron(:)-VKronold(:)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,vfoptions.n_e]);
else % N_z>0 && N_e>0
    Policy=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, [n_z,vfoptions.n_e], 0, vfoptions); % e passed as part of z

    pi_z_alt=shiftdim(pi_z',-1); % 1-by-zprime-by-z
    pi_e_alt=shiftdim(vfoptions.pi_e,-2); % 1-by-1-by-e

    if N_d==0
        Policy_a=shiftdim(Policy(1,:,:),1);
    else
        Policy_a=shiftdim(ceil(Policy(2,:,:)),1);
    end
    Policy_az=reshape(Policy_a,[N_a,N_z,N_e])+repmat(N_a*(0:1:N_z-1),N_a,1,N_e); % aprime index, add the index for z (for indexing EV, which is aprime-by-z)

    while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
        VKronold=VKron;

        % e is iid, so take the expectation over e' first, then over z' conditional on z
        EVKrontemp=VKron.*pi_e_alt;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=sum(EVKrontemp,3); % aprime-by-zprime
        EVKrontemp=EVKrontemp.*pi_z_alt;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]); % aprime-by-z
        VKron=FofPolicy+DiscountFactorParamsVec*EVKrontemp(Policy_az);

        currdist=max(abs(VKron(:)-VKronold(:)));
        itercount=itercount+1;
    end

    V=reshape(VKron,[n_a,n_z,vfoptions.n_e]);
end

if currdist>vfoptions.tolerance
    warning(['ValueFnFromPolicy_InfHorz: the policy-evaluation fixed point stopped on reaching ', ...
             'the maximum number of iterations, not on convergence (set by vfoptions.maxiter). ', ...
             'The returned V is not the value of the given Policy. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], currdist, vfoptions.tolerance)
end

varargout={V};

end
