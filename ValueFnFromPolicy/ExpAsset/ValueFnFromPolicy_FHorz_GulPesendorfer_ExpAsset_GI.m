function V=ValueFnFromPolicy_FHorz_GulPesendorfer_ExpAsset_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Parameters,DiscountFactorParamNames,vfoptions)
% Gul-Pesendorfer variant of ValueFnFromPolicy_FHorz_ExpAsset_GI: values the given Policy under
%   V_j = u(policy_j) + v(policy_j) - MostTempting_j + beta*E[V_{j+1} at policy_j]
% (no continuation term at j=N_j) when the model has an experience asset AND uses the grid
% interpolation layer (vfoptions.gridinterplayer==1).
%
% Under GI, Policy stores an extra L2 (layer-2 fine-grid) index at the end, used to interpolate
% a1prime (the first endogenous asset) between two adjacent a1_grid points. a2prime continues to
% be interpolated via aprimeFn (a2primeIndex/a2primeProbs onto a2_grid). Per-state EVnext lookup
% is therefore a 2x2 interpolation: lower/upper a1 × lower/upper a2 (4 corner V values, weighted
% product of marginals).
%
% The a1prime choice set is the FINE grid, so MostTempting_j(a,z) is the max of v over the FINE
% grid jointly with (d1,)d2, found by the same two-stage scheme as in the GP ExpAsset GI solver
% raws: around v's OWN coarse argmax (otherwise the chosen fine point could be more tempting
% than the coarse max of v, making the self-control cost negative). v never touches the
% continuation, so the temptation side needs none of the aprime-probs machinery.
%
% This is dispatched from ValueFnFromPolicy_FHorz_GulPesendorfer_ExpAsset AFTER the parent
% ValueFnFromPolicy_FHorz has run ExogShockSetup_FHorz, so z_gridvals_J/pi_z_J and
% vfoptions.e_gridvals_J/pi_e_J arrive pre-processed (vfoptions.n_e exists, 0-equivalent when
% there are no e variables).

%% Setup
if ~isfield(vfoptions,'aprimeFn')
    error('To use an experience asset you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;
TemptationFn=vfoptions.temptationFn;

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_d==0
    error('ValueFnFromPolicy_FHorz_GulPesendorfer_ExpAsset_GI: experienceasset requires at least one decision variable')
end
l_d=length(n_d);
l_a=length(n_a);

% noa1 case (n_a is scalar -- experience asset is the only endogenous state): GI refines a1, which
% doesn't apply when there's no a1. Fall back to non-GI version (which handles noa1 correctly).
% Matches the upstream VFI convention (noa1 has no GI/DC/DC+GI raw files).
if isscalar(n_a)
    vfoptions.gridinterplayer=0;
    V=ValueFnFromPolicy_FHorz_GulPesendorfer_ExpAsset(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Parameters,DiscountFactorParamNames,vfoptions);
    return
end
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
n_a2=n_a(end);
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);
l_aprime=l_a1; % Policy stores a1prime only (plus L2 in GI)

% Multi-dim n_a1 (the DC2A/GI2A tiers) is not yet implemented under Gul-Pesendorfer
if length(n_a1)>1
    error('GulPesendorfer with experienceasset is not yet implemented for two standard endogenous states (length(n_a1)>1)')
end

if isfield(vfoptions,'l_dexperienceasset')
    l_d2=vfoptions.l_dexperienceasset;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_d2+1):l_d;
n_d2=n_d(end-l_d2+1:end);
% n_d1/n_d2 split for the most-tempting creators (n_d1=0 when all of d drives the experience asset)
if l_d>l_d2
    n_d1=n_d(1:end-l_d2);
else
    n_d1=0;
end

% aprimeFnParamNames
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2)
    aprimeFnParamNames={temp{l_d2+l_a2+1:end}};
else
    aprimeFnParamNames={};
end

if N_z==0 && N_e==0
    N_ze=0;
elseif N_z>0 && N_e==0
    N_ze=N_z;
elseif N_z==0 && N_e>0
    N_ze=N_e;
else
    N_ze=N_z*N_e;
end

% Grid interpolation
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer (for the most-tempting term)

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
TemptationFnParamNames=ReturnFnParamNamesFn(TemptationFn,n_d,n_a,n_z,N_j,vfoptions,Parameters); % the temptation fn has the same leading model args as the return fn, so the same convention applies

a_gridvals=CreateGridvals(n_a,a_grid,1);
d_gridvals=CreateGridvals(n_d,d_grid,1); % the temptation matrix creators need gridvals
a1_gridvals=CreateGridvals(n_a1,a1_grid,1);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % the CreateReturnFnMatrix_ExpAsset_Disc* creators want gridvals, not the stacked a2_grid
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short)); % fine a1prime grid for the most-tempting two-stage max

%% PolicyValues (PolicyInd2Val_FHorz handles experienceasset + GI internally: drops a2prime, combines a1prime lower grid index + L2 into fine index)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
if N_z==0 && N_e==0
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]); % [N_a, l_daprime, N_j]
else
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_ze, l_daprime, N_j]
end

%% Strip trailing L2flag channel if present (Policy may carry it; we only need l_d+l_a1+1 channels)
size_first=l_d+l_a1+1;
if size(Policy,1) > size_first
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1), prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:size_first,:), [size_first, tempsize(2:end)]);
end

%% Reshape Policy to canonical Kron form: [l_d+l_a1+1, N_a, N_ze, N_j] (or no shock dim when no shocks)
if N_z==0 && N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_j]);
elseif N_z>0 && N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_z, N_j]);
elseif N_z==0 && N_e>0
    Policy_k=reshape(Policy,[size_first, N_a, N_e, N_j]);
else
    Policy_k=reshape(Policy,[size_first, N_a, N_z*N_e, N_j]);
end

%% Extract a1prime lower grid index and L2 from Policy
% Position l_d+1 is the a1prime lower grid index (first a1 component); l_d+2 .. l_d+l_a1 are other a1prime indices; l_d+l_a1+1 is L2
if N_z==0 && N_e==0
    a1_lower=ones(N_a, N_j, 'gpuArray');
else
    a1_lower=ones(N_a, N_ze, N_j, 'gpuArray');
end
% First a1 component (the interpolated one): lower grid index = Policy(l_d+1)
% ValueFnIter converts the midpoint to the lower grid index before returning Policy (the adjust
% block at the end of the GI raws), so this row is the lower index and not the midpoint.
a1_lowerind=shiftdim(Policy_k(l_d+1,:,:,:),1);
L2=shiftdim(Policy_k(l_d+l_a1+1,:,:,:),1);
w_a1_upper=(L2-1)/(n2short+1); % weight on upper a1 grid point
w_a1_lower=1-w_a1_upper;
% Other a1prime components (for l_a1>1): standard indices, contribute fixed kron offset
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
a1_lower=a1_lowerind; % first dim contribution (1*(a1_lowerind-1)+1 = a1_lowerind)
for ii=2:l_a1
    comp=shiftdim(Policy_k(l_d+ii,:,:,:),1);
    a1_lower=a1_lower+cumprods_a1(ii)*(comp-1);
end
% upper a1 differs only in the first a1 component (clamp at top of grid)
a1_upper=a1_lower+1;
a1_top_clamp=(a1_lowerind>=n_a1(1));
a1_upper(a1_top_clamp)=a1_lower(a1_top_clamp); % no-op when at top

%% Joint zegridvals for ReturnFn (when both z and e present)
if N_z>0 && N_e>0
    joint_zegridvals_J=zeros(N_z*N_e, length(n_z)+length(vfoptions.n_e), N_j, 'gpuArray');
    for jj=1:N_j
        joint_zegridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
    end
end

%% Backward iteration
if N_z==0 && N_e==0
    V=zeros(N_a, N_j, 'gpuArray');
elseif N_z==0 && N_e>0
    V=zeros(N_a, N_e, N_j, 'gpuArray');
elseif N_z>0 && N_e==0
    V=zeros(N_a, N_z, N_j, 'gpuArray');
else
    V=zeros(N_a, N_z, N_e, N_j, 'gpuArray');
end

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Step 1: a2primeIndex, a2primeProbs at this age
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_z==0 && N_e==0
        Policy_slice=Policy_k(:,:,jj); % [size_first, N_a]
    else
        Policy_slice=Policy_k(:,:,:,jj); % [size_first, N_a, N_ze]
    end
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAsset(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, N_ze, d_grid, a2_grid, aprimeFnParamsVec);

    % Step 2: ReturnFn and TemptationFn at policy
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
    if N_z==0 && N_e==0
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,jj), l_daprime, n_a, 0, a_gridvals, []);
        TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,jj), l_daprime, n_a, 0, a_gridvals, []);
    elseif N_z==0 && N_e>0
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, vfoptions.n_e, a_gridvals, vfoptions.e_gridvals_J(:,:,jj));
        TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, vfoptions.n_e, a_gridvals, vfoptions.e_gridvals_J(:,:,jj));
    elseif N_z>0 && N_e==0
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
        TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
    else
        FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
        TofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
    end

    % Most-tempting term: two-stage max of v over the FINE grid, around v's own coarse argmax
    % (same conventions as the GP ExpAsset GI solver raws: Level=1 coarse twin, per-d argmax over
    % a1prime, midpoint clamped to [2,n_a1(1)-1], fine window via Level=2 creator, max over the
    % joint ((d1,)d2, fine a1prime window) first dim)
    TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
    if N_z==0 && N_e==0
        TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, n_d1, n_d2, n_a1, n_a1, n_a2, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, TemptationFnParamsVec,1,0); % Level=1, Refine=0
        [~,maxindexT]=max(TemptationMatrix,[],2);
        midpointT=max(min(maxindexT,n_a1(1)-1),2);
        a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, n_d1, n_d2, n2long, n_a1, n_a2, d_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, TemptationFnParamsVec,2,0); % [N_d*n2long,N_a1*N_a2]; Level=2, Refine=0
        MostTempting=reshape(max(TemptationMatrix_Tii,[],1),[N_a,1]);
        MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
    elseif N_z==0 && N_e>0
        TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, n_d1, n_d2, n_a1, n_a1, n_a2, vfoptions.n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,1,0); % Level=1, Refine=0; Because no z, can treat e like z
        [~,maxindexT]=max(TemptationMatrix,[],2);
        midpointT=max(min(maxindexT,n_a1(1)-1),2);
        a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, n_d1, n_d2, n2long, n_a1, n_a2, vfoptions.n_e, d_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,2,0); % [N_d*n2long,N_a1*N_a2,N_e]; Level=2, Refine=0
        MostTempting=reshape(max(TemptationMatrix_Tii,[],1),[N_a,N_e]);
        MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
    elseif N_z>0 && N_e==0
        TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, n_d1, n_d2, n_a1, n_a1, n_a2, n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), TemptationFnParamsVec,1,0); % Level=1, Refine=0
        [~,maxindexT]=max(TemptationMatrix,[],2);
        midpointT=max(min(maxindexT,n_a1(1)-1),2);
        a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, n_d1, n_d2, n2long, n_a1, n_a2, n_z, d_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), TemptationFnParamsVec,2,0); % [N_d*n2long,N_a1*N_a2,N_z]; Level=2, Refine=0
        MostTempting=reshape(max(TemptationMatrix_Tii,[],1),[N_a,N_z]);
        MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
    else
        TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1, n_d2, n_a1, n_a1, n_a2, n_z, vfoptions.n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,1,0); % Level=1, Refine=0
        [~,maxindexT]=max(TemptationMatrix,[],2);
        midpointT=max(min(maxindexT,n_a1(1)-1),2);
        a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
        TemptationMatrix_Tii=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1, n_d2, n2long, n_a1, n_a2, n_z, vfoptions.n_e, d_gridvals, a1prime_grid(a1primeindexesT), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,2,0); % [N_d*n2long,N_a1*N_a2,N_z,N_e]; Level=2, Refine=0
        MostTempting=reshape(max(TemptationMatrix_Tii,[],1),[N_a,N_z,N_e]);
        MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
    end

    if jj==N_j
        if N_z==0 && N_e==0
            V(:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
        elseif N_z==0 && N_e>0
            V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
        elseif N_z>0 && N_e==0
            V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
        else
            V(:,:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting;
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

        % Step 3: build EVnext indexed by (anext, z_from)
        if N_z==0 && N_e==0
            EVnext=V(:,jj+1); % [N_a]
        elseif N_z==0 && N_e>0
            EVnext=sum(V(:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -1), 2); % [N_a, 1]
        elseif N_z>0 && N_e==0
            EVnext=V(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        else
            EVnext=sum(V(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3); % [N_a, N_z, 1]
            EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        end

        % Step 4: 2x2 interpolated lookup using (a1_lower/upper, w_a1) × (a2_lower/upper, a2primeProbs)
        % aprime_kron(corner) = a1_corner + N_a1 * (a2_corner - 1)
        if N_z==0 && N_e==0
            % aprime_low, aprime_up shapes [N_a]; a2primeIndex, a2primeProbs [N_a, 1]
            a1l=a1_lower(:,jj); a1u=a1_upper(:,jj);
            wa1l=w_a1_lower(:,jj); wa1u=w_a1_upper(:,jj);
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            EV_LL=EVnext(a1l+N_a1*(a2l-1));
            EV_LU=EVnext(a1l+N_a1*(a2u-1));
            EV_UL=EVnext(a1u+N_a1*(a2l-1));
            EV_UU=EVnext(a1u+N_a1*(a2u-1));
            EVnext_atpolicy=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % zero corner weights times -Inf next-states give NaN
            V(:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*EVnext_atpolicy;
        elseif N_z==0 && N_e>0
            a1l=a1_lower(:,:,jj); a1u=a1_upper(:,:,jj);
            wa1l=w_a1_lower(:,:,jj); wa1u=w_a1_upper(:,:,jj);
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            lin_LL=a1l+N_a1*(a2l-1); lin_LU=a1l+N_a1*(a2u-1);
            lin_UL=a1u+N_a1*(a2l-1); lin_UU=a1u+N_a1*(a2u-1);
            EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_e]);
            EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_e]);
            EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_e]);
            EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_e]);
            EVnext_atpolicy=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % zero corner weights times -Inf next-states give NaN
            V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*EVnext_atpolicy;
        elseif N_z>0 && N_e==0
            a1l=a1_lower(:,:,jj); a1u=a1_upper(:,:,jj);
            wa1l=w_a1_lower(:,:,jj); wa1u=w_a1_upper(:,:,jj);
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            zidxoffset=N_a*gpuArray(0:N_z-1); % [1, N_z]
            lin_LL=a1l+N_a1*(a2l-1)+zidxoffset; lin_LU=a1l+N_a1*(a2u-1)+zidxoffset;
            lin_UL=a1u+N_a1*(a2l-1)+zidxoffset; lin_UU=a1u+N_a1*(a2u-1)+zidxoffset;
            EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_z]);
            EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_z]);
            EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_z]);
            EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_z]);
            EVnext_atpolicy=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % zero corner weights times -Inf next-states give NaN
            V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*EVnext_atpolicy;
        else
            a1l=reshape(a1_lower(:,:,jj),[N_a,N_z,N_e]);  a1u=reshape(a1_upper(:,:,jj),[N_a,N_z,N_e]);
            wa1l=reshape(w_a1_lower(:,:,jj),[N_a,N_z,N_e]); wa1u=reshape(w_a1_upper(:,:,jj),[N_a,N_z,N_e]);
            a2l=reshape(a2primeIndex,[N_a,N_z,N_e]); a2u=a2l+1;
            wa2l=reshape(a2primeProbs,[N_a,N_z,N_e]); wa2u=1-wa2l;
            zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]);
            lin_LL=a1l+N_a1*(a2l-1)+zidxoffset; lin_LU=a1l+N_a1*(a2u-1)+zidxoffset;
            lin_UL=a1u+N_a1*(a2l-1)+zidxoffset; lin_UU=a1u+N_a1*(a2u-1)+zidxoffset;
            EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_z,N_e]);
            EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_z,N_e]);
            EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_z,N_e]);
            EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_z,N_e]);
            EVnext_atpolicy=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % zero corner weights times -Inf next-states give NaN
            V(:,:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*EVnext_atpolicy;
        end
    end
end

%% Reshape V out of Kron form
if N_z==0 && N_e==0
    V=reshape(V, [n_a, N_j]);
elseif N_z==0 && N_e>0
    V=reshape(V, [n_a, vfoptions.n_e, N_j]);
elseif N_z>0 && N_e==0
    V=reshape(V, [n_a, n_z, N_j]);
else
    V=reshape(V, [n_a, n_z, vfoptions.n_e, N_j]);
end


end
