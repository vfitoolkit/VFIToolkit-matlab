function varargout=ValueFnFromPolicy_FHorz_RiskyAsset(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model has riskyasset (vfoptions.riskyasset==1).
% riskyasset: a2prime = aprimeFn(d2,d3, u) -- depends on iid (between-period) shock u, NOT on current a2.
% u does NOT enter Policy; pi_u integrates u out when computing E[V'|policy].
%
% Structural cousin of ValueFnFromPolicy_FHorz_ExpAssetu (same u-shock structure). The only
% substantive differences are: (i) the aprime-from-policy helper is CreateaprimePolicyRiskyAsset
% (riskyasset's aprimeFn ignores current a2); (ii) which d feed the aprimeFn is set by
% vfoptions.refine_d (d2 and d3), not by l_dexperienceassetu.

%% Dispatch to SemiExo subfn if n_semiz>0
if prod(vfoptions.n_semiz)>0
    V=ValueFnFromPolicy_FHorz_RiskyAsset_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end
%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    V=ValueFnFromPolicy_FHorz_RiskyAsset_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%% Setup
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use riskyasset you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;
if ~isfield(vfoptions,'n_u')
    error('To use riskyasset you must define vfoptions.n_u')
end
if ~isfield(vfoptions,'u_grid')
    error('To use riskyasset you must define vfoptions.u_grid')
end
if ~isfield(vfoptions,'pi_u')
    error('To use riskyasset you must define vfoptions.pi_u')
end
n_u=vfoptions.n_u;
u_grid=gpuArray(vfoptions.u_grid);
pi_u=gpuArray(vfoptions.pi_u);
N_u=prod(n_u);
l_u=length(n_u);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_d==0
    error('ValueFnFromPolicy_FHorz_RiskyAsset: riskyasset requires at least one decision variable (driving a2prime)')
end
l_d=length(n_d);
l_a=length(n_a);

% Split a into a1 (standard) and a2 (risky asset).
% noa1 case (n_a is scalar -- risky asset is the only endogenous state): use n_a1=0, N_a1=0
% (toolkit convention; matches StationaryDist_FHorz_RiskyAsset). Note we override l_a1=0 because
% length(0)=1, not 0. Downstream, the lookup section has explicit `if N_a1==0` branches.
if isscalar(n_a)
    n_a1=0;
    N_a1=0;
    l_a1=0;
else
    n_a1=n_a(1:end-1);
    N_a1=prod(n_a1);
    l_a1=length(n_a1);
end
n_a2=n_a(end);
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_a2=length(n_a2);
l_aprime=l_a1; % Policy stores a1prime only (a2prime implicit); 0 in the noa1 case

% Which d feed the aprimeFn: d2 and d3 under vfoptions.refine_d (matches StationaryDist_FHorz_RiskyAsset).
if ~isfield(vfoptions,'refine_d')
    vfoptions.refine_d=[0,0,length(n_d)]; % everything implicitly a d3 (in both aprimeFn and ReturnFn)
end
whichisdforriskyasset=(vfoptions.refine_d(1)+1):1:sum(vfoptions.refine_d(1:3));
l_drisky=length(whichisdforriskyasset);

% aprimeFnParamNames: first inputs to the riskyasset aprimeFn are (d2,d3, u) -- NO current a2.
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_drisky+l_u)
    aprimeFnParamNames={temp{l_drisky+l_u+1:end}};
else
    aprimeFnParamNames={};
end

% Combined shock dim for CreateaprimePolicyRiskyAsset (passed in place of N_z)
if N_z==0 && N_e==0
    N_ze=0;
elseif N_z>0 && N_e==0
    N_ze=N_z;
elseif N_z==0 && N_e>0
    N_ze=N_e;
else
    N_ze=N_z*N_e;
end

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues (PolicyInd2Val_FHorz handles riskyasset internally)
% The riskyasset ReturnFn takes (d1, d3, a1prime, a1, a2, ...): d2 (aprimeFn-only) is NOT an input.
% This is the 'refine' split the VFI codes use (vfoptions.refine_d): d1 in ReturnFn only,
% d2 in aprimeFn only, d3 in both -- so the VFI's ReturnMatrix is built over n_d13=[n_d1,n_d3] and a1.
% So keep only the d1 and d3 decision rows (then the a1prime rows) for the ReturnFn evaluation.
d1rows=1:vfoptions.refine_d(1);
d3rows=(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1):(vfoptions.refine_d(1)+vfoptions.refine_d(2)+vfoptions.refine_d(3));
returnrows=[d1rows, d3rows, (l_d+1):(l_d+l_a1)]; % ReturnFn d's (d1 then d3), then a1prime
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
PolicyValues=PolicyValues(returnrows,:,:,:); % drop d2 (not in the ReturnFn)
l_daprime=size(PolicyValues,1); % = n_d1 + n_d3 + l_a1
if N_z==0 && N_e==0
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]); % [N_a, l_daprime, N_j]
else
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_ze, l_daprime, N_j]
end

%% Reshape Policy to canonical Kron form
if N_z==0 && N_e==0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_j]);
elseif N_z>0 && N_e==0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z, N_j]);
elseif N_z==0 && N_e>0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_e, N_j]);
else
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z*N_e, N_j]);
end

%% Build a1prime joint index
if N_z==0 && N_e==0
    a1prime_idx=ones(N_a, N_j, 'gpuArray');
else
    a1prime_idx=ones(N_a, N_ze, N_j, 'gpuArray');
end
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
for ii=1:l_a1
    comp=shiftdim(Policy_k(l_d+ii, :, :, :),1);
    a1prime_idx=a1prime_idx+cumprods_a1(ii)*(comp-1);
end

%% Joint z+e gridvals for ReturnFn when both present
if N_z>0 && N_e>0
    joint_zegridvals_J=zeros(N_z*N_e, length(n_z)+length(vfoptions.n_e), N_j, 'gpuArray');
    for jj=1:N_j
        joint_zegridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
    end
end

%% V allocation
if N_z==0 && N_e==0
    V=zeros(N_a, N_j, 'gpuArray');
elseif N_z==0 && N_e>0
    V=zeros(N_a, N_e, N_j, 'gpuArray');
elseif N_z>0 && N_e==0
    V=zeros(N_a, N_z, N_j, 'gpuArray');
else
    V=zeros(N_a, N_z, N_e, N_j, 'gpuArray');
end

%% Backward iteration
for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_z==0 && N_e==0
        Policy_slice=Policy_k(:,:,jj); % [l_d+l_a1, N_a]
    else
        Policy_slice=Policy_k(:,:,:,jj); % [l_d+l_a1, N_a, N_ze]
    end

    % Step 1: a2primeIndex, a2primeProbs -- helper adds the u dim (riskyasset: aprime ignores a2)
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyRiskyAsset(Policy_slice, aprimeFn, whichisdforriskyasset, n_d, n_a1, n_a2, N_ze, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    % shape:  N_ze==0 -> [N_a, N_u];  else -> [N_a, N_ze, N_u]

    % Step 2: ReturnFn at policy (u does not enter Return)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_z==0 && N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,jj), l_daprime, n_a, 0, a_gridvals, []);
    elseif N_z==0 && N_e>0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, vfoptions.n_e, a_gridvals, vfoptions.e_gridvals_J(:,:,jj));
    elseif N_z>0 && N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
    end

    if jj==N_j
        if N_z==0 && N_e==0
            V(:,jj)=F_jj;
        elseif N_z==0 && N_e>0
            V(:,:,jj)=F_jj;
        elseif N_z>0 && N_e==0
            V(:,:,jj)=F_jj;
        else
            V(:,:,:,jj)=F_jj;
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

        % Step 3: EVnext indexed by (anext, z_from) -- integrate e' (iid) then z' (markov)
        if N_z==0 && N_e==0
            EVnext=V(:,jj+1); % [N_a]
        elseif N_z==0 && N_e>0
            EVnext=sum(V(:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj), -1), 2); % [N_a, 1]
        elseif N_z>0 && N_e==0
            EVnext=V(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        else
            EVnext=sum(V(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj), -2), 3); % [N_a, N_z, 1]
            EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        end

        % Step 4: interpolated lookup. Mirrors standard VFI's skipinterp+isnan so that policies
        % landing on infeasible-on-both-sides next-states give the same finite V here. Order matters:
        % sum over u BEFORE pi_z multiplication and isnan clear.
        if N_z==0 && N_e==0
            if N_a1==0
                aprime_low=a2primeIndex;     % [N_a, N_u]
                aprime_up =a2primeIndex+1;
            else
                a1p=a1prime_idx(:,jj); % [N_a]
                aprime_low=a1p+N_a1*(a2primeIndex-1); % [N_a, N_u]
                aprime_up =a1p+N_a1*(a2primeIndex);
            end
            Vlower=reshape(EVnext(aprime_low(:)),[N_a,N_u]);
            Vupper=reshape(EVnext(aprime_up(:)), [N_a,N_u]);
            a2pPrb=a2primeProbs;
            a2pPrb(Vlower==Vupper)=0; % skipinterp
            EV=a2pPrb.*Vlower+(1-a2pPrb).*Vupper;
            EVnext_atpolicy=sum(EV .* shiftdim(pi_u,-1), 2); % sum over u -> [N_a, 1]
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0;
            V(:,jj)=F_jj+beta*EVnext_atpolicy;
        elseif N_z==0 && N_e>0
            if N_a1==0
                aprime_low=a2primeIndex;     % [N_a, N_e, N_u]
                aprime_up =a2primeIndex+1;
            else
                a1p=a1prime_idx(:,:,jj); % [N_a, N_e]
                aprime_low=a1p+N_a1*(a2primeIndex-1); % broadcast -> [N_a, N_e, N_u]
                aprime_up =a1p+N_a1*(a2primeIndex);
            end
            Vlower=reshape(EVnext(aprime_low(:)),[N_a,N_e,N_u]);
            Vupper=reshape(EVnext(aprime_up(:)), [N_a,N_e,N_u]);
            a2pPrb=a2primeProbs;
            a2pPrb(Vlower==Vupper)=0; % skipinterp on pi_e-collapsed EVnext
            EV=a2pPrb.*Vlower+(1-a2pPrb).*Vupper;
            EVnext_atpolicy=sum(EV .* shiftdim(pi_u,-2), 3); % sum over u -> [N_a, N_e]
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0;
            V(:,:,jj)=F_jj+beta*EVnext_atpolicy;
        elseif N_z>0 && N_e==0
            if N_a1==0
                aprime_low=a2primeIndex;     % [N_a, N_z, N_u]
                aprime_up =a2primeIndex+1;
            else
                a1p=a1prime_idx(:,:,jj);
                aprime_low=a1p+N_a1*(a2primeIndex-1); % broadcast -> [N_a, N_z, N_u]
                aprime_up =a1p+N_a1*(a2primeIndex);
            end
            % Per-z' interpolation against V(:,:,jj+1) directly (not EVnext, which has pi_z already summed).
            Vnext=V(:,:,jj+1); % [N_a, N_z']
            zprimeoffset=reshape(N_a*gpuArray(0:N_z-1),[1,1,1,N_z]); % [1,1,1,N_z']
            Vlower=reshape(Vnext(aprime_low+zprimeoffset),[N_a,N_z,N_u,N_z]);
            Vupper=reshape(Vnext(aprime_up +zprimeoffset),[N_a,N_z,N_u,N_z]);
            a2pPrb4=repmat(a2primeProbs,[1,1,1,N_z]);
            a2pPrb4(Vlower==Vupper)=0; % skipinterp
            EV4=a2pPrb4.*Vlower+(1-a2pPrb4).*Vupper; % [N_a, N_z, N_u, N_z']
            EV4=sum(EV4 .* shiftdim(pi_u,-2), 3); % sum over u -> [N_a, N_z, 1, N_z']
            EV4=EV4 .* reshape(pi_z_J(:,:,jj),[1,N_z,1,N_z]); % weight by pi_z(z, z')
            EV4(isnan(EV4))=0;
            EVnext_atpolicy=reshape(sum(EV4,4),[N_a,N_z]); % sum over z'
            V(:,:,jj)=F_jj+beta*EVnext_atpolicy;
        else
            % a2pi/pp flat [N_a, N_z*N_e, N_u] -> [N_a, N_z, N_e, N_u]
            a2pIdx=reshape(a2primeIndex,[N_a, N_z, N_e, N_u]);
            a2pPrb=reshape(a2primeProbs,[N_a, N_z, N_e, N_u]);
            if N_a1==0
                aprime_low=a2pIdx;     % [N_a, N_z, N_e, N_u]
                aprime_up =a2pIdx+1;
            else
                a1p=reshape(a1prime_idx(:,:,jj),[N_a, N_z, N_e]);
                aprime_low=a1p+N_a1*(a2pIdx-1); % broadcast a1p over u -> [N_a, N_z, N_e, N_u]
                aprime_up =a1p+N_a1*(a2pIdx);
            end
            % Pre-collapse pi_e (e iid), keep z' for per-z' skipinterp (matches standard VFI z+e branch).
            EVpre=reshape(sum(V(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj),-2), 3),[N_a,N_z]); % [N_a, N_z']
            zprimeoffset=reshape(N_a*gpuArray(0:N_z-1),[1,1,1,1,N_z]); % [1,1,1,1,N_z']
            Vlower=reshape(EVpre(aprime_low+zprimeoffset),[N_a,N_z,N_e,N_u,N_z]);
            Vupper=reshape(EVpre(aprime_up +zprimeoffset),[N_a,N_z,N_e,N_u,N_z]);
            a2pPrb5=repmat(a2pPrb,[1,1,1,1,N_z]);
            a2pPrb5(Vlower==Vupper)=0; % skipinterp
            EV5=a2pPrb5.*Vlower+(1-a2pPrb5).*Vupper; % [N_a, N_z, N_e, N_u, N_z']
            EV5=sum(EV5 .* shiftdim(pi_u,-3), 4); % sum over u -> [N_a, N_z, N_e, 1, N_z']
            EV5=EV5 .* reshape(pi_z_J(:,:,jj),[1,N_z,1,1,N_z]); % weight by pi_z(z, z')
            EV5(isnan(EV5))=0;
            EVnext_atpolicy=reshape(sum(EV5,5),[N_a,N_z,N_e]); % sum over z'
            V(:,:,:,jj)=F_jj+beta*EVnext_atpolicy;
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

varargout={V};

end
