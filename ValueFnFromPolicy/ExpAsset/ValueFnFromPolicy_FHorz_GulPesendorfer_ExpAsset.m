function V=ValueFnFromPolicy_FHorz_GulPesendorfer_ExpAsset(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Parameters,DiscountFactorParamNames,vfoptions)
% Gul-Pesendorfer variant of ValueFnFromPolicy_FHorz_ExpAsset: values the given Policy under
%   V_j = u(policy_j) + v(policy_j) - MostTempting_j + beta*E[V_{j+1} at policy_j]
% (no continuation term at j=N_j), where u is the ReturnFn, v is the temptation fn
% (vfoptions.temptationFn, same input signature convention as the ReturnFn, own parameters),
% and MostTempting_j(a,z) is the max of v over the FULL joint ((d1,)d2(,a1prime)) choice set.
% The continuation is at a2prime=aprimeFn(d2,a2) exactly as in the standard ExpAsset case:
% Policy stores d and a1prime only (a2prime is implicit), and for the lookup of V at next
% period a2prime is (in general) fractional and is linearly interpolated onto the a2_grid via
% a2primeIndex/a2primeProbs. v never touches the continuation, so the temptation side needs
% none of the aprime-probs machinery (as in the GP ExpAsset solver raws).
%
% This is dispatched from ValueFnFromPolicy_FHorz_GulPesendorfer (which rules out semiz and
% the u/z/e/ze/semiz experience-asset variants) AFTER the parent ValueFnFromPolicy_FHorz has
% run ExogShockSetup_FHorz, so z_gridvals_J/pi_z_J and vfoptions.e_gridvals_J/pi_e_J arrive
% pre-processed (vfoptions.n_e exists, 0-equivalent when there are no e variables). Handles
% gridinterplayer itself.

%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    V=ValueFnFromPolicy_FHorz_GulPesendorfer_ExpAsset_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Parameters,DiscountFactorParamNames,vfoptions);
    return
end

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
    error('ValueFnFromPolicy_FHorz_GulPesendorfer_ExpAsset: experienceasset requires at least one decision variable (the one driving a2prime)')
end
l_d=length(n_d);
l_a=length(n_a);

% Split a into a1 (standard) and a2 (experience asset).
% noa1 case: when length(n_a)<=l_a2 (experience asset is the only endogenous state) use n_a1=0, N_a1=0
% (toolkit convention; matches StationaryDist_FHorz_ExpAsset). Note we have to override l_a1=0
% because length(0)=1, not 0. Downstream, the lookup section has explicit `if N_a1==0` branches,
% and the most-tempting term uses the Case2 creators (as the GP noa1 solver raws).
l_a2=vfoptions.experienceasset; % l_a2 = number of a2 (experience-asset) dims
if length(n_a)<=l_a2
    n_a1=0;
    N_a1=0;
    l_a1=0;
else
    n_a1=n_a(1:end-l_a2);
    N_a1=prod(n_a1);
    l_a1=length(n_a1);
end
n_a2=n_a(end-l_a2+1:end);
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_aprime=l_a1; % Policy stores a1prime only (a2prime implicit); 0 in the noa1 case

% Which d affects the experience asset (default: last d only)
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
if length(temp)>(l_d2+l_a2+(l_a2>=2))
    aprimeFnParamNames={temp{l_d2+l_a2+(l_a2>=2)+1:end}};
else
    aprimeFnParamNames={};
end

% Combined shock dim for CreateaprimePolicyExperienceAsset
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
TemptationFnParamNames=ReturnFnParamNamesFn(TemptationFn,n_d,n_a,n_z,N_j,vfoptions,Parameters); % the temptation fn has the same leading model args as the return fn, so the same convention applies

a_gridvals=CreateGridvals(n_a,a_grid,1);
d_gridvals=CreateGridvals(n_d,d_grid,1); % the temptation matrix creators need gridvals
if N_a1>0
    a1_gridvals=CreateGridvals(n_a1,a1_grid,1);
end
a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % the CreateReturnFnMatrix_* creators want gridvals ([N_a2-by-l_a2]), not the stacked a2_grid

%% PolicyValues (PolicyInd2Val_FHorz handles experienceasset internally, drops a2prime; auto-adds n_e)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
% PolicyValues shape:
% - N_z==0 && N_e==0: [l_daprime, N_a, N_j]
% - else: [l_daprime, N_a, N_ze, N_j]
if N_z==0 && N_e==0
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]); % [N_a, l_daprime, N_j]
else
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_ze, l_daprime, N_j]
end

%% Reshape Policy to canonical Kron form for CreateaprimePolicyExperienceAsset
% Target: [l_d+l_a1, N_a, N_ze, N_j] (or [l_d+l_a1, N_a, N_j] if no shocks)
if N_z==0 && N_e==0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_j]);
elseif N_z>0 && N_e==0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z, N_j]);
elseif N_z==0 && N_e>0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_e, N_j]);
else
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z*N_e, N_j]);
end

%% Build a1prime joint index across N_a1 dims (from Policy)
% a1prime components are positions l_d+1 .. l_d+l_a1
if N_z==0 && N_e==0
    a1prime_idx=ones(N_a,N_j,'gpuArray');
else
    a1prime_idx=ones(N_a,N_ze,N_j,'gpuArray');
end
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
for ii=1:l_a1
    comp=shiftdim(Policy_k(l_d+ii, :, :, :),1);
    a1prime_idx=a1prime_idx+cumprods_a1(ii)*(comp-1);
end

%% Joint gridvals for ReturnFn (z + e combined when both present)
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

    % Step 1: a2primeIndex, a2primeProbs for each state at this age
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_z==0 && N_e==0
        Policy_slice=Policy_k(:,:,jj); % [l_d+l_a1, N_a]
    else
        Policy_slice=Policy_k(:,:,:,jj); % [l_d+l_a1, N_a, N_ze]
    end
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAsset(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, N_ze, d_grid, a2_grid, aprimeFnParamsVec);
    % a2primeIndex, a2primeProbs shape:
    % - N_z==0 && N_e==0: [N_a, 1]
    % - else: [N_a, N_ze]

    % Step 2: ReturnFn and TemptationFn at policy
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
    if N_z==0 && N_e==0
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,jj), l_daprime, n_a, 0, a_gridvals, []);
        TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,jj), l_daprime, n_a, 0, a_gridvals, []);
        % FofPolicy_jj, TofPolicy_jj shape: [N_a, 1]
    elseif N_z==0 && N_e>0
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, vfoptions.n_e, a_gridvals, vfoptions.e_gridvals_J(:,:,jj));
        TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, vfoptions.n_e, a_gridvals, vfoptions.e_gridvals_J(:,:,jj));
        % FofPolicy_jj, TofPolicy_jj shape: [N_a, N_e]
    elseif N_z>0 && N_e==0
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
        TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
        % FofPolicy_jj, TofPolicy_jj shape: [N_a, N_z]
    else
        FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
        TofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
    end

    % Most-tempting term: max of v over the full joint ((d1,)d2(,a1prime)) choice set
    TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
    if N_a1==0
        % With only the experience asset, can just use the Case2 creators (as the GP noa1 solver raws)
        if N_z==0 && N_e==0
            TemptationMatrix=CreateReturnFnMatrix_Case2_Disc_noz(TemptationFn, n_d, n_a2, d_gridvals, a2_gridvals, TemptationFnParamsVec);
            MostTempting=reshape(max(TemptationMatrix,[],1),[N_a,1]);
            MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        elseif N_z==0 && N_e>0
            TemptationMatrix=CreateReturnFnMatrix_Case2_Disc(TemptationFn, n_d, n_a2, vfoptions.n_e, d_gridvals, a2_gridvals, vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec); % Because no z, can treat e like z
            MostTempting=reshape(max(TemptationMatrix,[],1),[N_a,N_e]);
            MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        elseif N_z>0 && N_e==0
            TemptationMatrix=CreateReturnFnMatrix_Case2_Disc(TemptationFn, n_d, n_a2, n_z, d_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), TemptationFnParamsVec);
            MostTempting=reshape(max(TemptationMatrix,[],1),[N_a,N_z]);
            MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        else
            TemptationMatrix=CreateReturnFnMatrix_Case2_Disc_e(TemptationFn, n_d, n_a2, n_z, vfoptions.n_e, d_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec);
            MostTempting=reshape(max(TemptationMatrix,[],1),[N_a,N_z,N_e]);
            MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        end
    else
        % Level=0 creators put the joint (d,a1prime) choice set in the first dim (as the GP solver raws)
        if N_z==0 && N_e==0
            TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(TemptationFn, n_d1, n_d2, n_a1, n_a1, n_a2, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, TemptationFnParamsVec,0,0); % Level=0, Refine=0
            MostTempting=reshape(max(TemptationMatrix,[],1),[N_a,1]);
            MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        elseif N_z==0 && N_e>0
            TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, n_d1, n_d2, n_a1, n_a1, n_a2, vfoptions.n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,0,0); % Level=0, Refine=0; Because no z, can treat e like z
            MostTempting=reshape(max(TemptationMatrix,[],1),[N_a,N_e]);
            MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        elseif N_z>0 && N_e==0
            TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc(TemptationFn, n_d1, n_d2, n_a1, n_a1, n_a2, n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), TemptationFnParamsVec,0,0); % Level=0, Refine=0
            MostTempting=reshape(max(TemptationMatrix,[],1),[N_a,N_z]);
            MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        else
            TemptationMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(TemptationFn, n_d1, n_d2, n_a1, n_a1, n_a2, n_z, vfoptions.n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,0,0); % Level=0, Refine=0
            MostTempting=reshape(max(TemptationMatrix,[],1),[N_a,N_z,N_e]);
            MostTempting(MostTempting==-Inf)=0; % a state where EVERY choice is infeasible leaves this -Inf: V there must be -Inf (as in the standard solver), not -Inf-(-Inf)=NaN
        end
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

        % Step 3: build EVnext indexed by (anext, z_from) -- integrate over e' (iid) then over z' (markov)
        if N_z==0 && N_e==0
            EVnext=V(:,jj+1); % [N_a]
        elseif N_z==0 && N_e>0
            % e iid: integrate over e' to collapse to EVnext indexed by anext only
            EVnext=sum(V(:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -1), 2); % [N_a, 1]
        elseif N_z>0 && N_e==0
            % EVnext(anext, z_from) = sum_{z_to} pi(z_from, z_to) * V(anext, z_to)
            EVnext=V(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        else
            % Integrate over e' then over z'
            EVnext=sum(V(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3); % [N_a, N_z, 1]
            EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        end

        % Step 4: interpolated lookup using a1prime_idx and a2primeIndex/Probs
        % For each current state (a, [z, e]):
        %   aprime_low_kron = a1prime + N_a1 * (a2primeIndex - 1)
        %   aprime_up_kron  = a1prime + N_a1 *  a2primeIndex
        %   EVnext_atpolicy = a2primeProbs * EVnext[aprime_low, ...] + (1-a2primeProbs) * EVnext[aprime_up, ...]
        % In the noa1 case (N_a1==0), aprime_low/up reduce to a2primeIndex/a2primeIndex+1
        % (no a1prime to combine with; a1prime_idx is unused).
        if N_z==0 && N_e==0
            if l_a2==1
                if N_a1==0
                    aprime_low=a2primeIndex;     % [N_a, 1]
                    aprime_up =a2primeIndex+1;
                else
                    a1p=a1prime_idx(:,jj); % [N_a, 1]
                    aprime_low=a1p+N_a1*(a2primeIndex-1);
                    aprime_up =a1p+N_a1*(a2primeIndex);
                end
                EV_low=EVnext(aprime_low);
                EV_up =EVnext(aprime_up);
                EVnext_atpolicy=a2primeProbs.*EV_low+(1-a2primeProbs).*EV_up; % [N_a, 1]
            else
                % l_a2==2: a2primeIndex/a2primeProbs are [N_a, l_a2] per-dim factored.
                % Nested 2-corner with skipinterp at each level for bit-exactness when V flat.
                n_a2_1=n_a2(1);
                loIdx_1=a2primeIndex(:,1); % [N_a, 1]
                loIdx_2=a2primeIndex(:,2);
                prob_1=a2primeProbs(:,1);
                prob_2=a2primeProbs(:,2);

                if N_a1==0
                    a1p=ones(N_a,1,'gpuArray'); N_a1_eff=1; % the degenerate a1 dimension has the single index 1,
                    % not 0: with a1p=0 the aprime formula below returns a2kron-1, which is 0 at the
                    % first grid point and so is not a valid subscript
                else
                    a1p=a1prime_idx(:,jj); N_a1_eff=N_a1;
                end
                aprime_ll=a1p+N_a1_eff*(loIdx_1+n_a2_1*(loIdx_2-1)-1);
                aprime_hl=a1p+N_a1_eff*((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1);
                aprime_lh=a1p+N_a1_eff*(loIdx_1+n_a2_1*loIdx_2-1);
                aprime_hh=a1p+N_a1_eff*((loIdx_1+1)+n_a2_1*loIdx_2-1);
                V_ll=EVnext(aprime_ll);
                V_hl=EVnext(aprime_hl);
                V_lh=EVnext(aprime_lh);
                V_hh=EVnext(aprime_hh);

                p1_loy=prob_1; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EVnext_atpolicy=c_loy+c_hiy; % [N_a, 1]
            end
            V(:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*EVnext_atpolicy;
        elseif N_z==0 && N_e>0
            % a1prime_idx, a2primeIndex shape: [N_a, N_e]; EVnext shape: [N_a, 1]
            if l_a2==1
                if N_a1==0
                    aprime_low=a2primeIndex;
                    aprime_up =a2primeIndex+1;
                else
                    a1p=a1prime_idx(:,:,jj); % [N_a, N_e]
                    aprime_low=a1p+N_a1*(a2primeIndex-1);
                    aprime_up =a1p+N_a1*(a2primeIndex);
                end
                EV_low=reshape(EVnext(aprime_low(:)),[N_a,N_e]);
                EV_up =reshape(EVnext(aprime_up(:)), [N_a,N_e]);
                EVnext_atpolicy=a2primeProbs.*EV_low+(1-a2primeProbs).*EV_up;
            else
                % l_a2==2: a2primeIndex/a2primeProbs are [N_a,l_a2,N_e] per-dim factored.
                % Nested 2-corner with skipinterp at each level, as in the no-shock branch above.
                n_a2_1=n_a2(1);
                loIdx_1=reshape(a2primeIndex(:,1,:),[N_a,N_e]);
                loIdx_2=reshape(a2primeIndex(:,2,:),[N_a,N_e]);
                prob_1=reshape(a2primeProbs(:,1,:),[N_a,N_e]);
                prob_2=reshape(a2primeProbs(:,2,:),[N_a,N_e]);
                if N_a1==0
                    a1p=ones(N_a,N_e,'gpuArray'); N_a1_eff=1;
                else
                    a1p=a1prime_idx(:,:,jj); N_a1_eff=N_a1;
                end
                aprime_ll=a1p+N_a1_eff*(loIdx_1+n_a2_1*(loIdx_2-1)-1);
                aprime_hl=a1p+N_a1_eff*((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1);
                aprime_lh=a1p+N_a1_eff*(loIdx_1+n_a2_1*loIdx_2-1);
                aprime_hh=a1p+N_a1_eff*((loIdx_1+1)+n_a2_1*loIdx_2-1);
                V_ll=reshape(EVnext(aprime_ll(:)),[N_a,N_e]);
                V_hl=reshape(EVnext(aprime_hl(:)),[N_a,N_e]);
                V_lh=reshape(EVnext(aprime_lh(:)),[N_a,N_e]);
                V_hh=reshape(EVnext(aprime_hh(:)),[N_a,N_e]);
                p1_loy=prob_1; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EVnext_atpolicy=c_loy+c_hiy;
            end
            V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*EVnext_atpolicy;
        elseif N_z>0 && N_e==0
            % a1prime_idx, a2primeIndex shape: [N_a, N_z]; EVnext shape: [N_a, N_z]
            if l_a2==1
                if N_a1==0
                    aprime_low=a2primeIndex;
                    aprime_up =a2primeIndex+1;
                else
                    a1p=a1prime_idx(:,:,jj);
                    aprime_low=a1p+N_a1*(a2primeIndex-1);
                    aprime_up =a1p+N_a1*(a2primeIndex);
                end
                zidxoffset=N_a*gpuArray(0:N_z-1); % [1, N_z]
                lin_low=aprime_low+zidxoffset; % broadcast: [N_a, N_z]
                lin_up =aprime_up +zidxoffset;
                EV_low=reshape(EVnext(lin_low(:)),[N_a,N_z]);
                EV_up =reshape(EVnext(lin_up(:)), [N_a,N_z]);
                EVnext_atpolicy=a2primeProbs.*EV_low+(1-a2primeProbs).*EV_up;
            else
                % l_a2==2: a2primeIndex/a2primeProbs are [N_a,l_a2,N_z] per-dim factored.
                % Nested 2-corner with skipinterp at each level, as in the no-shock branch above.
                n_a2_1=n_a2(1);
                loIdx_1=reshape(a2primeIndex(:,1,:),[N_a,N_z]);
                loIdx_2=reshape(a2primeIndex(:,2,:),[N_a,N_z]);
                prob_1=reshape(a2primeProbs(:,1,:),[N_a,N_z]);
                prob_2=reshape(a2primeProbs(:,2,:),[N_a,N_z]);
                if N_a1==0
                    a1p=ones(N_a,N_z,'gpuArray'); N_a1_eff=1;
                else
                    a1p=a1prime_idx(:,:,jj); N_a1_eff=N_a1;
                end
                aprime_ll=a1p+N_a1_eff*(loIdx_1+n_a2_1*(loIdx_2-1)-1);
                aprime_hl=a1p+N_a1_eff*((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1);
                aprime_lh=a1p+N_a1_eff*(loIdx_1+n_a2_1*loIdx_2-1);
                aprime_hh=a1p+N_a1_eff*((loIdx_1+1)+n_a2_1*loIdx_2-1);
                zidxoffset=N_a*gpuArray(0:N_z-1); % [1, N_z], broadcasts over [N_a, N_z]
                lin_ll=aprime_ll+zidxoffset; lin_hl=aprime_hl+zidxoffset;
                lin_lh=aprime_lh+zidxoffset; lin_hh=aprime_hh+zidxoffset;
                V_ll=reshape(EVnext(lin_ll(:)),[N_a,N_z]);
                V_hl=reshape(EVnext(lin_hl(:)),[N_a,N_z]);
                V_lh=reshape(EVnext(lin_lh(:)),[N_a,N_z]);
                V_hh=reshape(EVnext(lin_hh(:)),[N_a,N_z]);
                p1_loy=prob_1; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EVnext_atpolicy=c_loy+c_hiy;
            end
            V(:,:,jj)=FofPolicy_jj+TofPolicy_jj-MostTempting+beta*EVnext_atpolicy;
        else
            % a1prime_idx, a2primeIndex shape: [N_a, N_z*N_e]; EVnext shape: [N_a, N_z]
            % For each (a, z, e), look up at aprime in dim 1 of EVnext, and z (=current state's z) in dim 2.
            if l_a2==1
                a2pIdx=reshape(a2primeIndex,[N_a, N_z, N_e]);
                a2pPrb=reshape(a2primeProbs,[N_a, N_z, N_e]);
                if N_a1==0
                    aprime_low=a2pIdx;
                    aprime_up =a2pIdx+1;
                else
                    a1p=reshape(a1prime_idx(:,:,jj),[N_a, N_z, N_e]);
                    aprime_low=a1p+N_a1*(a2pIdx-1);
                    aprime_up =a1p+N_a1*(a2pIdx);
                end
                zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]); % [1, N_z, 1]
                lin_low=aprime_low+zidxoffset;
                lin_up =aprime_up +zidxoffset;
                EV_low=reshape(EVnext(lin_low(:)),[N_a,N_z,N_e]);
                EV_up =reshape(EVnext(lin_up(:)), [N_a,N_z,N_e]);
                EVnext_atpolicy=a2pPrb.*EV_low+(1-a2pPrb).*EV_up;
            else
                % l_a2==2: a2primeIndex/a2primeProbs are [N_a,l_a2,N_z*N_e] per-dim factored.
                % Nested 2-corner with skipinterp at each level, as in the no-shock branch above.
                n_a2_1=n_a2(1);
                a2pIdx=reshape(a2primeIndex,[N_a,l_a2,N_z,N_e]);
                a2pPrb=reshape(a2primeProbs,[N_a,l_a2,N_z,N_e]);
                loIdx_1=reshape(a2pIdx(:,1,:,:),[N_a,N_z,N_e]);
                loIdx_2=reshape(a2pIdx(:,2,:,:),[N_a,N_z,N_e]);
                prob_1=reshape(a2pPrb(:,1,:,:),[N_a,N_z,N_e]);
                prob_2=reshape(a2pPrb(:,2,:,:),[N_a,N_z,N_e]);
                if N_a1==0
                    a1p=ones(N_a,N_z,N_e,'gpuArray'); N_a1_eff=1;
                else
                    a1p=reshape(a1prime_idx(:,:,jj),[N_a,N_z,N_e]); N_a1_eff=N_a1;
                end
                aprime_ll=a1p+N_a1_eff*(loIdx_1+n_a2_1*(loIdx_2-1)-1);
                aprime_hl=a1p+N_a1_eff*((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1);
                aprime_lh=a1p+N_a1_eff*(loIdx_1+n_a2_1*loIdx_2-1);
                aprime_hh=a1p+N_a1_eff*((loIdx_1+1)+n_a2_1*loIdx_2-1);
                zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]); % [1, N_z, 1]
                lin_ll=aprime_ll+zidxoffset; lin_hl=aprime_hl+zidxoffset;
                lin_lh=aprime_lh+zidxoffset; lin_hh=aprime_hh+zidxoffset;
                V_ll=reshape(EVnext(lin_ll(:)),[N_a,N_z,N_e]);
                V_hl=reshape(EVnext(lin_hl(:)),[N_a,N_z,N_e]);
                V_lh=reshape(EVnext(lin_lh(:)),[N_a,N_z,N_e]);
                V_hh=reshape(EVnext(lin_hh(:)),[N_a,N_z,N_e]);
                p1_loy=prob_1; p1_loy(V_ll==V_hl)=0;
                c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1; p1_hiy(V_lh==V_hh)=0;
                c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EVnext_atpolicy=c_loy+c_hiy;
            end
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
