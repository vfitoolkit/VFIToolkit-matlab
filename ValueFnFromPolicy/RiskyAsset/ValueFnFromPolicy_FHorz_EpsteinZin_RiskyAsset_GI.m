function V=ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Compute V from a given Policy when the model uses Epstein-Zin preferences
% AND has riskyasset (vfoptions.riskyasset==1) AND grid interpolation layer
% (vfoptions.gridinterplayer==1). Mirrors ValueFnFromPolicy_FHorz_RiskyAsset_GI
% with the Epstein-Zin transform chain: V' is transformed by ^ezc5 BEFORE the
% expectations (one joint certainty-equivalent over (u,zprime,eprime)); the
% 2x2 corner interpolation weights in (a1prime,a2prime) are applied to the
% TRANSFORMED EV (the established EZ GI convention, so the collapse tests are
% exact against the EZ GI solver), and the certainty-equivalent power ^ezc6
% is applied pointwise after the per-state lookup. The warm-glow of bequests
% is over the risky asset a2prime (as in the EZ riskyasset VFI raws), so it
% is unaffected by the GI layer (which refines a1prime only): it is
% transformed by ^ezc5 and averaged over the policy's a2prime u-lottery
% before being folded into the sj/ezc8 structure. The ezc1-ezc8/sj/warmglow
% preamble is done by the caller (ValueFnFromPolicy_FHorz_EpsteinZin) and
% passed through.
%
% riskyasset: a2prime = aprimeFn(d2,d3, u) -- u is iid between-period shock, NOT current a2.
% Under GI, Policy carries an L2 fine-grid index for a1prime; lookup is 2x2 in (a1,a2), then summed over u with pi_u.

%% Setup
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use riskyasset you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;
if ~isfield(vfoptions,'n_u'),    error('To use riskyasset you must define vfoptions.n_u'),    end
if ~isfield(vfoptions,'u_grid'), error('To use riskyasset you must define vfoptions.u_grid'), end
if ~isfield(vfoptions,'pi_u'),   error('To use riskyasset you must define vfoptions.pi_u'),   end
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
    error('ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset_GI: riskyasset requires at least one decision variable')
end
l_d=length(n_d);

% noa1 case (n_a is scalar -- risky asset is the only endogenous state): GI refines a1, which
% doesn't apply when there's no a1. Fall back to non-GI version (which handles noa1 correctly).
% Matches the upstream VFI convention (noa1 has no GI/DC/DC+GI raw files).
if isscalar(n_a)
    vfoptions.gridinterplayer=0;
    V=ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
    return
end
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
n_a2=n_a(end);
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);

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

if N_z==0 && N_e==0
    N_ze=0;
elseif N_z>0 && N_e==0
    N_ze=N_z;
elseif N_z==0 && N_e>0
    N_ze=N_e;
else
    N_ze=N_z*N_e;
end

n2short=vfoptions.ngridinterp;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues (PolicyInd2Val_FHorz handles riskyasset internally)
% The riskyasset ReturnFn takes (d1, d3, a1prime, a1, a2, ...): d2 (aprimeFn-only) is NOT an input.
% This is the 'refine' split the VFI codes use (vfoptions.refine_d): d1 in ReturnFn only,
% d2 in aprimeFn only, d3 in both -- so keep only the d1 and d3 decision rows (then a1prime).
d1rows=1:vfoptions.refine_d(1);
d3rows=(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1):(vfoptions.refine_d(1)+vfoptions.refine_d(2)+vfoptions.refine_d(3));
returnrows=[d1rows, d3rows, (l_d+1):(l_d+l_a1)]; % ReturnFn d's (d1 then d3), then a1prime
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
PolicyValues=PolicyValues(returnrows,:,:,:); % drop d2 (not in the ReturnFn)
l_daprime=size(PolicyValues,1); % = n_d1 + n_d3 + l_a1
if N_z==0 && N_e==0
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]);
else
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
end

%% Strip trailing L2flag channel if present
size_first=l_d+l_a1+1;
if size(Policy,1) > size_first
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1), prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:size_first,:), [size_first, tempsize(2:end)]);
end

%% Reshape Policy
if N_z==0 && N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_j]);
elseif N_z>0 && N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_z, N_j]);
elseif N_z==0 && N_e>0
    Policy_k=reshape(Policy,[size_first, N_a, N_e, N_j]);
else
    Policy_k=reshape(Policy,[size_first, N_a, N_z*N_e, N_j]);
end

%% Extract a1prime midpoint (lower) and L2
a1_mid=shiftdim(Policy_k(l_d+1,:,:,:),1);
L2=shiftdim(Policy_k(l_d+l_a1+1,:,:,:),1);
w_a1_upper=(L2-1)/(n2short+1);
w_a1_lower=1-w_a1_upper;
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
a1_lower=a1_mid;
for ii=2:l_a1
    comp=shiftdim(Policy_k(l_d+ii,:,:,:),1);
    a1_lower=a1_lower+cumprods_a1(ii)*(comp-1);
end
a1_upper=a1_lower+1;
a1_top_clamp=(a1_mid>=n_a1(1));
a1_upper(a1_top_clamp)=a1_lower(a1_top_clamp);

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
        Policy_slice=Policy_k(:,:,jj);
    else
        Policy_slice=Policy_k(:,:,:,jj);
    end

    % Step 1: a2primeIndex, a2primeProbs -- helper adds the u dim (riskyasset: aprime ignores a2)
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyRiskyAsset(Policy_slice, aprimeFn, whichisdforriskyasset, n_d, n_a1, n_a2, N_ze, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    % shape: N_z==0 && N_e==0 -> [N_a, N_u]; else -> [N_a, N_ze, N_u]

    % Step 2: ReturnFn at policy
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

    DiscountFactorParamsVec=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec;
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    % If there is a warm-glow, evaluate the warmglowfn over a2prime (the bequest is the risky
    % asset, mirrors the EZ riskyasset VFI raws; a2prime has no GI fine grid, so this block is
    % identical to the non-GI version), then the policy's a2prime u-lottery
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        % u-lottery of the transformed warm-glow at the policy's a2prime
        WGlower=WGmatrix(a2primeIndex);   % same shape as a2primeIndex
        WGupper=WGmatrix(a2primeIndex+1);
        wgPrb=a2primeProbs;
        wgPrb(WGlower==WGupper)=0; % skipinterp
        WGlott=wgPrb.*WGlower+(1-wgPrb).*WGupper;
        if N_z==0 && N_e==0
            WGofPolicy=sum(WGlott .* shiftdim(pi_u,-1), 2); % sum over u -> [N_a, 1]
        else
            WGofPolicy=sum(WGlott .* shiftdim(pi_u,-2), 3); % sum over u -> [N_a, N_ze]
        end
        WGofPolicy(isnan(WGofPolicy))=0;
        if N_z>0 && N_e>0
            WGofPolicy=reshape(WGofPolicy,[N_a,N_z,N_e]);
        end
    end

    if jj==N_j
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        Vjj=F_jj;
        becareful=logical(isfinite(F_jj).*(F_jj~=0)); % finite but not zero
        Vjj(becareful)=(ezc1*F_jj(becareful).^ezc2(N_j)).^ezc7(N_j);
        Vjj(F_jj==0)=-Inf;
        if warmglow==1
            becareful2=(WGofPolicy==0);
            WGofPolicy(isfinite(WGofPolicy))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGofPolicy(isfinite(WGofPolicy)).^ezc8(N_j)).^ezc6(N_j));
            WGofPolicy(becareful2)=0;
            Vjj=Vjj+WGofPolicy;
        end
        if N_z==0 && N_e==0
            V(:,jj)=Vjj;
        elseif N_z==0 && N_e>0
            V(:,:,jj)=Vjj;
        elseif N_z>0 && N_e==0
            V(:,:,jj)=Vjj;
        else
            V(:,:,:,jj)=Vjj;
        end
    else
        % Step 3 (EZ): transform V' by ^ezc5 BEFORE all the sums (one joint certainty-equivalent
        % over (u,zprime,eprime)), then collapse e' (iid) and z' (markov, elementwise with isnan
        % clear, mirrors the EZ riskyasset VFI raws)
        % Step 4: 2x2 corner interpolation in (a1prime,a2prime) + sum over u with pi_u, all acting
        % on the TRANSFORMED EV (the established EZ GI convention), then ^ezc6 pointwise after.
        if N_z==0 && N_e==0
            V_nextpre=V(:,jj+1); % [N_a]
            temp=V_nextpre;
            temp(isfinite(V_nextpre))=(ezc4*V_nextpre(isfinite(V_nextpre))).^ezc5(jj);
            temp(V_nextpre==0)=0;
            EVnext=temp; % no shocks other than u, nothing to collapse
            % a1l, a1u, wa1l, wa1u: [N_a]; a2primeIndex/Probs: [N_a, N_u]
            a1l=a1_lower(:,jj); a1u=a1_upper(:,jj);
            wa1l=w_a1_lower(:,jj); wa1u=w_a1_upper(:,jj);
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            % Broadcast a1l/u (N_a) over u -> [N_a, N_u]
            EV_LL=reshape(EVnext(a1l+N_a1*(a2l-1)),[N_a,N_u]);
            EV_LU=reshape(EVnext(a1l+N_a1*(a2u-1)),[N_a,N_u]);
            EV_UL=reshape(EVnext(a1u+N_a1*(a2l-1)),[N_a,N_u]);
            EV_UU=reshape(EVnext(a1u+N_a1*(a2u-1)),[N_a,N_u]);
            per_u=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            EVnextOfPolicy=sum(per_u .* shiftdim(pi_u,-1), 2); % sum over u -> [N_a, 1]
            EVnextOfPolicy(isnan(EVnextOfPolicy))=0; % zero corner weights times -Inf next-states
        elseif N_z==0 && N_e>0
            V_nextpre=V(:,:,jj+1); % [N_a, N_e]
            temp=V_nextpre;
            temp(isfinite(V_nextpre))=(ezc4*V_nextpre(isfinite(V_nextpre))).^ezc5(jj);
            temp(V_nextpre==0)=0;
            % Integrate over the iid e'
            EVnext=sum(temp .* shiftdim(vfoptions.pi_e_J(:,jj+1), -1), 2); % [N_a, 1]
            EVnext(isnan(EVnext))=0;
            % a1l/u, wa1l/u: [N_a, N_e]; a2primeIndex/Probs: [N_a, N_e, N_u]
            a1l=a1_lower(:,:,jj); a1u=a1_upper(:,:,jj);
            wa1l=w_a1_lower(:,:,jj); wa1u=w_a1_upper(:,:,jj);
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            lin_LL=a1l+N_a1*(a2l-1); lin_LU=a1l+N_a1*(a2u-1);
            lin_UL=a1u+N_a1*(a2l-1); lin_UU=a1u+N_a1*(a2u-1);
            EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_e,N_u]);
            EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_e,N_u]);
            EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_e,N_u]);
            EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_e,N_u]);
            per_u=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            EVnextOfPolicy=sum(per_u .* shiftdim(pi_u,-2), 3); % sum over u -> [N_a, N_e]
            EVnextOfPolicy(isnan(EVnextOfPolicy))=0; % zero corner weights times -Inf next-states
        elseif N_z>0 && N_e==0
            V_nextpre=V(:,:,jj+1); % [N_a, N_z']
            temp=V_nextpre;
            temp(isfinite(V_nextpre))=(ezc4*V_nextpre(isfinite(V_nextpre))).^ezc5(jj);
            temp(V_nextpre==0)=0;
            % Collapse z': EVnext(aprime, z_from) = sum_{z_to} pi(z_from, z_to) * temp(aprime, z_to)
            EVnext=temp.*shiftdim(pi_z_J(:,:,jj)',-1); % [N_a, N_z_to, N_z_from]
            EVnext(isnan(EVnext))=0; % -Inf times zero transition probability
            EVnext=reshape(sum(EVnext,2),[N_a,N_z]); % [N_a, N_z_from]
            % a1l/u, wa1l/u: [N_a, N_z]; a2primeIndex/Probs: [N_a, N_z, N_u]
            a1l=a1_lower(:,:,jj); a1u=a1_upper(:,:,jj);
            wa1l=w_a1_lower(:,:,jj); wa1u=w_a1_upper(:,:,jj);
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]);
            lin_LL=a1l+N_a1*(a2l-1)+zidxoffset; lin_LU=a1l+N_a1*(a2u-1)+zidxoffset;
            lin_UL=a1u+N_a1*(a2l-1)+zidxoffset; lin_UU=a1u+N_a1*(a2u-1)+zidxoffset;
            EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_z,N_u]);
            EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_z,N_u]);
            EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_z,N_u]);
            EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_z,N_u]);
            per_u=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            EVnextOfPolicy=sum(per_u .* shiftdim(pi_u,-2), 3); % sum over u -> [N_a, N_z]
            EVnextOfPolicy(isnan(EVnextOfPolicy))=0; % zero corner weights times -Inf next-states
        else
            V_nextpre=V(:,:,:,jj+1); % [N_a, N_z', N_e']
            temp=V_nextpre;
            temp(isfinite(V_nextpre))=(ezc4*V_nextpre(isfinite(V_nextpre))).^ezc5(jj);
            temp(V_nextpre==0)=0;
            % Integrate over the iid e' first, then collapse z' (mirrors the EZ riskyasset VFI raws)
            EVnext=sum(temp .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3); % [N_a, N_z_to]
            EVnext=EVnext.*shiftdim(pi_z_J(:,:,jj)',-1); % [N_a, N_z_to, N_z_from]
            EVnext(isnan(EVnext))=0; % -Inf times zero transition probability
            EVnext=reshape(sum(EVnext,2),[N_a,N_z]); % [N_a, N_z_from]
            % a1l/u, wa1l/u: [N_a, N_z*N_e] flat -> [N_a, N_z, N_e]
            % a2primeIndex/Probs: [N_a, N_z*N_e, N_u] -> [N_a, N_z, N_e, N_u]
            a1l=reshape(a1_lower(:,:,jj),[N_a,N_z,N_e]);  a1u=reshape(a1_upper(:,:,jj),[N_a,N_z,N_e]);
            wa1l=reshape(w_a1_lower(:,:,jj),[N_a,N_z,N_e]); wa1u=reshape(w_a1_upper(:,:,jj),[N_a,N_z,N_e]);
            a2l=reshape(a2primeIndex,[N_a,N_z,N_e,N_u]); a2u=a2l+1;
            wa2l=reshape(a2primeProbs,[N_a,N_z,N_e,N_u]); wa2u=1-wa2l;
            zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1,1]);
            lin_LL=a1l+N_a1*(a2l-1)+zidxoffset; lin_LU=a1l+N_a1*(a2u-1)+zidxoffset;
            lin_UL=a1u+N_a1*(a2l-1)+zidxoffset; lin_UU=a1u+N_a1*(a2u-1)+zidxoffset;
            EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_z,N_e,N_u]);
            EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_z,N_e,N_u]);
            EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_z,N_e,N_u]);
            EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_z,N_e,N_u]);
            per_u=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            EVnextOfPolicy=sum(per_u .* shiftdim(pi_u,-3), 4); % sum over u -> [N_a, N_z, N_e]
            EVnextOfPolicy(isnan(EVnextOfPolicy))=0; % zero corner weights times -Inf next-states
        end

        % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise at the policy
        temp4=EVnextOfPolicy;
        if warmglow==1
            becareful=logical(isfinite(temp4).*isfinite(WGofPolicy)); % both are finite
            temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGofPolicy(becareful).^ezc8(jj)).^ezc6(jj);
            temp4((EVnextOfPolicy==0)&(WGofPolicy==0))=0; % Is actually zero
        else % not using warmglow
            temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
            temp4(EVnextOfPolicy==0)=0;
        end

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(F_jj).*(F_jj~=0)); % finite but not zero
        temp2=F_jj;
        temp2(becareful)=F_jj(becareful).^ezc2(jj);
        temp2(F_jj==0)=-Inf;

        Vjj=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

        temp5=logical(isfinite(Vjj).*(Vjj~=0));
        Vjj(temp5)=Vjj(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
        Vjj(Vjj==0)=-Inf;

        if N_z==0 && N_e==0
            V(:,jj)=Vjj;
        elseif N_z==0 && N_e>0
            V(:,:,jj)=Vjj;
        elseif N_z>0 && N_e==0
            V(:,:,jj)=Vjj;
        else
            V(:,:,:,jj)=Vjj;
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
