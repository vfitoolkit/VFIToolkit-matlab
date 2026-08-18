function V=ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Compute V from a given Policy when the model uses Epstein-Zin preferences
% AND has riskyasset (vfoptions.riskyasset==1). Mirrors
% ValueFnFromPolicy_FHorz_RiskyAsset with the Epstein-Zin transform chain:
% V' is transformed by ^ezc5 BEFORE the expectations (one joint
% certainty-equivalent over (u,zprime,eprime); mirroring the EZ riskyasset
% VFI raws, e' is summed first, then z' is collapsed, and then the u-lottery
% over a2prime is applied to the collapsed object), and the
% certainty-equivalent power ^ezc6 is applied pointwise after the per-state
% lookup. The warm-glow of bequests is over the risky asset a2prime (as in
% the EZ riskyasset VFI raws): transformed by ^ezc5 and averaged over the
% policy's a2prime u-lottery before being folded into the sj/ezc8 structure.
% The ezc1-ezc8/sj/warmglow preamble is done by the caller
% (ValueFnFromPolicy_FHorz_EpsteinZin) and passed through.
%
% riskyasset: a2prime = aprimeFn(d2,d3, u) -- depends on iid (between-period) shock u, NOT on current a2.
% u does NOT enter Policy; pi_u integrates u out when computing E[V'|policy].
%
% Note: riskyasset combined with semiz already errors in the caller (that is a later step).

%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    V=ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
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
    error('ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset: riskyasset requires at least one decision variable (driving a2prime)')
end
l_d=length(n_d);

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
a2_grid=a_grid(sum(n_a1)+1:end);

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

    DiscountFactorParamsVec=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec;
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    % If there is a warm-glow, evaluate the warmglowfn over a2prime (the bequest is the risky
    % asset, mirrors the EZ riskyasset VFI raws), then the policy's a2prime u-lottery
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
        % Step 4: u-lottery lookup at the policy, acting on the transformed-and-collapsed object.
        % Mirrors standard riskyasset FromPolicy's skipinterp+isnan so that policies landing on
        % infeasible-on-both-sides next-states give the same finite V here.
        if N_z==0 && N_e==0
            V_nextpre=V(:,jj+1); % [N_a]
            temp=V_nextpre;
            temp(isfinite(V_nextpre))=(ezc4*V_nextpre(isfinite(V_nextpre))).^ezc5(jj);
            temp(V_nextpre==0)=0;
            EVnext=temp; % no shocks other than u, nothing to collapse
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
            a2pPrb(Vlower==Vupper)=0; % skipinterp (on the transformed EV, as the EZ VFI raws)
            EVlott=a2pPrb.*Vlower+(1-a2pPrb).*Vupper;
            EVnextOfPolicy=sum(EVlott .* shiftdim(pi_u,-1), 2); % sum over u -> [N_a, 1]
            EVnextOfPolicy(isnan(EVnextOfPolicy))=0;
        elseif N_z==0 && N_e>0
            V_nextpre=V(:,:,jj+1); % [N_a, N_e]
            temp=V_nextpre;
            temp(isfinite(V_nextpre))=(ezc4*V_nextpre(isfinite(V_nextpre))).^ezc5(jj);
            temp(V_nextpre==0)=0;
            % Integrate over the iid e'
            EVnext=sum(temp .* shiftdim(vfoptions.pi_e_J(:,jj+1), -1), 2); % [N_a, 1]
            EVnext(isnan(EVnext))=0;
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
            a2pPrb(Vlower==Vupper)=0; % skipinterp on pi_e-collapsed transformed EV
            EVlott=a2pPrb.*Vlower+(1-a2pPrb).*Vupper;
            EVnextOfPolicy=sum(EVlott .* shiftdim(pi_u,-2), 3); % sum over u -> [N_a, N_e]
            EVnextOfPolicy(isnan(EVnextOfPolicy))=0;
        elseif N_z>0 && N_e==0
            V_nextpre=V(:,:,jj+1); % [N_a, N_z']
            temp=V_nextpre;
            temp(isfinite(V_nextpre))=(ezc4*V_nextpre(isfinite(V_nextpre))).^ezc5(jj);
            temp(V_nextpre==0)=0;
            % Collapse z': EVnext(aprime, z_from) = sum_{z_to} pi(z_from, z_to) * temp(aprime, z_to)
            EVnext=temp.*shiftdim(pi_z_J(:,:,jj)',-1); % [N_a, N_z_to, N_z_from]
            EVnext(isnan(EVnext))=0; % -Inf times zero transition probability
            EVnext=reshape(sum(EVnext,2),[N_a,N_z]); % [N_a, N_z_from]
            if N_a1==0
                aprime_low=a2primeIndex;     % [N_a, N_z, N_u]
                aprime_up =a2primeIndex+1;
            else
                a1p=a1prime_idx(:,:,jj);
                aprime_low=a1p+N_a1*(a2primeIndex-1); % broadcast -> [N_a, N_z, N_u]
                aprime_up =a1p+N_a1*(a2primeIndex);
            end
            zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]); % offset by the state's own z (z_from)
            Vlower=reshape(EVnext(aprime_low+zidxoffset),[N_a,N_z,N_u]);
            Vupper=reshape(EVnext(aprime_up +zidxoffset),[N_a,N_z,N_u]);
            a2pPrb=a2primeProbs;
            a2pPrb(Vlower==Vupper)=0; % skipinterp (on the z-collapsed transformed EV, as the EZ VFI raws)
            EVlott=a2pPrb.*Vlower+(1-a2pPrb).*Vupper;
            EVnextOfPolicy=sum(EVlott .* shiftdim(pi_u,-2), 3); % sum over u -> [N_a, N_z]
            EVnextOfPolicy(isnan(EVnextOfPolicy))=0;
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
            zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1,1]); % offset by the state's own z (z_from)
            Vlower=reshape(EVnext(aprime_low+zidxoffset),[N_a,N_z,N_e,N_u]);
            Vupper=reshape(EVnext(aprime_up +zidxoffset),[N_a,N_z,N_e,N_u]);
            a2pPrb(Vlower==Vupper)=0; % skipinterp (on the z-collapsed transformed EV, as the EZ VFI raws)
            EVlott=a2pPrb.*Vlower+(1-a2pPrb).*Vupper;
            EVnextOfPolicy=sum(EVlott .* shiftdim(pi_u,-3), 4); % sum over u -> [N_a, N_z, N_e]
            EVnextOfPolicy(isnan(EVnextOfPolicy))=0;
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
