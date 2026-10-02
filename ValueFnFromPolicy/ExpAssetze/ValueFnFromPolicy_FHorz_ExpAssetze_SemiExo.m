function varargout=ValueFnFromPolicy_FHorz_ExpAssetze_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model has experienceassetze (vfoptions.experienceassetze>=1)
% AND semi-exogenous shocks (n_semiz>0).
%
% Combines:
%   - ExpAssetze: a2 is the experience asset; a2prime is determined by aprimeFn(d_expasset,a2,z,e,...)
%     where z is Markov (REQUIRED) and e is iid start-of-period (REQUIRED). aprimeFn does not depend on semiz.
%   - SemiExo: a semi-exogenous state semiz whose transition pi_semiz depends on the policy's d_semiz
%     choice (the last l_dsemiz components of d).
%
% Convention on d ordering with semiz: d = [...other d..., d_expasset, d_semiz]. d_semiz is the
% last l_dsemiz components; d_expasset is the l_d2 components immediately before them.

%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    V=ValueFnFromPolicy_FHorz_ExpAssetze_SemiExo_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%% Setup
% Semiz gridvals + pi_semiz_J
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end
% z gridvals
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3,0);

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetze you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;

n_semiz=vfoptions.n_semiz;
N_semiz=prod(n_semiz);

if isfield(vfoptions,'l_dsemiz')
    l_dsemiz=vfoptions.l_dsemiz;
else
    l_dsemiz=1;
end
n_dsemiz=n_d(end-l_dsemiz+1:end);
N_dsemiz=prod(n_dsemiz);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_z==0 || N_e==0
    error('ValueFnFromPolicy_FHorz_ExpAssetze_SemiExo: experienceassetze requires both N_z>0 and N_e>0 (aprimeFn depends on z and e)')
end
if N_d==0
    error('ValueFnFromPolicy_FHorz_ExpAssetze_SemiExo: experienceassetze+semiz requires at least one decision variable')
end
l_d=length(n_d);
l_z=length(n_z);
l_e=length(vfoptions.n_e);

% Split a into a1 (standard) and a2 (experience asset)
l_a2=vfoptions.experienceassetze; % l_a2 = number of a2 (experience-asset) dims
if length(n_a)<=l_a2
    % noa1: the experience asset(s) are the only endogenous states
    n_a1=0;
    N_a1=1; % so aprime_low_e=a1p_e+N_a1*(a2pIdx_e-1) reduces to a2pIdx_e (a1prime_idx stays 1)
    l_a1=0; % Policy contains only the d channels
else
    n_a1=n_a(1:end-l_a2);
    N_a1=prod(n_a1);
    l_a1=length(n_a1);
end
n_a2=n_a(end-l_a2+1:end); % the last l_a2 dims are the experience asset(s)
a2_grid=a_grid(sum(n_a1)+1:end);

% Which d drives the experience asset. With semiz, d ordering is [...other, d_expasset, d_semiz];
% the last l_dsemiz are for semiz, the l_d2 immediately before them drive the expasset.
if isfield(vfoptions,'l_dexperienceassetze')
    l_d2=vfoptions.l_dexperienceassetze;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_dsemiz-l_d2+1):(l_d-l_dsemiz);

% aprimeFnParamNames: leading inputs are (d_expasset..., a2, z..., e...)
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_z+l_e+(l_a2>=2))  % the (l_a2>=2) term is the 'whicha' selector slot, which aprimeFn only takes when there are two experience assets
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+l_e+(l_a2>=2)+1:end}}; % the first inputs are (d2,a2,z,e), plus the 'whicha' selector when l_a2>=2
else
    aprimeFnParamNames={};
end

% Joint Markov-like shock = [semiz, z] (semiz fastest); e separate
n_shocks=[n_semiz,n_z];
N_shocks=N_semiz*N_z;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

%% PolicyValues (PolicyInd2Val_FHorz handles experienceassetze internally; auto-adds n_semiz and n_e)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);

%% Reshape Policy to canonical Kron form
Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_shocks, N_e, N_j]);

%% d_semiz_idx: last l_dsemiz components of d
d_semiz_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
cumprods_dsemiz=[1, cumprod(n_dsemiz(1:end-1))];
for ii=1:l_dsemiz
    comp=shiftdim(Policy_k(l_d-l_dsemiz+ii, :, :, :, :),1);
    d_semiz_idx=d_semiz_idx+cumprods_dsemiz(ii)*(comp-1);
end

%% a1prime joint index across N_a1 dims (positions l_d+1 .. l_d+l_a1)
a1prime_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
for ii=1:l_a1
    comp=shiftdim(Policy_k(l_d+ii, :, :, :, :),1);
    a1prime_idx=a1prime_idx+cumprods_a1(ii)*(comp-1);
end

%% Joint Markov-like shock gridvals for ReturnFn (semiz, z)
joint_gridvals_J=zeros(N_shocks, length(n_semiz)+length(n_z), N_j, 'gpuArray');
for jj=1:N_j
    joint_gridvals_J(:,:,jj)=[repmat(semiz_gridvals_J(:,:,jj),N_z,1), repelem(z_gridvals_J(:,:,jj),N_semiz,1)];
end

%% Backward iteration
V=zeros(N_a, N_shocks, N_e, N_j, 'gpuArray');

[~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_z);

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Step 1: a2primeIndex, a2primeProbs at this age
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    Policy_slice=reshape(Policy_k(:,:,:,:,jj), [l_d+l_a1, N_a, N_shocks*N_e]);
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetze(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, vfoptions.n_e, N_semiz, N_z, N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), aprimeFnParamsVec);
    % shape [N_a, N_semiz*N_z*N_e]

    % Step 2: ReturnFn at policy
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);

    if jj==N_j
        V(:,:,:,jj)=F_jj;
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

        % Step 3a: integrate next-period V over e' (iid)
        V_next=V(:,:,:,jj+1);
        V_next=sum(V_next .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3);
        V_next=reshape(V_next, [N_a, N_shocks]);

        % Step 3b: integrate over z' (markov, does not depend on d_semiz)
        V_next_r=reshape(V_next, [N_a, N_semiz, N_z]);
        EVw=V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2); EVw(isnan(EVw))=0; % a zero weight against an infinite node gives 0*(-Inf)=NaN, so zero the terms BEFORE summing
        EV_after_z=sum(EVw,3);
        EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_z]);

        % Step 3c: for each d_semiz, integrate over semiz' -> EVnext_byd2(a, semiz_from, z_from, d_semiz)
        EVnext_byd2=zeros(N_a, N_semiz, N_z, N_dsemiz, 'gpuArray');
        for d2_c=1:N_dsemiz
            pi_d2c=pi_semiz_J(:,:,d2_c,jj)';
            pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]);
            EVw=EV_after_z .* pi_reshape; EVw(isnan(EVw))=0; % a zero weight against an infinite node gives 0*(-Inf)=NaN, so zero the terms BEFORE summing
            EVd2c=sum(EVw,2);
            EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_z]);
        end

        % Step 4: per-state lookup -- per-e iteration; a2 interpolation + d_semiz indirection
        EVnext_atpolicy=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
        for e_c=1:N_e
            block=(e_c-1)*N_shocks + (1:N_shocks);
            a1p_e=reshape(a1prime_idx(:,:,e_c,jj),[N_a, N_semiz, N_z]); % all ones in the noa1 case (where N_a1=1)
            d2_e =reshape(d_semiz_idx(:,:,e_c,jj),[N_a, N_semiz, N_z]);
            base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1), [N_a, N_semiz, N_z]);
            if l_a2==1
                a2pIdx_e=reshape(a2primeIndex(:,block),[N_a, N_semiz, N_z]);
                a2pPrb_e=reshape(a2primeProbs(:,block),[N_a, N_semiz, N_z]);
                aprime_low_e=a1p_e+N_a1*(a2pIdx_e-1);
                aprime_up_e =a1p_e+N_a1*(a2pIdx_e);
                lo_idx=aprime_low_e+base_off;
                up_idx=aprime_up_e +base_off;
                EV_lo=reshape(EVnext_byd2(lo_idx(:)),[N_a, N_semiz, N_z]);
                EV_up=reshape(EVnext_byd2(up_idx(:)),[N_a, N_semiz, N_z]);
                EVnext_atpolicy(:,:,:,e_c)=a2pPrb_e.*EV_lo+(1-a2pPrb_e).*EV_up;
            else
                % l_a2==2: a2primeIndex/a2primeProbs are [N_a,l_a2,N_shocks*N_e] per-dim factored.
                % Nested 2-corner interpolation with skipinterp at each level (mirrors ValueFnFromPolicy_FHorz_ExpAsset_SemiExo).
                n_a2_1=n_a2(1);
                loIdx_1=reshape(a2primeIndex(:,1,block),[N_a, N_semiz, N_z]);
                loIdx_2=reshape(a2primeIndex(:,2,block),[N_a, N_semiz, N_z]);
                prob_1=reshape(a2primeProbs(:,1,block),[N_a, N_semiz, N_z]);
                prob_2=reshape(a2primeProbs(:,2,block),[N_a, N_semiz, N_z]);
                aprime_ll=a1p_e+N_a1*(loIdx_1+n_a2_1*(loIdx_2-1)-1);
                aprime_hl=a1p_e+N_a1*((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1);
                aprime_lh=a1p_e+N_a1*(loIdx_1+n_a2_1*loIdx_2-1);
                aprime_hh=a1p_e+N_a1*((loIdx_1+1)+n_a2_1*loIdx_2-1);
                ll_idx=aprime_ll+base_off; hl_idx=aprime_hl+base_off;
                lh_idx=aprime_lh+base_off; hh_idx=aprime_hh+base_off;
                V_ll=reshape(EVnext_byd2(ll_idx(:)),[N_a, N_semiz, N_z]);
                V_hl=reshape(EVnext_byd2(hl_idx(:)),[N_a, N_semiz, N_z]);
                V_lh=reshape(EVnext_byd2(lh_idx(:)),[N_a, N_semiz, N_z]);
                V_hh=reshape(EVnext_byd2(hh_idx(:)),[N_a, N_semiz, N_z]);
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
                EVnext_atpolicy(:,:,:,e_c)=c_loy+c_hiy;
            end
        end
        V(:,:,:,jj)=F_jj+beta*reshape(EVnext_atpolicy, [N_a, N_shocks, N_e]);
    end
end

%% Reshape V out of Kron form
V=reshape(V, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);



varargout={V};

end
