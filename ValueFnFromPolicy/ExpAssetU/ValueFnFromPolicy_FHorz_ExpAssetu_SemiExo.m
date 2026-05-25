function V=ValueFnFromPolicy_FHorz_ExpAssetu_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model has experienceassetu (vfoptions.experienceassetu==1)
% AND semi-exogenous shocks (n_semiz>0).
%
% Combines:
%   - ExpAssetu: a2 is the experience asset; a2prime is determined by aprimeFn(d_expasset,a2,u,...) with
%     iid between-period shock u. u does NOT enter Policy; pi_u integrates u out in the EV lookup.
%   - SemiExo: a semi-exogenous state semiz whose transition pi_semiz depends on the policy's d_semiz
%     choice (the last l_dsemiz components of d).
%
% Convention on d ordering with semiz: d = [...other d..., d_expasset, d_semiz]. d_semiz is the
% last l_dsemiz components; d_expasset is the l_d2 components immediately before them.

%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    V=ValueFnFromPolicy_FHorz_ExpAssetu_SemiExo_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

%% Setup
% Semiz gridvals + pi_semiz_J
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end
% z gridvals
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetu you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;
if ~isfield(vfoptions,'n_u'),    error('To use experienceassetu you must define vfoptions.n_u'),    end
if ~isfield(vfoptions,'u_grid'), error('To use experienceassetu you must define vfoptions.u_grid'), end
if ~isfield(vfoptions,'pi_u'),   error('To use experienceassetu you must define vfoptions.pi_u'),   end
n_u=vfoptions.n_u;
u_grid=gpuArray(vfoptions.u_grid);
pi_u=gpuArray(vfoptions.pi_u);
N_u=prod(n_u);
l_u=length(n_u);

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
if N_d==0
    error('ValueFnFromPolicy_FHorz_ExpAssetu_SemiExo: experienceassetu+semiz requires at least one decision variable')
end
l_d=length(n_d);
l_a=length(n_a);

% Split a into a1 (standard) and a2 (experience asset)
if isscalar(n_a)
    error('ValueFnFromPolicy_FHorz_ExpAssetu_SemiExo: case with no a1 (experience asset as only asset) not yet implemented')
end
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
n_a2=n_a(end);
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);
l_aprime=l_a1; % Policy stores a1prime only (a2prime implicit)

% Which d drives the experience asset. With semiz, d ordering is [...other, d_expasset, d_semiz];
% the last l_dsemiz are for semiz, the l_d2 immediately before them drive the expasset.
if isfield(vfoptions,'l_dexperienceassetu')
    l_d2=vfoptions.l_dexperienceassetu;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_dsemiz-l_d2+1):(l_d-l_dsemiz);
n_d2=n_d(whichisdforexpasset);

% aprimeFnParamNames
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_u)
    aprimeFnParamNames={temp{l_d2+l_a2+l_u+1:end}};
else
    aprimeFnParamNames={};
end

% Treat (semiz, z) as joint shock for PolicyInd2Val and ReturnFn evaluation
if N_z==0
    n_shocks=n_semiz;
else
    n_shocks=[n_semiz,n_z];
end
N_shocks=N_semiz*max(N_z,1);

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

%% PolicyValues (PolicyInd2Val_FHorz handles experienceassetu internally -- drops a2prime, auto-adds n_semiz and n_e)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
% PolicyValues shape:
% - N_e==0: [l_daprime, N_a, N_shocks, N_j]
% - N_e>0:  [l_daprime, N_a, N_shocks*N_e, N_j]
if N_e==0
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
else
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
end

%% Reshape Policy to canonical Kron form
if N_e==0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_shocks, N_j]);
else
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_shocks, N_e, N_j]);
end

%% d_semiz_idx: last l_dsemiz components of d
if N_e==0
    d_semiz_idx=ones(N_a,N_shocks,N_j,'gpuArray');
else
    d_semiz_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
end
cumprods_dsemiz=[1, cumprod(n_dsemiz(1:end-1))];
for ii=1:l_dsemiz
    comp=shiftdim(Policy_k(l_d-l_dsemiz+ii, :, :, :, :),1);
    d_semiz_idx=d_semiz_idx+cumprods_dsemiz(ii)*(comp-1);
end

%% a1prime joint index across N_a1 dims (positions l_d+1 .. l_d+l_a1)
if N_e==0
    a1prime_idx=ones(N_a,N_shocks,N_j,'gpuArray');
else
    a1prime_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
end
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
for ii=1:l_a1
    comp=shiftdim(Policy_k(l_d+ii, :, :, :, :),1);
    a1prime_idx=a1prime_idx+cumprods_a1(ii)*(comp-1);
end

%% Joint shock gridvals for ReturnFn
if N_z==0
    joint_gridvals_J=semiz_gridvals_J;
else
    joint_gridvals_J=zeros(N_shocks, length(n_semiz)+length(n_z), N_j, 'gpuArray');
    for jj=1:N_j
        joint_gridvals_J(:,:,jj)=[repmat(semiz_gridvals_J(:,:,jj),N_z,1), repelem(z_gridvals_J(:,:,jj),N_semiz,1)];
    end
end

%% Backward iteration
if N_e==0
    V=zeros(N_a, N_shocks, N_j, 'gpuArray');
else
    V=zeros(N_a, N_shocks, N_e, N_j, 'gpuArray');
end

[~, SZ_grid_noz]=ndgrid(1:N_a, 1:N_semiz);
if N_z>0
    [~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_z);
end

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Step 1: a2primeIndex, a2primeProbs at this age -- helper adds the u dim
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_e==0
        Policy_slice=Policy_k(:,:,:,jj); % [l_d+l_a1, N_a, N_shocks]
    else
        Policy_slice=reshape(Policy_k(:,:,:,:,jj), [l_d+l_a1, N_a, N_shocks*N_e]);
    end
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetu(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, N_shocks*max(N_e,1), n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    % Shapes (helper passes N_z arg=N_shocks*max(N_e,1)>0 so the N_z>0 branch is always taken):
    %   N_e==0: [N_a, N_shocks,    N_u]
    %   N_e>0:  [N_a, N_shocks*N_e, N_u]

    % Step 2: ReturnFn at policy (u does not enter Return)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
    end

    if jj==N_j
        if N_e==0
            V(:,:,jj)=F_jj;
        else
            V(:,:,:,jj)=F_jj;
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

        % Step 3a: integrate next-period V over e' (if any)
        if N_e==0
            V_next=V(:,:,jj+1);
        else
            V_next=V(:,:,:,jj+1);
            V_next=sum(V_next .* shiftdim(vfoptions.pi_e_J(:,jj), -2), 3);
            V_next=reshape(V_next, [N_a, N_shocks]);
        end

        % Step 3b: integrate over z' (markov, does not depend on d_semiz)
        if N_z==0
            EV_after_z=V_next;
        else
            V_next_r=reshape(V_next, [N_a, N_semiz, N_z]);
            EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3); % [N_a, N_semiz_to, 1, N_z_from]
            EV_after_z(isnan(EV_after_z))=0;
            EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_z]);
        end

        % Step 3c: for each d_semiz, integrate over semiz' -> EVnext_byd2
        if N_z==0
            EVnext_byd2=zeros(N_a, N_semiz, N_dsemiz, 'gpuArray');
            for d2_c=1:N_dsemiz
                pi_d2c=pi_semiz_J(:,:,d2_c,jj)';
                EVd2c=sum(EV_after_z .* shiftdim(pi_d2c, -1), 2);
                EVd2c(isnan(EVd2c))=0;
                EVnext_byd2(:,:,d2_c)=reshape(EVd2c, [N_a, N_semiz]);
            end
        else
            EVnext_byd2=zeros(N_a, N_semiz, N_z, N_dsemiz, 'gpuArray');
            for d2_c=1:N_dsemiz
                pi_d2c=pi_semiz_J(:,:,d2_c,jj)';
                pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]);
                EVd2c=sum(EV_after_z .* pi_reshape, 2);
                EVd2c(isnan(EVd2c))=0;
                EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_z]);
            end
        end

        % Step 4: per-state lookup -- combine (i) d_semiz indirection, (ii) a2 interpolation, (iii) sum over u with pi_u
        if N_e==0
            a1p=a1prime_idx(:,:,jj);   % [N_a, N_shocks]
            d2_jj=d_semiz_idx(:,:,jj); % [N_a, N_shocks]
            if N_z==0
                a1p_r=reshape(a1p,[N_a, N_semiz]);
                d2_r =reshape(d2_jj,[N_a, N_semiz]);
                a2pIdx=a2primeIndex; a2pPrb=a2primeProbs;        % [N_a, N_semiz, N_u]
                aprime_low=a1p_r+N_a1*(a2pIdx-1);                 % [N_a, N_semiz, N_u]
                aprime_up =a1p_r+N_a1*(a2pIdx);
                base_off=reshape(N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_r(:)-1), [N_a, N_semiz]);
                lo_idx=aprime_low+base_off; % broadcast over u
                up_idx=aprime_up +base_off;
                EV_lo=reshape(EVnext_byd2(lo_idx(:)),[N_a, N_semiz, N_u]);
                EV_up=reshape(EVnext_byd2(up_idx(:)),[N_a, N_semiz, N_u]);
                per_u=a2pPrb.*EV_lo+(1-a2pPrb).*EV_up;
                EVnext_atpolicy=sum(per_u .* shiftdim(pi_u,-2), 3); % [N_a, N_semiz]
                V(:,:,jj)=F_jj+beta*EVnext_atpolicy;
            else
                a1p_r=reshape(a1p,[N_a, N_semiz, N_z]);
                d2_r =reshape(d2_jj,[N_a, N_semiz, N_z]);
                a2pIdx=reshape(a2primeIndex,[N_a, N_semiz, N_z, N_u]);
                a2pPrb=reshape(a2primeProbs,[N_a, N_semiz, N_z, N_u]);
                aprime_low=a1p_r+N_a1*(a2pIdx-1);
                aprime_up =a1p_r+N_a1*(a2pIdx);
                base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_r(:)-1), [N_a, N_semiz, N_z]);
                lo_idx=aprime_low+base_off;
                up_idx=aprime_up +base_off;
                EV_lo=reshape(EVnext_byd2(lo_idx(:)),[N_a, N_semiz, N_z, N_u]);
                EV_up=reshape(EVnext_byd2(up_idx(:)),[N_a, N_semiz, N_z, N_u]);
                per_u=a2pPrb.*EV_lo+(1-a2pPrb).*EV_up;
                EVnext_atpolicy=sum(per_u .* shiftdim(pi_u,-3), 4); % [N_a, N_semiz, N_z]
                V(:,:,jj)=F_jj+beta*reshape(EVnext_atpolicy, [N_a, N_shocks]);
            end
        else
            if N_z==0
                EVnext_atpolicy=zeros(N_a, N_semiz, N_e, 'gpuArray');
                for e_c=1:N_e
                    block=(e_c-1)*N_shocks + (1:N_shocks);
                    a1p_e=reshape(a1prime_idx(:,:,e_c,jj),[N_a, N_semiz]);
                    d2_e =reshape(d_semiz_idx(:,:,e_c,jj),[N_a, N_semiz]);
                    a2pIdx_e=reshape(a2primeIndex(:,block,:),[N_a, N_semiz, N_u]);
                    a2pPrb_e=reshape(a2primeProbs(:,block,:),[N_a, N_semiz, N_u]);
                    aprime_low_e=a1p_e+N_a1*(a2pIdx_e-1);
                    aprime_up_e =a1p_e+N_a1*(a2pIdx_e);
                    base_off=reshape(N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_e(:)-1), [N_a, N_semiz]);
                    lo_idx=aprime_low_e+base_off;
                    up_idx=aprime_up_e +base_off;
                    EV_lo=reshape(EVnext_byd2(lo_idx(:)),[N_a, N_semiz, N_u]);
                    EV_up=reshape(EVnext_byd2(up_idx(:)),[N_a, N_semiz, N_u]);
                    per_u=a2pPrb_e.*EV_lo+(1-a2pPrb_e).*EV_up;
                    EVnext_atpolicy(:,:,e_c)=sum(per_u .* shiftdim(pi_u,-2), 3);
                end
                V(:,:,:,jj)=F_jj+beta*EVnext_atpolicy;
            else
                EVnext_atpolicy=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
                for e_c=1:N_e
                    block=(e_c-1)*N_shocks + (1:N_shocks);
                    a1p_e=reshape(a1prime_idx(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    d2_e =reshape(d_semiz_idx(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    a2pIdx_e=reshape(a2primeIndex(:,block,:),[N_a, N_semiz, N_z, N_u]);
                    a2pPrb_e=reshape(a2primeProbs(:,block,:),[N_a, N_semiz, N_z, N_u]);
                    aprime_low_e=a1p_e+N_a1*(a2pIdx_e-1);
                    aprime_up_e =a1p_e+N_a1*(a2pIdx_e);
                    base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1), [N_a, N_semiz, N_z]);
                    lo_idx=aprime_low_e+base_off;
                    up_idx=aprime_up_e +base_off;
                    EV_lo=reshape(EVnext_byd2(lo_idx(:)),[N_a, N_semiz, N_z, N_u]);
                    EV_up=reshape(EVnext_byd2(up_idx(:)),[N_a, N_semiz, N_z, N_u]);
                    per_u=a2pPrb_e.*EV_lo+(1-a2pPrb_e).*EV_up;
                    EVnext_atpolicy(:,:,:,e_c)=sum(per_u .* shiftdim(pi_u,-3), 4);
                end
                V(:,:,:,jj)=F_jj+beta*reshape(EVnext_atpolicy, [N_a, N_shocks, N_e]);
            end
        end
    end
end

%% Reshape V out of Kron form
if N_z==0 && N_e==0
    V=reshape(V, [n_a, n_semiz, N_j]);
elseif N_z==0 && N_e>0
    V=reshape(V, [n_a, n_semiz, vfoptions.n_e, N_j]);
elseif N_z>0 && N_e==0
    V=reshape(V, [n_a, n_semiz, n_z, N_j]);
else
    V=reshape(V, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
end


end
