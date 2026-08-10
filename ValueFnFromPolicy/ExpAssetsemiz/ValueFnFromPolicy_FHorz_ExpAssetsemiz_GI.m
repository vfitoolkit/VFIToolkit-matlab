function varargout=ValueFnFromPolicy_FHorz_ExpAssetsemiz_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model has experienceassetsemiz (vfoptions.experienceassetsemiz>=1)
% AND uses the grid interpolation layer (vfoptions.gridinterplayer==1).
% The experience asset a2 is driven by the semi-exogenous state: a2prime=aprimeFn(d_expasset,a2,semiz,...).
% semiz is always present; z is an OPTIONAL ordinary Markov shock.
%
% Per-state EVnext lookup combines three pieces:
%   - a1 fine-grid interpolation (lower/upper a1 grid point + L2 weight)
%   - a2 interpolation via a2primeIndex/a2primeProbs (from CreateaprimePolicyExperienceAssetsemiz)
%   - d_semiz indirection: pick the d2-slice of EVnext_byd2 via d_semiz_idx

%% Setup
% Semiz gridvals + pi_semiz_J
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetsemiz you must define vfoptions.aprimeFn')
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
if N_d==0
    error('ValueFnFromPolicy_FHorz_ExpAssetsemiz_GI: experienceassetsemiz requires at least one decision variable')
end
l_d=length(n_d);
l_semiz=length(n_semiz);

% z is optional. Use N_zloc (=1 when no z) for the reshapes; build a trivial pi_z when there is no z.
N_zloc=max(N_z,1);
if N_z==0
    z_gridvals_J=[];
    pi_z_J=ones(1,1,N_j,'gpuArray'); % single 'z' that transitions to itself (integration over z' is a no-op)
else
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);
end

if isscalar(n_a)
    error('ValueFnFromPolicy_FHorz_ExpAssetsemiz_GI: case with no a1 (experience asset as only asset) not yet implemented')
end
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
n_a2=n_a(end);
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);

if isfield(vfoptions,'l_dexperienceassetsemiz')
    l_d2=vfoptions.l_dexperienceassetsemiz;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_dsemiz-l_d2+1):(l_d-l_dsemiz);

temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_semiz)
    aprimeFnParamNames={temp{l_d2+l_a2+l_semiz+1:end}};
else
    aprimeFnParamNames={};
end

% Joint shock = [semiz, z] (semiz fastest); when no z it is just semiz
if N_z==0
    n_shocks=n_semiz;
else
    n_shocks=[n_semiz,n_z];
end
N_shocks=N_semiz*N_zloc;

n2short=vfoptions.ngridinterp;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

%% PolicyValues (PolicyInd2Val_FHorz handles experienceassetsemiz+GI internally)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);

%% Strip trailing L2flag channel from Policy if present
size_first=l_d+l_a1+1; % under GI: d, a1mid, L2 -> l_d + l_a1 + 1 channels
if size(Policy,1) > size_first
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1), prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:size_first,:), [size_first, tempsize(2:end)]);
end

%% Reshape Policy to canonical Kron form
if N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_shocks, N_j]);
else
    Policy_k=reshape(Policy,[size_first, N_a, N_shocks, N_e, N_j]);
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

%% a1prime: midpoint (position l_d+1) + L2 (last). Other a1 components at l_d+2..l_d+l_a1.
a1_mid=shiftdim(Policy_k(l_d+1,:,:,:,:),1);
L2    =shiftdim(Policy_k(l_d+l_a1+1,:,:,:,:),1);
w_a1_upper=(L2-1)/(n2short+1);
w_a1_lower=1-w_a1_upper;
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
a1_lower=a1_mid;
for ii=2:l_a1
    comp=shiftdim(Policy_k(l_d+ii,:,:,:,:),1);
    a1_lower=a1_lower+cumprods_a1(ii)*(comp-1);
end
a1_upper=a1_lower+1;
a1_top_clamp=(a1_mid>=n_a1(1));
a1_upper(a1_top_clamp)=a1_lower(a1_top_clamp);

%% Joint shock gridvals for ReturnFn
if N_z==0
    joint_gridvals_J=semiz_gridvals_J; % bothz = semiz
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

[~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_zloc);

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Step 1: a2primeIndex, a2primeProbs at this age (aprime driven by semiz)
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_e==0
        Policy_slice=Policy_k(:,:,:,jj); % [size_first, N_a, N_shocks]
    else
        Policy_slice=reshape(Policy_k(:,:,:,:,jj), [size_first, N_a, N_shocks*N_e]);
    end
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetsemiz(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_semiz, N_semiz, N_zloc, N_e, d_grid, a2_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec);
    % Shapes:
    %   N_e==0: [N_a, N_shocks]
    %   N_e>0:  [N_a, N_shocks*N_e]

    % Step 2: ReturnFn at policy
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

        % Step 3a: integrate over e' (if any)
        if N_e==0
            V_next=V(:,:,jj+1);
        else
            V_next=V(:,:,:,jj+1);
            V_next=sum(V_next .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3);
            V_next=reshape(V_next, [N_a, N_shocks]);
        end

        % Step 3b: integrate over z' (markov). Trivial when no z.
        V_next_r=reshape(V_next, [N_a, N_semiz, N_zloc]);
        EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3);
        EV_after_z(isnan(EV_after_z))=0;
        EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_zloc]);

        % Step 3c: for each d_semiz, integrate over semiz' -> EVnext_byd2(a, semiz_from, z_from, d_semiz)
        EVnext_byd2=zeros(N_a, N_semiz, N_zloc, N_dsemiz, 'gpuArray');
        for d2_c=1:N_dsemiz
            pi_d2c=pi_semiz_J(:,:,d2_c,jj)';
            pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]);
            EVd2c=sum(EV_after_z .* pi_reshape, 2);
            EVd2c(isnan(EVd2c))=0;
            EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_zloc]);
        end

        % Step 4: per-state 2x2 corner interpolation on EVnext_byd2 at d_semiz-selected slice.
        % Corners: (a1_low, a2_low), (a1_low, a2_up), (a1_up, a2_low), (a1_up, a2_up)
        if N_e==0
            a1l=a1_lower(:,:,jj); a1u=a1_upper(:,:,jj);
            wa1l=w_a1_lower(:,:,jj); wa1u=w_a1_upper(:,:,jj);
            d2_jj=d_semiz_idx(:,:,jj);
            a1l_r=reshape(a1l,[N_a, N_semiz, N_zloc]); a1u_r=reshape(a1u,[N_a, N_semiz, N_zloc]);
            wa1l_r=reshape(wa1l,[N_a, N_semiz, N_zloc]); wa1u_r=reshape(wa1u,[N_a, N_semiz, N_zloc]);
            a2l=reshape(a2primeIndex,[N_a, N_semiz, N_zloc]); a2u=a2l+1;
            wa2l=reshape(a2primeProbs,[N_a, N_semiz, N_zloc]); wa2u=1-wa2l;
            d2_r=reshape(d2_jj,[N_a, N_semiz, N_zloc]);
            base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_zloc*(d2_r(:)-1), [N_a, N_semiz, N_zloc]);
            lin_LL=a1l_r+N_a1*(a2l-1)+base_off;
            lin_LU=a1l_r+N_a1*(a2u-1)+base_off;
            lin_UL=a1u_r+N_a1*(a2l-1)+base_off;
            lin_UU=a1u_r+N_a1*(a2u-1)+base_off;
            EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_zloc]);
            EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_zloc]);
            EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_zloc]);
            EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_zloc]);
            EVnext_atpolicy=wa1l_r.*wa2l.*EV_LL + wa1l_r.*wa2u.*EV_LU + wa1u_r.*wa2l.*EV_UL + wa1u_r.*wa2u.*EV_UU;
            V(:,:,jj)=F_jj+beta*reshape(EVnext_atpolicy, [N_a, N_shocks]);
        else
            EVnext_atpolicy=zeros(N_a, N_semiz, N_zloc, N_e, 'gpuArray');
            for e_c=1:N_e
                block=(e_c-1)*N_shocks + (1:N_shocks);
                a1l_e=reshape(a1_lower(:,:,e_c,jj),[N_a, N_semiz, N_zloc]);
                a1u_e=reshape(a1_upper(:,:,e_c,jj),[N_a, N_semiz, N_zloc]);
                wa1l_e=reshape(w_a1_lower(:,:,e_c,jj),[N_a, N_semiz, N_zloc]);
                wa1u_e=reshape(w_a1_upper(:,:,e_c,jj),[N_a, N_semiz, N_zloc]);
                a2l_e=reshape(a2primeIndex(:,block),[N_a, N_semiz, N_zloc]); a2u_e=a2l_e+1;
                wa2l_e=reshape(a2primeProbs(:,block),[N_a, N_semiz, N_zloc]); wa2u_e=1-wa2l_e;
                d2_e=reshape(d_semiz_idx(:,:,e_c,jj),[N_a, N_semiz, N_zloc]);
                base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_zloc*(d2_e(:)-1), [N_a, N_semiz, N_zloc]);
                lin_LL=a1l_e+N_a1*(a2l_e-1)+base_off;
                lin_LU=a1l_e+N_a1*(a2u_e-1)+base_off;
                lin_UL=a1u_e+N_a1*(a2l_e-1)+base_off;
                lin_UU=a1u_e+N_a1*(a2u_e-1)+base_off;
                EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_zloc]);
                EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_zloc]);
                EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_zloc]);
                EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_zloc]);
                EVnext_atpolicy(:,:,:,e_c)=wa1l_e.*wa2l_e.*EV_LL + wa1l_e.*wa2u_e.*EV_LU + wa1u_e.*wa2l_e.*EV_UL + wa1u_e.*wa2u_e.*EV_UU;
            end
            V(:,:,:,jj)=F_jj+beta*reshape(EVnext_atpolicy, [N_a, N_shocks, N_e]);
        end
    end
end

%% Reshape V out of Kron form
if N_z==0
    if N_e==0
        V=reshape(V, [n_a, n_semiz, N_j]);
    else
        V=reshape(V, [n_a, n_semiz, vfoptions.n_e, N_j]);
    end
else
    if N_e==0
        V=reshape(V, [n_a, n_semiz, n_z, N_j]);
    else
        V=reshape(V, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
    end
end


varargout={V};

end
