function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V and Valt from a Policy under QuasiHyperbolic discounting with semi-exogenous shocks.
%
% Returns [V, Valt] matching ValueFnIter_FHorz_QuasiHyperbolic:
%   Sophisticated: V = Vhat (QH-discounted from current self's perspective)
%                  Valt = Vunderbar (realised continuation under future selves' own QH choices)
%   Naive:         V = Vtilde (QH-discounted, agent's perceived value at Policy)
%                  Valt = exponential-discounter value computed at Policyalt
%                  Requires vfoptions.Policyalt.
%
% semiz transition pi_semiz depends on the policy's d_semiz choice (last l_dsemiz components of d).
% For Naive, Vtilde looks up EVnext at Policy's d_semiz; Valt looks up EVnext at Policyalt's d_semiz.

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');
if ~isNaive && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    error('vfoptions.quasi_hyperbolic must be ''Naive'' or ''Sophisticated''')
end

%% Naive requires Policyalt
if isNaive
    if ~isfield(vfoptions,'Policyalt')
        error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_SemiExo (Naive): vfoptions.Policyalt is required. It is returned as the 4th output of ValueFnIter_FHorz_QuasiHyperbolic for Naive.')
    end
    Policyalt=gpuArray(vfoptions.Policyalt);
end

%% Setup (mirrors ValueFnFromPolicy_FHorz_SemiExo)
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

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
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_SemiExo: SemiExo requires at least one decision variable')
end
l_d=length(n_d);
l_a=length(n_a);
l_aprime=l_a;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

if N_z==0
    n_shocks=n_semiz;
else
    n_shocks=[n_semiz,n_z];
end
N_shocks=N_semiz*max(N_z,1);

%% PolicyValues for ReturnFn evaluation (Policy, and Policyalt if Naive)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
l_daprime=size(PolicyValues,1);

if isNaive
    PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
end

%% Joint shock gridvals for ReturnFn (shared between GI and non-GI paths)
if N_z==0
    joint_gridvals_J=semiz_gridvals_J;
else
    joint_gridvals_J=zeros(N_shocks, length(n_semiz)+length(n_z), N_j, 'gpuArray');
    for jj=1:N_j
        joint_gridvals_J(:,:,jj)=[repmat(semiz_gridvals_J(:,:,jj),N_z,1), repelem(z_gridvals_J(:,:,jj),N_semiz,1)];
    end
end

%% GI1 dispatch (l_a==1 only; GI2A deferred)
if vfoptions.gridinterplayer==1
    if l_a>=2
        error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_SemiExo: GI2A (l_a>=2) not yet implemented for semiz QH dual {V,Valt}')
    end
    n2short=vfoptions.ngridinterp;

    % Extract GI indices (a1_lower, a1_upper, a1_weight) AND d_semiz_idx from raw Policy (Naive: also from Policyalt)
    [a1_lower, a1_upper, a1_weight, d_semiz_idx]=extract_gi_indices(Policy, l_d, l_dsemiz, l_aprime, n_a, n_dsemiz, n2short, N_a, N_shocks, N_e, N_j);
    if isNaive
        [a1_lower_alt, a1_upper_alt, a1_weight_alt, d_semiz_idx_alt]=extract_gi_indices(Policyalt, l_d, l_dsemiz, l_aprime, n_a, n_dsemiz, n2short, N_a, N_shocks, N_e, N_j);
    end

    % Initialise value arrays
    if N_e==0
        sz=[N_a, N_shocks, N_j];
    else
        sz=[N_a, N_shocks, N_e, N_j];
    end
    if isNaive
        Valt=zeros(sz,'gpuArray');
        Vtilde=zeros(sz,'gpuArray');
    else
        Vunderbar=zeros(sz,'gpuArray');
        Vhat=zeros(sz,'gpuArray');
    end

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        if N_e==0
            F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
            if isNaive
                F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
            end
        else
            F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
            if isNaive
                F_alt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
            end
        end

        if jj==N_j
            if isNaive
                if N_e==0
                    Valt(:,:,jj)=F_alt_jj;
                    Vtilde(:,:,jj)=F_jj;
                else
                    Valt(:,:,:,jj)=F_alt_jj;
                    Vtilde(:,:,:,jj)=F_jj;
                end
            else
                if N_e==0
                    Vunderbar(:,:,jj)=F_jj;
                    Vhat(:,:,jj)=F_jj;
                else
                    Vunderbar(:,:,:,jj)=F_jj;
                    Vhat(:,:,:,jj)=F_jj;
                end
            end
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
            beta0beta=beta0*beta;

            if isNaive
                if N_e==0
                    V_next=Valt(:,:,jj+1);
                else
                    V_next=Valt(:,:,:,jj+1);
                end
            else
                if N_e==0
                    V_next=Vunderbar(:,:,jj+1);
                else
                    V_next=Vunderbar(:,:,:,jj+1);
                end
            end

            if N_e>0
                V_next=sum(V_next .* shiftdim(vfoptions.pi_e_J(:,jj), -2), 3);
                V_next=reshape(V_next, [N_a, N_shocks]);
            end

            EVnext_byd2=precompute_EVnext_byd2(V_next, pi_semiz_J(:,:,:,jj), pi_z_J, jj, N_a, N_semiz, N_z, N_dsemiz);

            if isNaive
                EV_at_P =lookup_EVnext_GI(EVnext_byd2, a1_lower,     a1_upper,     a1_weight,     d_semiz_idx,     jj, N_a, N_semiz, N_z, N_e);
                EV_at_Pa=lookup_EVnext_GI(EVnext_byd2, a1_lower_alt, a1_upper_alt, a1_weight_alt, d_semiz_idx_alt, jj, N_a, N_semiz, N_z, N_e);
                if N_e==0
                    Valt(:,:,jj)  =F_alt_jj + beta    *reshape(EV_at_Pa,[N_a,N_shocks]);
                    Vtilde(:,:,jj)=F_jj     + beta0beta*reshape(EV_at_P, [N_a,N_shocks]);
                else
                    Valt(:,:,:,jj)  =F_alt_jj + beta    *reshape(EV_at_Pa,[N_a,N_shocks,N_e]);
                    Vtilde(:,:,:,jj)=F_jj     + beta0beta*reshape(EV_at_P, [N_a,N_shocks,N_e]);
                end
            else
                EV_at_P=lookup_EVnext_GI(EVnext_byd2, a1_lower, a1_upper, a1_weight, d_semiz_idx, jj, N_a, N_semiz, N_z, N_e);
                if N_e==0
                    Vunderbar(:,:,jj)=F_jj + beta    *reshape(EV_at_P,[N_a,N_shocks]);
                    Vhat(:,:,jj)     =F_jj + beta0beta*reshape(EV_at_P,[N_a,N_shocks]);
                else
                    Vunderbar(:,:,:,jj)=F_jj + beta    *reshape(EV_at_P,[N_a,N_shocks,N_e]);
                    Vhat(:,:,:,jj)     =F_jj + beta0beta*reshape(EV_at_P,[N_a,N_shocks,N_e]);
                end
            end
        end
    end

    if isNaive
        Vtemp=Vtilde;
        Valt_out=Valt;
    else
        Vtemp=Vhat;
        Valt_out=Vunderbar;
    end
    if N_z==0 && N_e==0
        V=reshape(Vtemp, [n_a, n_semiz, N_j]);
        Valt=reshape(Valt_out, [n_a, n_semiz, N_j]);
    elseif N_z==0 && N_e>0
        V=reshape(Vtemp, [n_a, n_semiz, vfoptions.n_e, N_j]);
        Valt=reshape(Valt_out, [n_a, n_semiz, vfoptions.n_e, N_j]);
    elseif N_z>0 && N_e==0
        V=reshape(Vtemp, [n_a, n_semiz, n_z, N_j]);
        Valt=reshape(Valt_out, [n_a, n_semiz, n_z, N_j]);
    else
        V=reshape(Vtemp, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
        Valt=reshape(Valt_out, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
    end
    return
end

%% Non-GI path
%% Extract per-state (aprime_idx, d_semiz_idx) for Policy (and Policyalt if Naive)
[aprime_idx, d_semiz_idx]=extract_indices(Policy, l_d, l_dsemiz, l_aprime, n_a, n_dsemiz, N_a, N_shocks, N_e, N_j);
if isNaive
    [aprime_idx_alt, d_semiz_idx_alt]=extract_indices(Policyalt, l_d, l_dsemiz, l_aprime, n_a, n_dsemiz, N_a, N_shocks, N_e, N_j);
end

%% Backward iteration: dual-V recursion
% Sophisticated: Vunderbar (beta, drives recursion) and Vhat (beta0beta, perceived).
% Naive:         Valt (beta at Policyalt, drives recursion) and Vtilde (beta0beta at Policy).
if N_e==0
    sz=[N_a, N_shocks, N_j];
else
    sz=[N_a, N_shocks, N_e, N_j];
end
if isNaive
    Valt=zeros(sz,'gpuArray');
    Vtilde=zeros(sz,'gpuArray');
else
    Vunderbar=zeros(sz,'gpuArray');
    Vhat=zeros(sz,'gpuArray');
end

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
        if isNaive
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
        end
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
        if isNaive
            F_alt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
        end
    end

    if jj==N_j
        if isNaive
            if N_e==0
                Valt(:,:,jj)=F_alt_jj;
                Vtilde(:,:,jj)=F_jj;
            else
                Valt(:,:,:,jj)=F_alt_jj;
                Vtilde(:,:,:,jj)=F_jj;
            end
        else
            if N_e==0
                Vunderbar(:,:,jj)=F_jj;
                Vhat(:,:,jj)=F_jj;
            else
                Vunderbar(:,:,:,jj)=F_jj;
                Vhat(:,:,:,jj)=F_jj;
            end
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
        beta0beta=beta0*beta;

        % V_next is the continuation-driving fn: Vunderbar(jj+1) for Soph, Valt(jj+1) for Naive
        if isNaive
            if N_e==0
                V_next=Valt(:,:,jj+1);
            else
                V_next=Valt(:,:,:,jj+1);
            end
        else
            if N_e==0
                V_next=Vunderbar(:,:,jj+1);
            else
                V_next=Vunderbar(:,:,:,jj+1);
            end
        end

        % Integrate out e' (if present)
        if N_e>0
            V_next=sum(V_next .* shiftdim(vfoptions.pi_e_J(:,jj), -2), 3);
            V_next=reshape(V_next, [N_a, N_shocks]);
        end

        % Precompute EVnext_byd2: E[V_next | d_semiz] over (semiz', z' if present), for each d_semiz
        EVnext_byd2=precompute_EVnext_byd2(V_next, pi_semiz_J(:,:,:,jj), pi_z_J, jj, N_a, N_semiz, N_z, N_dsemiz);

        if isNaive
            % Naive: Vtilde uses Policy's (aprime, d_semiz); Valt uses Policyalt's
            EV_at_P =lookup_EVnext_full(EVnext_byd2, aprime_idx,     d_semiz_idx,     jj, N_a, N_semiz, N_z, N_e);
            EV_at_Pa=lookup_EVnext_full(EVnext_byd2, aprime_idx_alt, d_semiz_idx_alt, jj, N_a, N_semiz, N_z, N_e);
            if N_e==0
                Valt(:,:,jj)  =F_alt_jj + beta    *reshape(EV_at_Pa,[N_a,N_shocks]);
                Vtilde(:,:,jj)=F_jj     + beta0beta*reshape(EV_at_P, [N_a,N_shocks]);
            else
                Valt(:,:,:,jj)  =F_alt_jj + beta    *reshape(EV_at_Pa,[N_a,N_shocks,N_e]);
                Vtilde(:,:,:,jj)=F_jj     + beta0beta*reshape(EV_at_P, [N_a,N_shocks,N_e]);
            end
        else
            % Sophisticated: both Vhat and Vunderbar lookups use the same (Policy) state
            EV_at_P=lookup_EVnext_full(EVnext_byd2, aprime_idx, d_semiz_idx, jj, N_a, N_semiz, N_z, N_e);
            if N_e==0
                Vunderbar(:,:,jj)=F_jj + beta    *reshape(EV_at_P,[N_a,N_shocks]);
                Vhat(:,:,jj)     =F_jj + beta0beta*reshape(EV_at_P,[N_a,N_shocks]);
            else
                Vunderbar(:,:,:,jj)=F_jj + beta    *reshape(EV_at_P,[N_a,N_shocks,N_e]);
                Vhat(:,:,:,jj)     =F_jj + beta0beta*reshape(EV_at_P,[N_a,N_shocks,N_e]);
            end
        end
    end
end

%% Reshape outputs
if isNaive
    Vtemp=Vtilde;
    Valt_out=Valt;
else
    Vtemp=Vhat;
    Valt_out=Vunderbar;
end

if N_z==0 && N_e==0
    V=reshape(Vtemp, [n_a, n_semiz, N_j]);
    Valt=reshape(Valt_out, [n_a, n_semiz, N_j]);
elseif N_z==0 && N_e>0
    V=reshape(Vtemp, [n_a, n_semiz, vfoptions.n_e, N_j]);
    Valt=reshape(Valt_out, [n_a, n_semiz, vfoptions.n_e, N_j]);
elseif N_z>0 && N_e==0
    V=reshape(Vtemp, [n_a, n_semiz, n_z, N_j]);
    Valt=reshape(Valt_out, [n_a, n_semiz, n_z, N_j]);
else
    V=reshape(Vtemp, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
    Valt=reshape(Valt_out, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
end

end


function [aprime_idx, d_semiz_idx]=extract_indices(Policy, l_d, l_dsemiz, l_aprime, n_a, n_dsemiz, N_a, N_shocks, N_e, N_j)
% Decompose Policy into per-state aprime_idx and d_semiz_idx (joint index into n_dsemiz). Non-GI Policy layout.
if N_e==0
    Policy_k=reshape(Policy,[l_d+l_aprime, N_a, N_shocks, N_j]);
    d_semiz_idx=ones(N_a,N_shocks,N_j,'gpuArray');
    aprime_idx=ones(N_a,N_shocks,N_j,'gpuArray');
else
    Policy_k=reshape(Policy,[l_d+l_aprime, N_a, N_shocks, N_e, N_j]);
    d_semiz_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
    aprime_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
end
cumprods_dsemiz=[1, cumprod(n_dsemiz(1:end-1))];
for ii=1:l_dsemiz
    comp=shiftdim(Policy_k(l_d-l_dsemiz+ii, :, :, :, :),1);
    d_semiz_idx=d_semiz_idx+cumprods_dsemiz(ii)*(comp-1);
end
cumprods_a=[1, cumprod(n_a(1:end-1))];
for ii=1:l_aprime
    comp=shiftdim(Policy_k(l_d+ii, :, :, :, :),1);
    aprime_idx=aprime_idx+cumprods_a(ii)*(comp-1);
end
end


function [a1_lower, a1_upper, a1_weight, d_semiz_idx]=extract_gi_indices(Policy, l_d, l_dsemiz, l_aprime, n_a, n_dsemiz, n2short, N_a, N_shocks, N_e, N_j)
% Decompose GI1 Policy: a1 (midpoint) + L2 (fine-grid index) → lower/upper indices + interpolation weight.
% Also returns d_semiz_idx (joint index into n_dsemiz). Assumes l_a==1.
% Strips trailing L2flag channel if Policy has more than (l_d+l_aprime+1) channels.
if size(Policy,1) > (l_d+l_aprime+1)
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1), prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:l_d+l_aprime+1,:), [(l_d+l_aprime+1), tempsize(2:end)]);
end

if N_e==0
    Policy_k=reshape(Policy,[l_d+l_aprime+1, N_a, N_shocks, N_j]);
    d_semiz_idx=ones(N_a,N_shocks,N_j,'gpuArray');
else
    Policy_k=reshape(Policy,[l_d+l_aprime+1, N_a, N_shocks, N_e, N_j]);
    d_semiz_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
end
cumprods_dsemiz=[1, cumprod(n_dsemiz(1:end-1))];
for ii=1:l_dsemiz
    comp=shiftdim(Policy_k(l_d-l_dsemiz+ii, :, :, :, :),1);
    d_semiz_idx=d_semiz_idx+cumprods_dsemiz(ii)*(comp-1);
end

a1_mid=shiftdim(Policy_k(l_d+1, :, :, :, :), 1);
L2_idx=shiftdim(Policy_k(l_d+l_aprime+1, :, :, :, :), 1);

% Fine-grid index then fractional position in original a1 grid
a1_fine_idx=(n2short+1)*(a1_mid-1)+L2_idx;
a1_frac=1+(a1_fine_idx-1)/(n2short+1);
a1_lower=floor(a1_frac);
a1_weight=a1_frac-a1_lower; % weight on upper
a1_upper=min(a1_lower+1, n_a(1));
a1_upper(a1_lower>=n_a(1))=n_a(1);
a1_lower(a1_lower<1)=1;
end


function EVnext_byd2=precompute_EVnext_byd2(V_next, pi_semiz_jj, pi_z_J, jj, N_a, N_semiz, N_z, N_dsemiz)
% Build EVnext_byd2[a, semiz_from, (z_from), d_semiz] = E[V_next | d_semiz] integrated over (semiz', z' if present).
% V_next shape: [N_a, N_shocks] where N_shocks = N_semiz (no z) or N_semiz*N_z.
% pi_semiz_jj shape: [N_semiz_from, N_semiz_to, N_dsemiz] (the jj slice of pi_semiz_J).
if N_z==0
    EV_after_z=V_next; % [N_a, N_semiz_to]
    EVnext_byd2=zeros(N_a, N_semiz, N_dsemiz, 'gpuArray');
    for d2_c=1:N_dsemiz
        pi_d2c=pi_semiz_jj(:,:,d2_c)'; % [N_semiz_to, N_semiz_from]
        EVd2c=sum(EV_after_z .* shiftdim(pi_d2c, -1), 2); % [N_a, 1, N_semiz_from]
        EVd2c(isnan(EVd2c))=0;
        EVnext_byd2(:,:,d2_c)=reshape(EVd2c, [N_a, N_semiz]);
    end
else
    V_next_r=reshape(V_next, [N_a, N_semiz, N_z]);
    EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3);
    EV_after_z(isnan(EV_after_z))=0;
    EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_z]);
    EVnext_byd2=zeros(N_a, N_semiz, N_z, N_dsemiz, 'gpuArray');
    for d2_c=1:N_dsemiz
        pi_d2c=pi_semiz_jj(:,:,d2_c)'; % [N_semiz_to, N_semiz_from]
        pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]);
        EVd2c=sum(EV_after_z .* pi_reshape, 2);
        EVd2c(isnan(EVd2c))=0;
        EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_z]);
    end
end
end


function EVnext_atpolicy=lookup_EVnext_full(EVnext_byd2, aprime_idx, d_semiz_idx, jj, N_a, N_semiz, N_z, N_e)
% Non-GI lookup: EVnext at each state's (aprime, semiz_from, z_from if present, d_semiz).
% Returns array shaped (N_a, N_semiz [, N_z] [, N_e]).
[~, SZ_grid_noz]=ndgrid(1:N_a, 1:N_semiz);
if N_z>0
    [~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_z);
end
if N_e==0
    aprime_jj=aprime_idx(:,:,jj);
    d2_jj=d_semiz_idx(:,:,jj);
    if N_z==0
        aprime_jj_r=reshape(aprime_jj, [N_a, N_semiz]);
        d2_jj_r=reshape(d2_jj, [N_a, N_semiz]);
        linear_idx=aprime_jj_r(:)+N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_jj_r(:)-1);
        EVnext_atpolicy=reshape(EVnext_byd2(linear_idx), [N_a, N_semiz]);
    else
        aprime_jj_r=reshape(aprime_jj, [N_a, N_semiz, N_z]);
        d2_jj_r=reshape(d2_jj, [N_a, N_semiz, N_z]);
        linear_idx=aprime_jj_r(:)+N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_jj_r(:)-1);
        EVnext_atpolicy=reshape(EVnext_byd2(linear_idx), [N_a, N_semiz, N_z]);
    end
else
    if N_z==0
        EVnext_atpolicy=zeros(N_a, N_semiz, N_e, 'gpuArray');
        for e_c=1:N_e
            aprime_e=reshape(aprime_idx(:,:,e_c,jj), [N_a, N_semiz]);
            d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz]);
            linear_idx=aprime_e(:)+N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_e(:)-1);
            EVnext_atpolicy(:,:,e_c)=reshape(EVnext_byd2(linear_idx), [N_a, N_semiz]);
        end
    else
        EVnext_atpolicy=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
        for e_c=1:N_e
            aprime_e=reshape(aprime_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
            d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
            linear_idx=aprime_e(:)+N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1);
            EVnext_atpolicy(:,:,:,e_c)=reshape(EVnext_byd2(linear_idx), [N_a, N_semiz, N_z]);
        end
    end
end
end


function EVnext_atpolicy=lookup_EVnext_GI(EVnext_byd2, a1_lower, a1_upper, a1_weight, d_semiz_idx, jj, N_a, N_semiz, N_z, N_e)
% GI lookup: weighted EVnext between (a1_lower, ...) and (a1_upper, ...) at each state's (semiz_from, z_from if present, d_semiz).
% Returns array shaped (N_a, N_semiz [, N_z] [, N_e]).
[~, SZ_grid_noz]=ndgrid(1:N_a, 1:N_semiz);
if N_z>0
    [~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_z);
end
if N_e==0
    aprime_lo_jj=a1_lower(:,:,jj);
    aprime_up_jj=a1_upper(:,:,jj);
    w_jj=a1_weight(:,:,jj);
    d2_jj=d_semiz_idx(:,:,jj);
    if N_z==0
        aprime_lo_r=reshape(aprime_lo_jj, [N_a, N_semiz]);
        aprime_up_r=reshape(aprime_up_jj, [N_a, N_semiz]);
        w_r=reshape(w_jj, [N_a, N_semiz]);
        d2_r=reshape(d2_jj, [N_a, N_semiz]);
        base_off=N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_r(:)-1);
        lo_idx=aprime_lo_r(:)+base_off;
        up_idx=aprime_up_r(:)+base_off;
        EVnext_atpolicy=reshape((1-w_r(:)).*EVnext_byd2(lo_idx)+w_r(:).*EVnext_byd2(up_idx), [N_a, N_semiz]);
    else
        aprime_lo_r=reshape(aprime_lo_jj, [N_a, N_semiz, N_z]);
        aprime_up_r=reshape(aprime_up_jj, [N_a, N_semiz, N_z]);
        w_r=reshape(w_jj, [N_a, N_semiz, N_z]);
        d2_r=reshape(d2_jj, [N_a, N_semiz, N_z]);
        base_off=N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_r(:)-1);
        lo_idx=aprime_lo_r(:)+base_off;
        up_idx=aprime_up_r(:)+base_off;
        EVnext_atpolicy=reshape((1-w_r(:)).*EVnext_byd2(lo_idx)+w_r(:).*EVnext_byd2(up_idx), [N_a, N_semiz, N_z]);
    end
else
    if N_z==0
        EVnext_atpolicy=zeros(N_a, N_semiz, N_e, 'gpuArray');
        for e_c=1:N_e
            aprime_lo_e=reshape(a1_lower(:,:,e_c,jj), [N_a, N_semiz]);
            aprime_up_e=reshape(a1_upper(:,:,e_c,jj), [N_a, N_semiz]);
            w_e=reshape(a1_weight(:,:,e_c,jj), [N_a, N_semiz]);
            d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz]);
            base_off=N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_e(:)-1);
            lo_idx=aprime_lo_e(:)+base_off;
            up_idx=aprime_up_e(:)+base_off;
            EVnext_atpolicy(:,:,e_c)=reshape((1-w_e(:)).*EVnext_byd2(lo_idx)+w_e(:).*EVnext_byd2(up_idx), [N_a, N_semiz]);
        end
    else
        EVnext_atpolicy=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
        for e_c=1:N_e
            aprime_lo_e=reshape(a1_lower(:,:,e_c,jj), [N_a, N_semiz, N_z]);
            aprime_up_e=reshape(a1_upper(:,:,e_c,jj), [N_a, N_semiz, N_z]);
            w_e=reshape(a1_weight(:,:,e_c,jj), [N_a, N_semiz, N_z]);
            d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
            base_off=N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1);
            lo_idx=aprime_lo_e(:)+base_off;
            up_idx=aprime_up_e(:)+base_off;
            EVnext_atpolicy(:,:,:,e_c)=reshape((1-w_e(:)).*EVnext_byd2(lo_idx)+w_e(:).*EVnext_byd2(up_idx), [N_a, N_semiz, N_z]);
        end
    end
end
end
