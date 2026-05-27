function V=ValueFnFromPolicy_FHorz_QuasiHyperbolic(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model uses Quasi-Hyperbolic discounting.
%
% Returns Vunderbar (the long-run value of the time-inconsistent policy, used as the
% recursion driver) -- matches the V1 output of ValueFnIter_Case1_FHorz for QH-Sophisticated.
% Vhat (the short-run "perceived" value at the stored policy) is computed internally but NOT returned.
%
% Currently Sophisticated only -- Naive throws "not yet implemented" (the Naive case requires
% a separate optimisation of V at V-optimal policy, which isn't recoverable from the stored
% Vtilde-argmax Policy by recursion alone).

%% Naive vs Sophisticated
if strcmp(vfoptions.quasi_hyperbolic,'Naive')
    warning('ValueFnFromPolicy_FHorz_QuasiHyperbolic: not yet implemented for Naive Quasi-hyperbolic discounting')
    V=0;
    return
elseif ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    error('vfoptions.quasi_hyperbolic must be ''Naive'' or ''Sophisticated''')
end

%% Scope limits -- plain case only for now (no semiz, no ExpAsset family, no GI)
if prod(vfoptions.n_semiz)>0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic: not yet implemented for semiz')
end
if vfoptions.experienceasset==1
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic: not yet implemented for experienceasset')
end
if vfoptions.experienceassetu==1
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic: not yet implemented for experienceassetu')
end
if vfoptions.experienceassetz==1
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic: not yet implemented for experienceassetz')
end
if vfoptions.experienceassete==1
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic: not yet implemented for experienceassete')
end
if vfoptions.experienceassetze==1
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic: not yet implemented for experienceassetze')
end
if vfoptions.gridinterplayer==1
    % Sophisticated under gridinterplayer: Vunderbar_j = u(policy_j) + beta*E[Vunderbar_{j+1}]
    % at the L2-interpolated aprime, which is exactly the recursion ValueFnFromPolicy_FHorz_GI
    % implements (it uses regular beta — appropriate for Vunderbar). Dispatch to it; the V it
    % returns IS Vunderbar (matches the V1 output of ValueFnIter_Case1_FHorz for QH-S).
    V=ValueFnFromPolicy_FHorz_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

%% Setup (mirrors ValueFnFromPolicy_FHorz)
% Caller already moved grids and Policy to GPU and ran ExogShockSetup_FHorz.
% Re-run shock setup here to get z_gridvals_J / pi_z_J locally.
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% Backward iteration: compute Vunderbar (long-run, beta) and Vhat (short-run, beta0*beta) jointly.
% Vunderbar is the recursion driver; Vhat is what gets returned as V.
if N_z==0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]); %[N_a,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1_noz(Policy, n_d, n_a,N_j,vfoptions);

    Vunderbar=zeros(N_a,N_j,'gpuArray');
    Vhat=zeros(N_a,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);

        if jj==N_j
            Vunderbar(:,jj)=FofPolicy_jj;
            Vhat(:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
            beta0beta=beta0*beta;

            EVnext=Vunderbar(:,jj+1);

            if N_d==0
                optaprime=PolicyIndexesKron(1,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,jj),1);
            end

            aprimez_index=reshape(optaprime,[N_a,1]);
            EVnextOfPolicy=reshape(EVnext(aprimez_index),[N_a,1]);

            Vunderbar(:,jj)=FofPolicy_jj+beta*EVnextOfPolicy;
            Vhat(:,jj)     =FofPolicy_jj+beta0beta*EVnextOfPolicy;
        end
    end

    V=reshape(Vunderbar,[n_a,N_j]);

elseif N_z==0 && N_e>0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_e,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, vfoptions.n_e,N_j,vfoptions);

    Vunderbar=zeros(N_a,N_e,N_j,'gpuArray');
    Vhat=zeros(N_a,N_e,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));

        if jj==N_j
            Vunderbar(:,:,jj)=FofPolicy_jj;
            Vhat(:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
            beta0beta=beta0*beta;

            EVnext=sum(Vunderbar(:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-1),2); % expectation over iid e

            if N_d==0
                optaprime=PolicyIndexesKron(1,:,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
            end

            EVnextOfPolicy=reshape(EVnext(reshape(optaprime,[N_a*N_e,1])),[N_a,N_e]);

            Vunderbar(:,:,jj)=FofPolicy_jj+beta*EVnextOfPolicy;
            Vhat(:,:,jj)     =FofPolicy_jj+beta0beta*EVnextOfPolicy;
        end
    end

    V=reshape(Vunderbar,[n_a,vfoptions.n_e,N_j]);

elseif N_z>0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_z,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j,vfoptions);

    Vunderbar=zeros(N_a,N_z,N_j,'gpuArray');
    Vhat=zeros(N_a,N_z,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));

        if jj==N_j
            Vunderbar(:,:,jj)=FofPolicy_jj;
            Vhat(:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
            beta0beta=beta0*beta;

            EVnext=Vunderbar(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z_from]
            EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)

            if N_d==0
                optaprime=PolicyIndexesKron(1,:,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
            end

            aprimez_index=reshape(optaprime,[N_a*N_z,1])+N_a*(kron((1:1:N_z)',ones(N_a,1,'gpuArray'))-1);
            EVnextOfPolicy=reshape(EVnext(aprimez_index),[N_a,N_z]);

            Vunderbar(:,:,jj)=FofPolicy_jj+beta*EVnextOfPolicy;
            Vhat(:,:,jj)     =FofPolicy_jj+beta0beta*EVnextOfPolicy;
        end
    end

    V=reshape(Vunderbar,[n_a,n_z,N_j]);

elseif N_z>0 && N_e>0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a,N_z*N_e,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1_e(Policy, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);

    Vunderbar=zeros(N_a,N_z,N_e,N_j,'gpuArray');
    Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);

        if jj==N_j
            Vunderbar(:,:,:,jj)=FofPolicy_jj;
            Vhat(:,:,:,jj)=FofPolicy_jj;
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
            beta0beta=beta0*beta;

            EVnext=sum(Vunderbar(:,:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-2),3); % expectation over iid e
            EVnext=EVnext.*shiftdim(pi_z_J(:,:,jj)',-1);
            EVnext(isnan(EVnext))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EVnext=sum(EVnext,2); % sum over z', leaving a singular second dimension

            if N_d==0
                optaprime=PolicyIndexesKron(1,:,:,:,jj);
            else
                optaprime=shiftdim(PolicyIndexesKron(2,:,:,:,jj),1);
            end

            aprimez_index=reshape(optaprime,[N_a*N_z*N_e,1])+N_a*(kron(kron(ones(N_e,1,'gpuArray'),(1:1:N_z)'),ones(N_a,1,'gpuArray'))-1);
            EVnextOfPolicy=reshape(EVnext(aprimez_index),[N_a,N_z,N_e]);

            Vunderbar(:,:,:,jj)=FofPolicy_jj+beta*EVnextOfPolicy;
            Vhat(:,:,:,jj)     =FofPolicy_jj+beta0beta*EVnextOfPolicy;
        end
    end

    V=reshape(Vunderbar,[n_a,n_z,vfoptions.n_e,N_j]);
end


end
