function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V and Valt from a given Policy when the model uses Quasi-Hyperbolic discounting.
%
% Returns [V, Valt] matching the convention of ValueFnIter_FHorz_QuasiHyperbolic:
%   Sophisticated: V = Vhat (QH-discounted from current self's perspective)
%                  Valt = Vunderbar (realised continuation under future selves' own QH choices)
%   Naive:         V = Vtilde (QH-discounted, agent's perceived value at Policy)
%                  Valt = exponential-discounter value computed at Policyalt
%                  Requires vfoptions.Policyalt (the exponential-discounter argmax,
%                  available as the 4th output of ValueFnIter_FHorz_QuasiHyperbolic for Naive).

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');
if ~isNaive && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    error('vfoptions.quasi_hyperbolic must be ''Naive'' or ''Sophisticated''')
end

%% Naive requires Policyalt
if isNaive
    if ~isfield(vfoptions,'Policyalt')
        error('ValueFnFromPolicy_FHorz_QuasiHyperbolic (Naive): vfoptions.Policyalt is required. Naive QH stores Policy at the QH (beta0*beta) argmax but V is the exponential-discounter value at the std argmax. To reconstruct V from policies alone, pass the exponential-discounter argmax (Policyalt) via vfoptions.Policyalt. It is returned as the 4th output of ValueFnIter_FHorz_QuasiHyperbolic for Naive.')
    end
    Policyalt=gpuArray(vfoptions.Policyalt);
end

%% Dispatch to SemiExo subfn if n_semiz>0
if prod(vfoptions.n_semiz)>0
    [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

%% Scope limits -- plain case only for now (no ExpAsset family)
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
if vfoptions.gridinterplayer==1 && length(n_a)>=2
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic: gridinterplayer with l_a>=2 (GI2A) and dual {V,Valt} output is not yet implemented.')
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

%% GI1 variant: replace integer-aprime lookup with L2-interpolated lookup.
% Mirrors the non-GI structure below, but EVnext is read at (alower, alower+1)
% with weights from L2: PolicyProbs(:,...,2) = (L2-1)/(n2short+1).
if vfoptions.gridinterplayer==1
    n2short=vfoptions.ngridinterp;
    index_a1=1+(N_d>0);

    if N_z==0 && N_e==0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,1,3]);
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, 1, N_j, vfoptions);
        alower=reshape(PolicyIndexesKron(index_a1,:,:),[N_a,N_j]);
        L2    =reshape(PolicyIndexesKron(end,      :,:),[N_a,N_j]);
        PolicyProbs=zeros(N_a,N_j,2,'gpuArray');
        PolicyProbs(:,:,2)=(L2-1)/(n2short+1);
        PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2);

        if isNaive
            PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
            PolicyaltValuesPermute=permute(PolicyaltValues,[2,1,3]);
            PolicyIndexesKronAlt=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, 1, N_j, vfoptions);
            alower_alt=reshape(PolicyIndexesKronAlt(index_a1,:,:),[N_a,N_j]);
            L2_alt    =reshape(PolicyIndexesKronAlt(end,      :,:),[N_a,N_j]);
            PolicyProbs_alt=zeros(N_a,N_j,2,'gpuArray');
            PolicyProbs_alt(:,:,2)=(L2_alt-1)/(n2short+1);
            PolicyProbs_alt(:,:,1)=1-PolicyProbs_alt(:,:,2);
            Valt=zeros(N_a,N_j,'gpuArray');
            Vtilde=zeros(N_a,N_j,'gpuArray');
        else
            Vunderbar=zeros(N_a,N_j,'gpuArray');
            Vhat=zeros(N_a,N_j,'gpuArray');
        end

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            if isNaive
                FofPolicyalt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyaltValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            end

            if jj==N_j
                if isNaive
                    Valt(:,jj)=FofPolicyalt_jj;
                    Vtilde(:,jj)=FofPolicy_jj;
                else
                    Vunderbar(:,jj)=FofPolicy_jj;
                    Vhat(:,jj)=FofPolicy_jj;
                end
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
                beta0beta=beta0*beta;

                if isNaive
                    EVnext=Valt(:,jj+1);
                    bL =alower(:,jj);
                    bLa=alower_alt(:,jj);
                    EVnextAtPolicy   =PolicyProbs(:,jj,1)    .*EVnext(bL ) +PolicyProbs(:,jj,2)    .*EVnext(bL +1);
                    EVnextAtPolicyalt=PolicyProbs_alt(:,jj,1).*EVnext(bLa) +PolicyProbs_alt(:,jj,2).*EVnext(bLa+1);
                    Valt(:,jj)  =FofPolicyalt_jj +beta    *EVnextAtPolicyalt;
                    Vtilde(:,jj)=FofPolicy_jj    +beta0beta*EVnextAtPolicy;
                else
                    EVnext=Vunderbar(:,jj+1);
                    bL=alower(:,jj);
                    EVnextAtPolicy=PolicyProbs(:,jj,1).*EVnext(bL)+PolicyProbs(:,jj,2).*EVnext(bL+1);
                    Vunderbar(:,jj)=FofPolicy_jj+beta    *EVnextAtPolicy;
                    Vhat(:,jj)     =FofPolicy_jj+beta0beta*EVnextAtPolicy;
                end
            end
        end

        if isNaive
            V   =reshape(Vtilde,[n_a,N_j]);
            Valt=reshape(Valt, [n_a,N_j]);
        else
            V   =reshape(Vhat,     [n_a,N_j]);
            Valt=reshape(Vunderbar,[n_a,N_j]);
        end

    elseif N_z==0 && N_e>0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, vfoptions.n_e, N_j, vfoptions);
        alower=reshape(PolicyIndexesKron(index_a1,:,:,:),[N_a,N_e,N_j]);
        L2    =reshape(PolicyIndexesKron(end,      :,:,:),[N_a,N_e,N_j]);
        PolicyProbs=zeros(N_a,N_e,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,2)=(L2-1)/(n2short+1);
        PolicyProbs(:,:,:,1)=1-PolicyProbs(:,:,:,2);

        if isNaive
            PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
            PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
            PolicyIndexesKronAlt=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, vfoptions.n_e, N_j, vfoptions);
            alower_alt=reshape(PolicyIndexesKronAlt(index_a1,:,:,:),[N_a,N_e,N_j]);
            L2_alt    =reshape(PolicyIndexesKronAlt(end,      :,:,:),[N_a,N_e,N_j]);
            PolicyProbs_alt=zeros(N_a,N_e,N_j,2,'gpuArray');
            PolicyProbs_alt(:,:,:,2)=(L2_alt-1)/(n2short+1);
            PolicyProbs_alt(:,:,:,1)=1-PolicyProbs_alt(:,:,:,2);
            Valt=zeros(N_a,N_e,N_j,'gpuArray');
            Vtilde=zeros(N_a,N_e,N_j,'gpuArray');
        else
            Vunderbar=zeros(N_a,N_e,N_j,'gpuArray');
            Vhat=zeros(N_a,N_e,N_j,'gpuArray');
        end

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));
            if isNaive
                FofPolicyalt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyaltValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));
            end

            if jj==N_j
                if isNaive
                    Valt(:,:,jj)=FofPolicyalt_jj;
                    Vtilde(:,:,jj)=FofPolicy_jj;
                else
                    Vunderbar(:,:,jj)=FofPolicy_jj;
                    Vhat(:,:,jj)=FofPolicy_jj;
                end
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
                beta0beta=beta0*beta;

                if isNaive
                    EVnext=sum(Valt(:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-1),2); % integrate over iid e -> (N_a,1)
                    bL =alower(:,:,jj);
                    bLa=alower_alt(:,:,jj);
                    EVlower    =reshape(EVnext(bL ),  [N_a,N_e]);
                    EVupper    =reshape(EVnext(bL +1),[N_a,N_e]);
                    EVlower_alt=reshape(EVnext(bLa),  [N_a,N_e]);
                    EVupper_alt=reshape(EVnext(bLa+1),[N_a,N_e]);
                    EVnextAtPolicy   =PolicyProbs(:,:,jj,1)    .*EVlower    +PolicyProbs(:,:,jj,2)    .*EVupper;
                    EVnextAtPolicyalt=PolicyProbs_alt(:,:,jj,1).*EVlower_alt+PolicyProbs_alt(:,:,jj,2).*EVupper_alt;
                    Valt(:,:,jj)  =FofPolicyalt_jj+beta    *EVnextAtPolicyalt;
                    Vtilde(:,:,jj)=FofPolicy_jj   +beta0beta*EVnextAtPolicy;
                else
                    EVnext=sum(Vunderbar(:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-1),2);
                    bL=alower(:,:,jj);
                    EVlower=reshape(EVnext(bL ),  [N_a,N_e]);
                    EVupper=reshape(EVnext(bL +1),[N_a,N_e]);
                    EVnextAtPolicy=PolicyProbs(:,:,jj,1).*EVlower+PolicyProbs(:,:,jj,2).*EVupper;
                    Vunderbar(:,:,jj)=FofPolicy_jj+beta    *EVnextAtPolicy;
                    Vhat(:,:,jj)     =FofPolicy_jj+beta0beta*EVnextAtPolicy;
                end
            end
        end

        if isNaive
            V   =reshape(Vtilde,[n_a,vfoptions.n_e,N_j]);
            Valt=reshape(Valt, [n_a,vfoptions.n_e,N_j]);
        else
            V   =reshape(Vhat,     [n_a,vfoptions.n_e,N_j]);
            Valt=reshape(Vunderbar,[n_a,vfoptions.n_e,N_j]);
        end

    elseif N_z>0 && N_e==0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, N_j, vfoptions);
        alower=reshape(PolicyIndexesKron(index_a1,:,:,:),[N_a,N_z,N_j]);
        L2    =reshape(PolicyIndexesKron(end,      :,:,:),[N_a,N_z,N_j]);
        PolicyProbs=zeros(N_a,N_z,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,2)=(L2-1)/(n2short+1);
        PolicyProbs(:,:,:,1)=1-PolicyProbs(:,:,:,2);

        if isNaive
            PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
            PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
            PolicyIndexesKronAlt=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, n_z, N_j, vfoptions);
            alower_alt=reshape(PolicyIndexesKronAlt(index_a1,:,:,:),[N_a,N_z,N_j]);
            L2_alt    =reshape(PolicyIndexesKronAlt(end,      :,:,:),[N_a,N_z,N_j]);
            PolicyProbs_alt=zeros(N_a,N_z,N_j,2,'gpuArray');
            PolicyProbs_alt(:,:,:,2)=(L2_alt-1)/(n2short+1);
            PolicyProbs_alt(:,:,:,1)=1-PolicyProbs_alt(:,:,:,2);
            Valt=zeros(N_a,N_z,N_j,'gpuArray');
            Vtilde=zeros(N_a,N_z,N_j,'gpuArray');
        else
            Vunderbar=zeros(N_a,N_z,N_j,'gpuArray');
            Vhat=zeros(N_a,N_z,N_j,'gpuArray');
        end
        zidxoffset=N_a*gpuArray(0:N_z-1); % (1, N_z)

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
            if isNaive
                FofPolicyalt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyaltValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
            end

            if jj==N_j
                if isNaive
                    Valt(:,:,jj)=FofPolicyalt_jj;
                    Vtilde(:,:,jj)=FofPolicy_jj;
                else
                    Vunderbar(:,:,jj)=FofPolicy_jj;
                    Vhat(:,:,jj)=FofPolicy_jj;
                end
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
                beta0beta=beta0*beta;

                if isNaive
                    EVnext=Valt(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z_from]
                    EVnext(isnan(EVnext))=0;
                    lower_lin    =alower(:,:,jj)    +zidxoffset;
                    lower_lin_alt=alower_alt(:,:,jj)+zidxoffset;
                    EVnextAtPolicy   =PolicyProbs(:,:,jj,1)    .*EVnext(lower_lin    )+PolicyProbs(:,:,jj,2)    .*EVnext(lower_lin    +1);
                    EVnextAtPolicyalt=PolicyProbs_alt(:,:,jj,1).*EVnext(lower_lin_alt)+PolicyProbs_alt(:,:,jj,2).*EVnext(lower_lin_alt+1);
                    Valt(:,:,jj)  =FofPolicyalt_jj+beta    *EVnextAtPolicyalt;
                    Vtilde(:,:,jj)=FofPolicy_jj   +beta0beta*EVnextAtPolicy;
                else
                    EVnext=Vunderbar(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z_from]
                    EVnext(isnan(EVnext))=0;
                    lower_lin=alower(:,:,jj)+zidxoffset;
                    EVnextAtPolicy=PolicyProbs(:,:,jj,1).*EVnext(lower_lin)+PolicyProbs(:,:,jj,2).*EVnext(lower_lin+1);
                    Vunderbar(:,:,jj)=FofPolicy_jj+beta    *EVnextAtPolicy;
                    Vhat(:,:,jj)     =FofPolicy_jj+beta0beta*EVnextAtPolicy;
                end
            end
        end

        if isNaive
            V   =reshape(Vtilde,[n_a,n_z,N_j]);
            Valt=reshape(Valt, [n_a,n_z,N_j]);
        else
            V   =reshape(Vhat,     [n_a,n_z,N_j]);
            Valt=reshape(Vunderbar,[n_a,n_z,N_j]);
        end

    elseif N_z>0 && N_e>0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, [n_z,vfoptions.n_e], N_j, vfoptions);
        alower=reshape(PolicyIndexesKron(index_a1,:,:,:),[N_a,N_z,N_e,N_j]);
        L2    =reshape(PolicyIndexesKron(end,      :,:,:),[N_a,N_z,N_e,N_j]);
        PolicyProbs=zeros(N_a,N_z,N_e,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,:,2)=(L2-1)/(n2short+1);
        PolicyProbs(:,:,:,:,1)=1-PolicyProbs(:,:,:,:,2);

        if isNaive
            PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
            PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
            PolicyIndexesKronAlt=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, [n_z,vfoptions.n_e], N_j, vfoptions);
            alower_alt=reshape(PolicyIndexesKronAlt(index_a1,:,:,:),[N_a,N_z,N_e,N_j]);
            L2_alt    =reshape(PolicyIndexesKronAlt(end,      :,:,:),[N_a,N_z,N_e,N_j]);
            PolicyProbs_alt=zeros(N_a,N_z,N_e,N_j,2,'gpuArray');
            PolicyProbs_alt(:,:,:,:,2)=(L2_alt-1)/(n2short+1);
            PolicyProbs_alt(:,:,:,:,1)=1-PolicyProbs_alt(:,:,:,:,2);
            Valt=zeros(N_a,N_z,N_e,N_j,'gpuArray');
            Vtilde=zeros(N_a,N_z,N_e,N_j,'gpuArray');
        else
            Vunderbar=zeros(N_a,N_z,N_e,N_j,'gpuArray');
            Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');
        end
        zidxoffset=N_a*gpuArray(0:N_z-1); % (1, N_z)

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);
            if isNaive
                FofPolicyalt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyaltValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);
            end

            if jj==N_j
                if isNaive
                    Valt(:,:,:,jj)=FofPolicyalt_jj;
                    Vtilde(:,:,:,jj)=FofPolicy_jj;
                else
                    Vunderbar(:,:,:,jj)=FofPolicy_jj;
                    Vhat(:,:,:,jj)=FofPolicy_jj;
                end
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
                beta0beta=beta0*beta;

                if isNaive
                    EVnext=sum(Valt(:,:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-2),3); % integrate over iid e -> (N_a,N_z)
                    EVnext=EVnext*pi_z_J(:,:,jj)'; % (N_a, N_z)
                    EVnext(isnan(EVnext))=0;
                    lower_lin    =alower(:,:,:,jj)    +zidxoffset; % (N_a, N_z, N_e) — broadcasting
                    lower_lin_alt=alower_alt(:,:,:,jj)+zidxoffset;
                    EVnextAtPolicy   =PolicyProbs(:,:,:,jj,1)    .*EVnext(lower_lin    )+PolicyProbs(:,:,:,jj,2)    .*EVnext(lower_lin    +1);
                    EVnextAtPolicyalt=PolicyProbs_alt(:,:,:,jj,1).*EVnext(lower_lin_alt)+PolicyProbs_alt(:,:,:,jj,2).*EVnext(lower_lin_alt+1);
                    Valt(:,:,:,jj)  =FofPolicyalt_jj+beta    *EVnextAtPolicyalt;
                    Vtilde(:,:,:,jj)=FofPolicy_jj   +beta0beta*EVnextAtPolicy;
                else
                    EVnext=sum(Vunderbar(:,:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-2),3); % integrate over iid e -> (N_a,N_z)
                    EVnext=EVnext*pi_z_J(:,:,jj)'; % (N_a, N_z)
                    EVnext(isnan(EVnext))=0;
                    lower_lin=alower(:,:,:,jj)+zidxoffset;
                    EVnextAtPolicy=PolicyProbs(:,:,:,jj,1).*EVnext(lower_lin)+PolicyProbs(:,:,:,jj,2).*EVnext(lower_lin+1);
                    Vunderbar(:,:,:,jj)=FofPolicy_jj+beta    *EVnextAtPolicy;
                    Vhat(:,:,:,jj)     =FofPolicy_jj+beta0beta*EVnextAtPolicy;
                end
            end
        end

        if isNaive
            V   =reshape(Vtilde,[n_a,n_z,vfoptions.n_e,N_j]);
            Valt=reshape(Valt, [n_a,n_z,vfoptions.n_e,N_j]);
        else
            V   =reshape(Vhat,     [n_a,n_z,vfoptions.n_e,N_j]);
            Valt=reshape(Vunderbar,[n_a,n_z,vfoptions.n_e,N_j]);
        end
    end

    return
end

%% Backward iteration: compute both halves jointly.
% Sophisticated: Vunderbar (beta, drives recursion) and Vhat (beta0*beta, perceived).
% Naive:         Valt (beta at Policyalt, drives recursion) and Vtilde (beta0*beta at Policy).
if N_z==0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]); %[N_a,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, 1, N_j, vfoptions);

    if isNaive
        PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,0,N_j,d_grid,a_grid,vfoptions,1);
        PolicyaltValuesPermute=permute(PolicyaltValues,[2,1,3]);
        PolicyIndexesKronAlt=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, 1, N_j, vfoptions);
        Valt=zeros(N_a,N_j,'gpuArray');
        Vtilde=zeros(N_a,N_j,'gpuArray');
    else
        Vunderbar=zeros(N_a,N_j,'gpuArray');
        Vhat=zeros(N_a,N_j,'gpuArray');
    end

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
        if isNaive
            FofPolicyalt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyaltValuesPermute(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
        end

        if jj==N_j
            if isNaive
                Valt(:,jj)=FofPolicyalt_jj;
                Vtilde(:,jj)=FofPolicy_jj;
            else
                Vunderbar(:,jj)=FofPolicy_jj;
                Vhat(:,jj)=FofPolicy_jj;
            end
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
            beta0beta=beta0*beta;

            if isNaive
                EVnext=Valt(:,jj+1);
                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                    optaprime_alt=PolicyIndexesKronAlt(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                    optaprime_alt=shiftdim(PolicyIndexesKronAlt(2,:,:,jj),1);
                end
                EVnextOfPolicy   =reshape(EVnext(reshape(optaprime,    [N_a,1])),[N_a,1]);
                EVnextOfPolicyalt=reshape(EVnext(reshape(optaprime_alt,[N_a,1])),[N_a,1]);

                Valt(:,jj) =FofPolicyalt_jj+beta    *EVnextOfPolicyalt;
                Vtilde(:,jj)=FofPolicy_jj   +beta0beta*EVnextOfPolicy;
            else
                EVnext=Vunderbar(:,jj+1);

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                end

                aprimez_index=reshape(optaprime,[N_a,1]);
                EVnextOfPolicy=reshape(EVnext(aprimez_index),[N_a,1]);

                Vunderbar(:,jj)=FofPolicy_jj+beta    *EVnextOfPolicy;
                Vhat(:,jj)     =FofPolicy_jj+beta0beta*EVnextOfPolicy;
            end
        end
    end

    if isNaive
        V   =reshape(Vtilde,   [n_a,N_j]);
        Valt=reshape(Valt,    [n_a,N_j]);
    else
        V   =reshape(Vhat,     [n_a,N_j]);
        Valt=reshape(Vunderbar,[n_a,N_j]);
    end

elseif N_z==0 && N_e>0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_e,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, vfoptions.n_e, N_j, vfoptions);

    if isNaive
        PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
        PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
        PolicyIndexesKronAlt=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, vfoptions.n_e, N_j, vfoptions);
        Valt=zeros(N_a,N_e,N_j,'gpuArray');
        Vtilde=zeros(N_a,N_e,N_j,'gpuArray');
    else
        Vunderbar=zeros(N_a,N_e,N_j,'gpuArray');
        Vhat=zeros(N_a,N_e,N_j,'gpuArray');
    end

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));
        if isNaive
            FofPolicyalt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyaltValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));
        end

        if jj==N_j
            if isNaive
                Valt(:,:,jj)=FofPolicyalt_jj;
                Vtilde(:,:,jj)=FofPolicy_jj;
            else
                Vunderbar(:,:,jj)=FofPolicy_jj;
                Vhat(:,:,jj)=FofPolicy_jj;
            end
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
            beta0beta=beta0*beta;

            if isNaive
                EVnext=sum(Valt(:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-1),2); % expectation over iid e
                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                    optaprime_alt=PolicyIndexesKronAlt(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                    optaprime_alt=shiftdim(PolicyIndexesKronAlt(2,:,:,jj),1);
                end
                EVnextOfPolicy   =reshape(EVnext(reshape(optaprime,    [N_a*N_e,1])),[N_a,N_e]);
                EVnextOfPolicyalt=reshape(EVnext(reshape(optaprime_alt,[N_a*N_e,1])),[N_a,N_e]);

                Valt(:,:,jj) =FofPolicyalt_jj+beta    *EVnextOfPolicyalt;
                Vtilde(:,:,jj)=FofPolicy_jj   +beta0beta*EVnextOfPolicy;
            else
                EVnext=sum(Vunderbar(:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-1),2); % expectation over iid e

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                end

                EVnextOfPolicy=reshape(EVnext(reshape(optaprime,[N_a*N_e,1])),[N_a,N_e]);

                Vunderbar(:,:,jj)=FofPolicy_jj+beta    *EVnextOfPolicy;
                Vhat(:,:,jj)     =FofPolicy_jj+beta0beta*EVnextOfPolicy;
            end
        end
    end

    if isNaive
        V   =reshape(Vtilde,   [n_a,vfoptions.n_e,N_j]);
        Valt=reshape(Valt,    [n_a,vfoptions.n_e,N_j]);
    else
        V   =reshape(Vhat,     [n_a,vfoptions.n_e,N_j]);
        Valt=reshape(Vunderbar,[n_a,vfoptions.n_e,N_j]);
    end

elseif N_z>0 && N_e==0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_z,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, N_j, vfoptions);

    if isNaive
        PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
        PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
        PolicyIndexesKronAlt=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, n_z, N_j, vfoptions);
        Valt=zeros(N_a,N_z,N_j,'gpuArray');
        Vtilde=zeros(N_a,N_z,N_j,'gpuArray');
    else
        Vunderbar=zeros(N_a,N_z,N_j,'gpuArray');
        Vhat=zeros(N_a,N_z,N_j,'gpuArray');
    end
    z_kron=kron((1:1:N_z)',ones(N_a,1,'gpuArray'));

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
        if isNaive
            FofPolicyalt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyaltValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
        end

        if jj==N_j
            if isNaive
                Valt(:,:,jj)=FofPolicyalt_jj;
                Vtilde(:,:,jj)=FofPolicy_jj;
            else
                Vunderbar(:,:,jj)=FofPolicy_jj;
                Vhat(:,:,jj)=FofPolicy_jj;
            end
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
            beta0beta=beta0*beta;

            if isNaive
                EVnext=Valt(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z_from]
                EVnext(isnan(EVnext))=0;
                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                    optaprime_alt=PolicyIndexesKronAlt(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                    optaprime_alt=shiftdim(PolicyIndexesKronAlt(2,:,:,jj),1);
                end
                aprimez_index   =reshape(optaprime,    [N_a*N_z,1])+N_a*(z_kron-1);
                aprimez_index_alt=reshape(optaprime_alt,[N_a*N_z,1])+N_a*(z_kron-1);
                EVnextOfPolicy   =reshape(EVnext(aprimez_index   ),[N_a,N_z]);
                EVnextOfPolicyalt=reshape(EVnext(aprimez_index_alt),[N_a,N_z]);

                Valt(:,:,jj) =FofPolicyalt_jj+beta    *EVnextOfPolicyalt;
                Vtilde(:,:,jj)=FofPolicy_jj   +beta0beta*EVnextOfPolicy;
            else
                EVnext=Vunderbar(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z_from]
                EVnext(isnan(EVnext))=0;

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                end

                aprimez_index=reshape(optaprime,[N_a*N_z,1])+N_a*(z_kron-1);
                EVnextOfPolicy=reshape(EVnext(aprimez_index),[N_a,N_z]);

                Vunderbar(:,:,jj)=FofPolicy_jj+beta    *EVnextOfPolicy;
                Vhat(:,:,jj)     =FofPolicy_jj+beta0beta*EVnextOfPolicy;
            end
        end
    end

    if isNaive
        V   =reshape(Vtilde,   [n_a,n_z,N_j]);
        Valt=reshape(Valt,    [n_a,n_z,N_j]);
    else
        V   =reshape(Vhat,     [n_a,n_z,N_j]);
        Valt=reshape(Vunderbar,[n_a,n_z,N_j]);
    end

elseif N_z>0 && N_e>0

    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    l_daprime=size(PolicyValues,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a,N_z*N_e,l_d+l_a,N_j]
    PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, [n_z,vfoptions.n_e], N_j, vfoptions);

    if isNaive
        PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
        PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
        PolicyIndexesKronAlt=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, [n_z,vfoptions.n_e], N_j, vfoptions);
        Valt=zeros(N_a,N_z,N_e,N_j,'gpuArray');
        Vtilde=zeros(N_a,N_z,N_e,N_j,'gpuArray');
    else
        Vunderbar=zeros(N_a,N_z,N_e,N_j,'gpuArray');
        Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');
    end
    zN_e_kron=kron(kron(ones(N_e,1,'gpuArray'),(1:1:N_z)'),ones(N_a,1,'gpuArray'));

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);
        if isNaive
            FofPolicyalt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyaltValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);
        end

        if jj==N_j
            if isNaive
                Valt(:,:,:,jj)=FofPolicyalt_jj;
                Vtilde(:,:,:,jj)=FofPolicy_jj;
            else
                Vunderbar(:,:,:,jj)=FofPolicy_jj;
                Vhat(:,:,:,jj)=FofPolicy_jj;
            end
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
            beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
            beta0beta=beta0*beta;

            if isNaive
                EVnext=sum(Valt(:,:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-2),3);
                EVnext=EVnext.*shiftdim(pi_z_J(:,:,jj)',-1);
                EVnext(isnan(EVnext))=0;
                EVnext=sum(EVnext,2);

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                    optaprime_alt=PolicyIndexesKronAlt(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                    optaprime_alt=shiftdim(PolicyIndexesKronAlt(2,:,:,jj),1);
                end
                aprimez_index   =reshape(optaprime,    [N_a*N_z*N_e,1])+N_a*(zN_e_kron-1);
                aprimez_index_alt=reshape(optaprime_alt,[N_a*N_z*N_e,1])+N_a*(zN_e_kron-1);
                EVnextOfPolicy   =reshape(EVnext(aprimez_index   ),[N_a,N_z,N_e]);
                EVnextOfPolicyalt=reshape(EVnext(aprimez_index_alt),[N_a,N_z,N_e]);

                Valt(:,:,:,jj) =FofPolicyalt_jj+beta    *EVnextOfPolicyalt;
                Vtilde(:,:,:,jj)=FofPolicy_jj   +beta0beta*EVnextOfPolicy;
            else
                EVnext=sum(Vunderbar(:,:,:,jj+1).*shiftdim(vfoptions.pi_e_J(:,jj),-2),3); % expectation over iid e
                EVnext=EVnext.*shiftdim(pi_z_J(:,:,jj)',-1);
                EVnext(isnan(EVnext))=0;
                EVnext=sum(EVnext,2);

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                end

                aprimez_index=reshape(optaprime,[N_a*N_z*N_e,1])+N_a*(zN_e_kron-1);
                EVnextOfPolicy=reshape(EVnext(aprimez_index),[N_a,N_z,N_e]);

                Vunderbar(:,:,:,jj)=FofPolicy_jj+beta    *EVnextOfPolicy;
                Vhat(:,:,:,jj)     =FofPolicy_jj+beta0beta*EVnextOfPolicy;
            end
        end
    end

    if isNaive
        V   =reshape(Vtilde,   [n_a,n_z,vfoptions.n_e,N_j]);
        Valt=reshape(Valt,    [n_a,n_z,vfoptions.n_e,N_j]);
    else
        V   =reshape(Vhat,     [n_a,n_z,vfoptions.n_e,N_j]);
        Valt=reshape(Vunderbar,[n_a,n_z,vfoptions.n_e,N_j]);
    end
end


end
