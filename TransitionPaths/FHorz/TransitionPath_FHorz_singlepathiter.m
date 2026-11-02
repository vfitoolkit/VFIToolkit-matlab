function [GEcondnPath,AggVarsPath,PolicyIndexesPath,PricePathNew,aprimeReferencePath,lsdiag]=TransitionPath_FHorz_singlepathiter(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, aprimeReferencePath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj)
% One iteration of the transition path at a given price path: the value function backwards over t,
% then the agent distribution forwards over t, evaluating the general eqm conditions as it goes.
% This is a pure function of PricePathOld given the setup: the agent distribution is reset to
% AgentDist_initial, every _tminus1 entry is reset from transpathoptions.initialvalues, and
% PolicyIndexesPath is only scratch that Step1_ValueFnIter overwrites. That is what lets the Newton
% options call it repeatedly at perturbed price paths to build a Jacobian.
%
% PricePathNew is an output rather than an input: it is only filled when
% transpathoptions.updatepert=1, which updates each price inside the loop over t. With updatepert=0
% (the default) the update happens afterwards, in updatePricePathNew_TPath_T, and PricePathNew comes
% back holding only the period-T terminal condition.
% itercounter is the current iteration, needed by the additional-factor ramp under updatepert=1.
%
% pi_e_J_vf is the form of pi_e_J the SemiExo value fn wants, set up once by the caller (it is only
% used when N_semiz>0).

PricePathNew=zeros(size(PricePathOld),'gpuArray');
PricePathNew(T,:)=PricePathOld(T,:);
AggVarsPath=zeros(T-1,length(AggVarNames),'gpuArray'); % Note: does not include the final AggVars
GEcondnPath=zeros(T-1,length(GeneralEqmEqnsCell),'gpuArray');

%%
% interpret jequalOneDist input
if transpathoptions.trivialjequalonedist==0
    % the jequalOneDist input is actually jequalOneDist_T
    jequalOneDist_T=jequalOneDist;
    jequalOneDist=jequalOneDist_T(:,1);
end

%% Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
% Skipped when the caller has already solved the value fn path itself and is handing in the finished
% PolicyIndexesPath. Only the triangular Jacobian build does that: it solves the backward pass for
% the periods a perturbed price can reach and reuses the baseline solve for the rest.
lsdiag=[]; % Step1 sets it; this covers the skip path and the SemiExo branch, where there is no local search
if ~isfield(transpathoptions,'skipStep1ValueFnIter')
    transpathoptions.skipStep1ValueFnIter=0;
end
if transpathoptions.skipStep1ValueFnIter==0
    if N_semiz==0
        [~,PolicyIndexesPath,aprimeReferencePath,lsdiag]=TransitionPath_FHorz_substeps_Step1_ValueFnIter(T,PolicyIndexesPath,aprimeReferencePath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_j,N_z,N_e,d_gridvals, a_grid, z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J,ReturnFn,DiscountFactorParamNames, ReturnFnParamNames, transpathoptions,vfoptions);
    else
        [~,PolicyIndexesPath]=TransitionPath_FHorz_substeps_Step1_ValueFnIter_SemiExo(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,simoptions.setup_semiexo.n_d1,simoptions.setup_semiexo.n_d2,n_a,simoptions.n_semiz,n_z,n_e,N_j,N_z,N_e,simoptions.setup_semiexo.d1_gridvals,simoptions.setup_semiexo.d2_gridvals,a_grid,z_gridvals_J,semiz_gridvals_J,e_gridvals_J,pi_z_J,pi_semiz_J,pi_e_J_vf,ReturnFn,DiscountFactorParamNames, ReturnFnParamNames, transpathoptions,vfoptions);
    end
end

%% Modify PolicyIndexesPath into forms needed for forward iteration
% Done period by period, inside the loop over t below: Step2 is called on one period's slice of PolicyIndexesPath
% (with T=2, so its T-1 periods are just that one), and the whole-path versions of the arrays it makes are never
% built. A single operation on a whole-path array that size was seen to return zeros on the GPU (a 12 GB PolicyPath).
% PolicyIndexesPath is (rows,a,j,[z],[e],t) for fastOLG and (rows,a,[z],[e],j,t) for slowOLG, semiz counting as z.
nPolicyDims=4+(N_z>0 || N_semiz>0)+(N_e>0);
slicePolicy=repmat({':'},1,nPolicyDims-1);
% Step2 is handed the Parameters as they are here, before the loop over t changes them (as when it was called once on
% the whole path). With an experience asset it sets each period's prices and parameters itself, from the PricePath
% and ParamPath rows of that period, and reads that period's z grid.
Parameters_Step2=Parameters;
simoptions_Step2=simoptions;
PricePath_Step2=PricePathOld;
ParamPath_Step2=ParamPath;

%% Iterate forward over t: iterate agent dist, calculate aggvars, evaluate general eqm
% Call AgentDist the current periods distn and AgentDistnext the next periods distn which we must calculate
AgentDist=AgentDist_initial;

% Initialise _tminus1 entries in Parameters from initialvalues (used at tt=1)
if use_tminus1price==1
    for pp=1:length(tminus1priceNames)
        Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
    end
end
if use_tminus1params==1
    for pp=1:length(tminus1paramNames)
        Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
    end
end
if use_tminus1AggVars==1
    for pp=1:length(tminus1AggVarsNames)
        Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
    end
end
if use_stockvars==1
    for pp=1:length(stockvarsNames)
        Parameters.([stockvarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(stockvarsNames{pp});
    end
end

for tt=1:T-1
    %% Setup the Parameters for period tt

    % Get t-1 PricePath, ParamPath and AggVars before we update them
    if tt>1
        if use_tminus1price==1
            for pp=1:length(tminus1priceNames)
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
            end
        end
        if use_tminus1params==1
            for pp=1:length(tminus1paramNames)
                Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
            end
        end
        if use_tminus1AggVars==1
            for pp=1:length(tminus1AggVarsNames)
                % The AggVars have not yet been updated, so they still contain previous period values
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
            end
        end
        if use_stockvars==1 % Comes from PricePathNew, unlike the _tminus1price, which comes from the PricePathOld
            % Only filled under updatepert=1, which TransitionPath_Case1_FHorz enforces whenever there are stockvars
            for pp=1:length(stockvarsNames)
                Parameters.([stockvarsNames{pp},'_tminus1'])=PricePathNew(tt-1,PricePathSizeVec(1,stockvarInPricePathNames(pp)):PricePathSizeVec(2,stockvarInPricePathNames(pp)));
            end
        end
    end

    % Update current PricePath and ParamPath
    for pp=1:length(PricePathNames)
        Parameters.(PricePathNames{pp})=PricePathOld(tt,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp));
    end
    for pp=1:length(ParamPathNames)
        Parameters.(ParamPathNames{pp})=ParamPath(tt,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp));
    end

    % Get t+1 PricePath
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
        end
    end

    %% Get the current optimal policy, and iterate the agent dist
    if transpathoptions.trivialjequalonedist==0
        jequalOneDist=jequalOneDist_T(:,tt+1);  % Note: t+1 as we are about to create the next period AgentDist
    end
    if simoptions.fastOLG==0 || N_e>0
        AgeWeights=AgeWeights_T(:,:,tt);
    else % simoptions.fastOLG==1
        AgeWeights=AgeWeights_T(:,tt);
    end

    % This period's policy, in the forms the agent dist iteration and the AggVars need (see above the loop)
    PolicyIndexes_tt=PolicyIndexesPath(slicePolicy{:},tt);
    if N_semiz==0
        if simoptions.experienceasset>=1 || simoptions.experienceassetz>=1
            PricePath_Step2=PricePathOld(tt,:);
            ParamPath_Step2=ParamPath(tt,:);
            if simoptions.experienceassetz>=1 && transpathoptions.zpathtrivial==0
                simoptions_Step2.setup_experienceasset.z_gridvals_J_T=simoptions.setup_experienceasset.z_gridvals_J_T(:,:,:,tt);
            end
        end
        [PolicyForAgentDistIter_tt,PolicyProbs_tt,PolicyValues_tt]=TransitionPath_FHorz_substeps_Step2_AdjustPolicy(PolicyIndexes_tt,2,Parameters_Step2,PricePath_Step2,PricePathNames,PricePathSizeVec,ParamPath_Step2,ParamPathNames,ParamPathSizeVec,n_d,n_a,n_z,n_e,N_j,l_d,l_aprime,N_a,N_z,N_e,N_probs,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions_Step2);
        AgentDistnext=TransitionPath_FHorz_substeps_Step3tt_IterAgentDist(AgentDist,PolicyForAgentDistIter_tt,PolicyProbs_tt,tt,N_a,N_z,N_e,N_j,N_probs,pi_z_J,pi_z_J_sim,pi_e_J,pi_e_J_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDist,transpathoptions,simoptions);
    else
        [Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,PolicyValues_tt]=TransitionPath_FHorz_substeps_Step2_AdjustPolicy_SemiExo(PolicyIndexes_tt,2,n_d,n_a,n_z,n_e,N_j,N_a,N_z,N_e,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions);
        AgentDistnext=TransitionPath_FHorz_substeps_Step3tt_IterAgentDist_SemiExo(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,tt,N_a,N_z,N_e,N_j,N_probs,pi_z_J,pi_z_J_sim,pi_e_J,pi_e_J_sim,pi_semiz_J_sim,jequalOneDist,transpathoptions,simoptions);
    end

    %% AggVars
    % simoptions.fastOLG tells Step4tt how the dist is laid out ((a,j,z) when =1, (a,z,j) when =0), with
    % semiz as part of z either way
    AggVars=TransitionPath_FHorz_substeps_Step4tt_AggVars(AgentDist,AgeWeights,PolicyValues_tt,tt,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_semiz,l_z,l_e,N_d,N_a,N_semiz,N_z,N_e,a_gridvals,semizze_gridvals_J_fastOLG,transpathoptions,simoptions.fastOLG);

    for ff=1:length(AggVarNames) % Note: needed for _tminus1 as well as GeneralEqmEqns
        Parameters.(AggVarNames{ff})=AggVars.(AggVarNames{ff}).Mean;
    end

    %% Intermediate Eqns
    if transpathoptions.useintermediateEqns==1
        % Note: intermediateEqns just take in things from the Parameters structure, as do GeneralEqmEqns (AggVars get put into structure), hence just use the GeneralEqmConditions_Case1_v3g().
        intEqnnames=fieldnames(transpathoptions.intermediateEqns);
        intermediateEqnsVec=zeros(1,length(intEqnnames));
        % Do the intermediateEqns, in order
        for gg=1:length(intEqnnames)
            intermediateEqnsVec(gg)=real(GeneralEqmConditions_Case1_v3g(transpathoptions.intermediateEqnsCell{gg}, transpathoptions.intermediateEqnParamNames(gg).Names, Parameters));
            Parameters.(intEqnnames{gg})=intermediateEqnsVec(gg);
        end
    end

    %% General Eqm Eqns
    if transpathoptions.updatepert==1
        % Evaluate the general eqm conditions, and based on them create PricePathNew (interpretation depends on transpathoptions)
        [PricePathNew_tt,GEcondnPath_tt]=updatePricePathNew_TPath_tt(Parameters,GeneralEqmEqnsCell,GeneralEqmEqnParamNames,PricePathOld(tt,:),itercounter,transpathoptions);
        PricePathNew(tt,:)=PricePathNew_tt;
        GEcondnPath(tt,:)=GEcondnPath_tt;
    else
        % Just evaluate the general eqm conditions for this period. Creating PricePathNew from
        % them happens after the tt loop, in updatePricePathNew_TPath_T, because the Newton
        % options need the conditions from every period before they can produce an update.
        for gg=1:length(GeneralEqmEqnsCell)
            % Note: _v3 rather than _v3g, so on CPU rather than GPU
            GEcondnPath(tt,gg)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
            % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
        end
    end

    % Sometimes, want to keep the AggVars to plot them
    if transpathoptions.graphaggvarspath==1
        for ii=1:length(AggVarNames)
            AggVarsPath(tt,ii)=AggVars.(AggVarNames{ii}).Mean;
        end
    end

    AgentDist=AgentDistnext;
end

end
