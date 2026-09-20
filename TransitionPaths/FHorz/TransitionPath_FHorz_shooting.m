function [PricePathOld,GEcondnPath]=TransitionPath_FHorz_shooting(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_semiz,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions)
% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

if transpathoptions.verbose==1
    % Set up some things to be used later
    pathnametitles=strjoin(PricePathNames,' ');
    wpathnametitle=10*length(PricePathNames); % roughly the space that will use to print the prices themselves
    % fprintf('%-*s || %-*s \n',wpathnametitle,'Old',wpathnametitle,'New')
    % fprintf('%-*s || %-*s \n',wpathnametitle,pathnametitles,wpathnametitle,pathnametitles)
end

%%
% interpret jequalOneDist input
if transpathoptions.trivialjequalonedist==0
    % the jequalOneDist input is actually jequalOneDist_T
    jequalOneDist_T=jequalOneDist;
    jequalOneDist=jequalOneDist_T(:,1);
end

%%

PricePathNew=zeros(size(PricePathOld),'gpuArray');
PricePathNew(T,:)=PricePathOld(T,:);
AggVarsPath=zeros(T-1,length(FnsToEvaluateCell),'gpuArray'); % Note: does not include the final AggVars, might be good to add them later as a way to make if obvious to user it things are incorrect
GEcondnPath=zeros(T-1,length(GeneralEqmEqnsCell),'gpuArray');

% Setup, the shapes of various of these objects vary depending on the setting
[PolicyIndexesPath,N_probs,II1,II2,exceptlastj,exceptfirstj,justfirstj]=TransitionPath_FHorz_substeps_Step0_setup(l_d,l_aprime,N_a,N_semiz,N_z,N_e,N_j,T,transpathoptions,vfoptions,simoptions);
% Note: some of these outputs are empty, depending on the setting
% PolicyIndexesPath is just pre-allocating matrices of the appropriate size, contains zeros

%% Semi-exogenous state: prepare the value-function-form grids/transitions (the substeps below dispatch to SemiExo variants)
if N_semiz>0
    pi_e_J_vf=pi_e_J;
    if transpathoptions.fastOLG==1 && N_e>0 && N_z==0
        pi_e_J_vf=reshape(pi_e_J,[N_a*N_j,1,N_e]); % SemiExo value fn keeps the bothz dim, so needs (a,j)-by-1-by-e even when N_z==0
    end
end

%%
PricePathDist=Inf;
GEcondnPathDist=Inf;
itercounter=1;
converged=0;
while itercounter<=transpathoptions.maxiter % convergence is tested further down, at the point where the distances are known, so that the loop stops on the path it just evaluated

    %% Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    if N_semiz==0
        [~,PolicyIndexesPath]=TransitionPath_FHorz_substeps_Step1_ValueFnIter(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_j,N_z,N_e,d_gridvals, a_grid, z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J,ReturnFn,DiscountFactorParamNames, ReturnFnParamNames, transpathoptions,vfoptions);
    else
        [~,PolicyIndexesPath]=TransitionPath_FHorz_substeps_Step1_ValueFnIter_SemiExo(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,simoptions.setup_semiexo.n_d1,simoptions.setup_semiexo.n_d2,n_a,simoptions.n_semiz,n_z,n_e,N_j,N_z,N_e,simoptions.setup_semiexo.d1_gridvals,simoptions.setup_semiexo.d2_gridvals,a_grid,z_gridvals_J,semiz_gridvals_J,e_gridvals_J,pi_z_J,pi_semiz_J,pi_e_J_vf,ReturnFn,DiscountFactorParamNames, ReturnFnParamNames, transpathoptions,vfoptions);
    end

    %% Modify PolicyIndexesPath into forms needed for forward iteration
    if N_semiz==0
        [PolicyPath_ForAgentDistIter,PolicyProbsPath,PolicyValuesPath]=TransitionPath_FHorz_substeps_Step2_AdjustPolicy(PolicyIndexesPath,T,Parameters,n_d,n_a,n_z,n_e,N_j,l_d,l_aprime,N_a,N_z,N_e,N_probs,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions);
    else
        [Policy_dsemiexoPath,Policy_aprimePath,PolicyProbsPath,PolicyValuesPath]=TransitionPath_FHorz_substeps_Step2_AdjustPolicy_SemiExo(PolicyIndexesPath,T,n_d,n_a,n_z,n_e,N_j,N_a,N_z,N_e,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions);
    end

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

        if N_semiz==0
            AgentDistnext=TransitionPath_FHorz_substeps_Step3tt_IterAgentDist(AgentDist,PolicyPath_ForAgentDistIter,PolicyProbsPath,tt,N_a,N_z,N_e,N_j,N_probs,pi_z_J,pi_z_J_sim,pi_e_J,pi_e_J_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDist,transpathoptions,simoptions);
        else
            AgentDistnext=TransitionPath_FHorz_substeps_Step3tt_IterAgentDist_SemiExo(AgentDist,Policy_dsemiexoPath,Policy_aprimePath,PolicyProbsPath,tt,N_a,N_z,N_e,N_j,N_probs,pi_z_J,pi_z_J_sim,pi_e_J,pi_e_J_sim,pi_semiz_J_sim,jequalOneDist,transpathoptions,simoptions);
        end

        %% AggVars
        if N_z==0 && N_e==0 && N_semiz==0
            AggVars=TransitionPath_FHorz_substeps_Step4tt_AggVars(AgentDist,AgeWeights,PolicyValuesPath(:,:,:,tt),tt,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_semiz,l_z,l_e,N_d,N_a,N_semiz,N_z,N_e,a_gridvals,semizze_gridvals_J_fastOLG,transpathoptions);
        else
            AggVars=TransitionPath_FHorz_substeps_Step4tt_AggVars(AgentDist,AgeWeights,PolicyValuesPath(:,:,:,:,tt),tt,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_semiz,l_z,l_e,N_d,N_a,N_semiz,N_z,N_e,a_gridvals,semizze_gridvals_J_fastOLG,transpathoptions);
        end

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
        % Evaluate the general eqm conditions, and based on them create PricePathNew (interpretation depends on transpathoptions)
        [PricePathNew_tt,GEcondnPath_tt]=updatePricePathNew_TPath_tt(Parameters,GeneralEqmEqnsCell,GeneralEqmEqnParamNames,PricePathOld(tt,:),itercounter,transpathoptions);
        PricePathNew(tt,:)=PricePathNew_tt;
        GEcondnPath(tt,:)=GEcondnPath_tt;

        % Sometimes, want to keep the AggVars to plot them
        if transpathoptions.graphaggvarspath==1
            for ii=1:length(AggVarNames)
                AggVarsPath(tt,ii)=AggVars.(AggVarNames{ii}).Mean;
            end
        end

        AgentDist=AgentDistnext;
    end


    %% Now update prices, give verbose feedback, and check for convergence

    % See how far apart the price paths are
    % A price path can reach somewhere the model cannot be solved, and the general eqm conditions then
    % come back non-finite. Stop rather than carry on: every later update just propagates it, and
    % there is no good path left to fall back to. Note that a NaN passes silently through any > or <=
    % test, so without this the iteration would either run out its full maxiter or, worse, look
    % converged and return the NaN path as the answer.
    if any(~isfinite(GEcondnPath),'all')
        error(['TransitionPath_FHorz_shooting: the general eqm conditions are NaN/Inf at iteration %i. ' ...
            'The price path has reached somewhere the model cannot be solved. Try a smaller factor ' ...
            'in GEnewprice3.howtoupdate, a less aggressive GEnewprice3.additionalfactor, or a ' ...
            'starting price path closer to the solution.'],itercounter)
    end

    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    % Notice that the distance is always calculated ignoring the time t=T periods, as these needn't ever converges
    % And how far the general eqm conditions are from zero. Scalarize across the general eqm eqns in each
    % time period the same way the stationary general eqm does, then take the L-Infinity norm over time
    % (the same norm as is used for the prices). GEcondnPath is the raw conditions, before the permute
    % and before updateaccuracycutoff is applied.
    if transpathoptions.multiGEcriterion==0
        GEcondnPathDist=max(sum(abs(transpathoptions.multiGEweights.*GEcondnPath),2));
    elseif transpathoptions.multiGEcriterion==1
        GEcondnPathDist=max(sqrt(sum(transpathoptions.multiGEweights.*(GEcondnPath.^2),2)));
    end

    if transpathoptions.verbose==1
        fprintf(' \n')
        fprintf('%-*s || %-*s \n',wpathnametitle,'Old',wpathnametitle,'New')
        fprintf('%-*s || %-*s \n',wpathnametitle,pathnametitles,wpathnametitle,pathnametitles)

        % Would be nice to have a way to get the iteration count without having the whole printout of path values (I think that would be useful?)
        [PricePathOld,PricePathNew]
    end

    % Create plots of the transition path (before we update pricepath)
    createTPathFeedbackPlots(PricePathNames,AggVarNames,GEeqnNames,PricePathOld,AggVarsPath,GEcondnPath,transpathoptions);

    % Update PricePathOld

    TransPathConvergence=max(PricePathDist/transpathoptions.toleranceGEprices,GEcondnPathDist/transpathoptions.toleranceGEcondns); % So when this gets to 1 we have convergence, we require convergence in both
    if transpathoptions.verbose==1
        fprintf('Number of iterations on transition path: %i \n',itercounter)
        if isfinite(transpathoptions.toleranceGEprices)
            fprintf('Current distance between old and new price path (in L-Infinity norm): %8.6f \n', PricePathDist)
        end
        fprintf('Current distance of the general eqm conditions from zero: %8.6f \n', GEcondnPathDist)
        fprintf('Ratio of current distance to the convergence tolerance: %.2f (convergence when reaches 1) \n',TransPathConvergence)
    end

    if transpathoptions.historyofpricepath==1
        % Store the whole history of the price path and save it every ten iterations
        PricePathHistory{itercounter,1}=PricePathDist;
        PricePathHistory{itercounter,2}=PricePathOld;
        if rem(itercounter,10)==1
            save ./SavedOutput/TransPath_Internal.mat PricePathHistory
        end
    end


    % Convergence. Tested here, after the distances are known but before the price path is updated,
    % so that what gets returned is the path whose general eqm conditions were actually evaluated.
    % Testing it at the top of the loop instead would leave the loop having applied one more update
    % than it checked, and so return a path one step past the GEcondnPath returned alongside it.
    if PricePathDist<=transpathoptions.toleranceGEprices && GEcondnPathDist<=transpathoptions.toleranceGEcondns
        converged=1;
        break
    end

    PricePathOld=updatePricePath(PricePathOld,PricePathNew,transpathoptions,T);

    itercounter=itercounter+1;


end

if converged==0
    warning(['TransitionPath_FHorz_shooting: reached maxiter (%i) without convergence; the general eqm ' ...
        'conditions are %8.6f from zero, against toleranceGEcondns=%g. Consider increasing ' ...
        'transpathoptions.maxiter, or adjusting the GEnewprice3.howtoupdate factors.'], ...
        transpathoptions.maxiter,GEcondnPathDist,transpathoptions.toleranceGEcondns)
end


end
