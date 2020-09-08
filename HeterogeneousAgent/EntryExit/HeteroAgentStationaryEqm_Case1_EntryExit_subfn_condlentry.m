function CondlEntryDecision=HeteroAgentStationaryEqm_Case1_EntryExit_subfn_condlentry(p, n_d, n_a, n_z, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions)
% Is just a copy-pase of HeteroAgentStationaryEqm_Case1_EntryExit_subfn(),
% which just outputs
% Parameters.(EntryExitParamNames.CondlEntryDecisions{1}) instead of usual
% output of p_eqm. Have commented out a bunch of lines of code that are not
% required as a result.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% I'm being lazy, this should be outside the subfn, and just the needed parts passed to the subfunction (would be faster that way)
% Figure out which general eqm conditions are normal
GeneralEqmConditionsVec=zeros(1,length(GeneralEqmEqns));
standardgeneqmcondnsused=0;
specialgeneqmcondnsused=0;
entrycondnexists=0; condlentrycondnexists=0;
if ~isfield(heteroagentoptions,'specialgeneqmcondn')
    standardgeneqmcondnindex=1:1:length(GeneralEqmEqns);
else
    standardgeneqmcondnindex=zeros(1,length(GeneralEqmEqns));
    jj=1;
    GeneralEqmEqnParamNames_Full=GeneralEqmEqnParamNames;
    clear GeneralEqmEqnParamNames
    for ii=1:length(GeneralEqmEqns)
        if isnumeric(heteroagentoptions.specialgeneqmcondn{ii}) % numeric means equal to zero and is a standard GEqm
            standardgeneqmcondnsused=1;
            standardgeneqmcondnindex(jj)=ii;
            GeneralEqmEqnParamNames(jj).Names=GeneralEqmEqnParamNames_Full(ii).Names;
            jj=jj+1;
        elseif strcmp(heteroagentoptions.specialgeneqmcondn{ii},'entry')
            specialgeneqmcondnsused=1;
            entrycondnexists=1;
            % currently 'entry' is the only kind of specialgeneqmcondn
            entrygeneqmcondnindex=ii;
            EntryCondnEqn=GeneralEqmEqns(ii);
            EntryCondnEqnParamNames(1).Names=GeneralEqmEqnParamNames_Full(ii).Names;
        elseif strcmp(heteroagentoptions.specialgeneqmcondn{ii},'condlentry')
            specialgeneqmcondnsused=1;
            condlentrycondnexists=1;
            condlentrygeneqmcondnindex=ii;
            CondlEntryCondnEqn=GeneralEqmEqns(ii);
            CondlEntryCondnEqnParamNames(1).Names=GeneralEqmEqnParamNames_Full(ii).Names;
        end
    end
    standardgeneqmcondnindex=standardgeneqmcondnindex(standardgeneqmcondnindex>0); % get rid of zeros at the end
    GeneralEqmEqns=GeneralEqmEqns(standardgeneqmcondnindex);
end

%% 
for ii=1:length(GEPriceParamNames)
    Parameters.(GEPriceParamNames{ii})=p(ii);
end

% [~,Policy]=ValueFnIter_Case1(V0Kron, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames,ReturnFnParamNames,vfoptions);
if simoptions.endogenousexit==1
    [V,Policy,ExitPolicy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % With entry-exit will need to keep the value function to be able to evaluate entry condition.
    Parameters.(EntryExitParamNames.CondlProbOfSurvival{1})=1-ExitPolicy;
elseif simoptions.endogenousexit==2 % Mixture of both endog and exog exit (which occurs at end of period)
    [V,Policy,PolicyWhenExiting,ExitPolicy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % With entry-exit will need to keep the value function to be able to evaluate entry condition.
    Parameters.(EntryExitParamNames.CondlProbOfSurvival{1})=vfoptions.exitprobabilities(1)+vfoptions.exitprobabilities(2)*(1-gather(ExitPolicy)); %1-gather(ExitPolicy);
else
    [V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % With entry-exit will need to keep the value function to be able to evaluate entry condition.
end

%Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
% StationaryDistKron=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z,simoptions);
StationaryDistKron=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z, simoptions,Parameters,EntryExitParamNames);
% Parameters.(EntryExitParamNames.MassOfExistingAgents{1})=MassOfExistingAgents;

if simoptions.endogenousexit==2
    AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDistKron, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions.parallel, simoptions, EntryExitParamNames, PolicyWhenExiting);
else
    AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDistKron, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions.parallel,simoptions,EntryExitParamNames);
end

% The following line is often a useful double-check if something is going wrong.
%    AggVars

% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
% GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames, simoptions.parallel));
if standardgeneqmcondnsused==1
    % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
    GeneralEqmConditionsVec(standardgeneqmcondnindex)=gather(real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames, simoptions.parallel)));
end
% Now fill in the 'non-standard' cases
% if specialgeneqmcondnsused==1
%     if condlentrycondnexists==1
        % Evaluate the conditional equilibrium condition on the (potential entrants) grid,
        % and where it is >=0 use this to set new values for the
        % EntryExitParamNames.CondlEntryDecisions parameter.
        CondlEntryCondnEqnParamsVec=CreateVectorFromParams(Parameters, CondlEntryCondnEqnParamNames(1).Names);
        CondlEntryCondnEqnParamsCell=cell(length(CondlEntryCondnEqnParamsVec),1);
        for jj=1:length(CondlEntryCondnEqnParamsVec)
            CondlEntryCondnEqnParamsCell(jj,1)={CondlEntryCondnEqnParamsVec(jj)};
        end
        
%         Parameters.(EntryExitParamNames.CondlEntryDecisions{1})=(CondlEntryCondnEqn{1}(V,p,CondlEntryCondnEqnParamsCell{:}) >=0);
        CondlEntryDecision=(CondlEntryCondnEqn{1}(V,p,CondlEntryCondnEqnParamsCell{:}) >=0);

%         GeneralEqmConditionsVec(condlentrygeneqmcondnindex)=0; % Because the EntryExitParamNames.CondlEntryDecisions is set to hold exactly we can consider this as contributing 0
%         if entrycondnexists==1
%             % Calculate the expected (based on entrants distn) value fn (note, DistOfNewAgents is the pdf, so this is already 'normalized' EValueFn.
%             EValueFn=sum(reshape(V,[numel(V),1]).*reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[numel(V),1]).*reshape(Parameters.(EntryExitParamNames.CondlEntryDecisions{1}),[numel(V),1]));
%             % @(EValueFn,ce)
%             % And use entrants distribution, not the stationary distn
%             GeneralEqmConditionsVec(entrygeneqmcondnindex)=real(GeneralEqmConditions_Case1(EValueFn,p, EntryCondnEqn, Parameters,EntryCondnEqnParamNames, simoptions.parallel));
%         end
%     else
%         if entrycondnexists==1
%             % Calculate the expected (based on entrants distn) value fn (note, DistOfNewAgents is the pdf, so this is already 'normalized' EValueFn.
%             EValueFn=sum(reshape(V,[numel(V),1]).*reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[numel(V),1]));
%             % @(EValueFn,ce)
%             % And use entrants distribution, not the stationary distn
%             GeneralEqmConditionsVec(entrygeneqmcondnindex)=real(GeneralEqmConditions_Case1(EValueFn,p, EntryCondnEqn, Parameters,EntryCondnEqnParamNames, simoptions.parallel));
%         end
%     end
% %     if entrycondnexists==1
% %         % Calculate the expected (based on entrants distn) value fn (note, DistOfNewAgents is the pdf, so this is already 'normalized' EValueFn.
% %         EValueFn=sum(reshape(V,[numel(V),1]).*reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[numel(V),1]));
% %         % @(EValueFn,ce)
% %         % And use entrants distribution, not the stationary distn
% %         GeneralEqmConditionsVec(entrygeneqmcondnindex)=real(GeneralEqmConditions_Case1(EValueFn,p, EntryCondnEqn, Parameters,EntryCondnEqnParamNames, simoptions.parallel));
% %     end
% end

% if heteroagentoptions.multiGEcriterion==0 %only used when there is only one price 
%     GeneralEqmConditions=sum(abs(GeneralEqmConditionsVec));
% elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
%     GeneralEqmConditions=sqrt(sum(GeneralEqmConditionsVec.^2));                                                                                                         
% end
% 
% GeneralEqmConditions=gather(GeneralEqmConditions);
% 
% if heteroagentoptions.verbose==1
%     fprintf('Current Aggregates: \n')
%     AggVars
%     fprintf('Current GE prices and GeneralEqmConditionsVec: \n')
%     p
%     GeneralEqmConditionsVec
% end

end
