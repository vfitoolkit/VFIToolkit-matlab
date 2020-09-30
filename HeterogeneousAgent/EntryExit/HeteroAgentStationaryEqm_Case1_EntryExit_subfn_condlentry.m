function CondlEntryDecision=HeteroAgentStationaryEqm_Case1_EntryExit_subfn_condlentry(p, n_d, n_a, n_z, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions)
% Is just a copy-pase of HeteroAgentStationaryEqm_Case1_EntryExit_subfn(),
% which just outputs Parameters.(EntryExitParamNames.CondlEntryDecisions{1}) instead of usual
% output of p_eqm. Have removed bunch of lines of code that are not
% required as a result.

%% I'm being lazy, this should be outside the subfn, and just the needed parts passed to the subfunction (would be faster that way)
if isfield(heteroagentoptions,'specialgeneqmcondn')
    standardgeneqmcondnindex=zeros(1,length(GeneralEqmEqns));
    jj=1;
    GeneralEqmEqnParamNames_Full=GeneralEqmEqnParamNames;
    clear GeneralEqmEqnParamNames
    for ii=1:length(GeneralEqmEqns)
        if isnumeric(heteroagentoptions.specialgeneqmcondn{ii}) % numeric means equal to zero and is a standard GEqm
            standardgeneqmcondnindex(jj)=ii;
            GeneralEqmEqnParamNames(jj).Names=GeneralEqmEqnParamNames_Full(ii).Names;
            jj=jj+1;
        elseif strcmp(heteroagentoptions.specialgeneqmcondn{ii},'condlentry')
            CondlEntryCondnEqn=GeneralEqmEqns(ii);
            CondlEntryCondnEqnParamNames(1).Names=GeneralEqmEqnParamNames_Full(ii).Names;
        end
    end
end

%% 
for ii=1:length(GEPriceParamNames)
    Parameters.(GEPriceParamNames{ii})=p(ii);
end

if simoptions.endogenousexit==1
    [V,Policy,ExitPolicy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % With entry-exit will need to keep the value function to be able to evaluate entry condition.
    Parameters.(EntryExitParamNames.CondlProbOfSurvival{1})=1-ExitPolicy;
elseif simoptions.endogenousexit==2 % Mixture of both endog and exog exit (which occurs at end of period)
    [V,Policy,PolicyWhenExiting,ExitPolicy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % With entry-exit will need to keep the value function to be able to evaluate entry condition.
    Parameters.(EntryExitParamNames.CondlProbOfSurvival{1})=1-ExitPolicy;
%     Parameters.(EntryExitParamNames.CondlProbOfSurvival{1})=vfoptions.exitprobabilities(1)+vfoptions.exitprobabilities(2)*(1-gather(ExitPolicy)); %1-gather(ExitPolicy);
else
    [V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % With entry-exit will need to keep the value function to be able to evaluate entry condition.
end

% Evaluate the conditional equilibrium condition on the (potential entrants) grid,
% and where it is >=0 use this to set new values for the
% EntryExitParamNames.CondlEntryDecisions parameter.
CondlEntryCondnEqnParamsVec=CreateVectorFromParams(Parameters, CondlEntryCondnEqnParamNames(1).Names);
CondlEntryCondnEqnParamsCell=cell(length(CondlEntryCondnEqnParamsVec),1);
for jj=1:length(CondlEntryCondnEqnParamsVec)
    CondlEntryCondnEqnParamsCell(jj,1)={CondlEntryCondnEqnParamsVec(jj)};
end

CondlEntryDecision=(CondlEntryCondnEqn{1}(V,p,CondlEntryCondnEqnParamsCell{:}) >=0);

end
