function GeneralEqmConditions=HeteroAgentStationaryEqm_Case2_subfn(p, n_d, n_a, n_s, pi_s, d_grid, a_grid, s_grid, Phi_aprimeKron, Case2_Type, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_s=prod(n_s);

%% 
for ii=1:length(GEPriceParamNames)
    Parameters.(GEPriceParamNames{ii})=p(ii);
end

[~, Policy]=ValueFnIter_Case2(n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s, Phi_aprimeKron, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);


%Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
StationaryDistKron=StationaryDist_Case2(Policy,Phi_aprimeKron,Case2_Type,n_d,n_a,n_s,pi_s,simoptions);
AggVars=EvalFnOnAgentDist_AggVars_Case2(StationaryDistKron, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_s, d_grid, a_grid, s_grid, simoptions.parallel);

% The following line is often a useful double-check if something is going wrong.
%    SSvalues_AggVars

% use of real() is a hack that could disguise errors, but I couldn't
% find why matlab was treating output as complex
GeneralEqmConditionsVec=real(GeneralEqmConditions_Case2(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames, simoptions.parallel));

if heteroagentoptions.multiGEcriterion==0 
    GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
elseif heteroagentoptions.multiGEcriterion==1 %the measure of general eqm is to take the sum of squares of each of the general eqm conditions being satisfied 
    GeneralEqmConditions=sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2));                                                                                                         
end

GeneralEqmConditions=gather(GeneralEqmConditions);

if heteroagentoptions.verbose==1
    fprintf('Current Aggregates: \n')
    AggVars
    fprintf('Current GE prices and GeneralEqmConditionsVec. \n')
    p
    GeneralEqmConditionsVec
end

end
