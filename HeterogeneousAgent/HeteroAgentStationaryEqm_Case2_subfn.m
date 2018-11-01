function GeneralEqmConditions=HeteroAgentStationaryEqm_Case2_subfn(p, V0Kron, n_d, n_a, n_s, pi_s, d_grid, a_grid, s_grid, Phi_aprimeKron, Case2_Type, ReturnFn, SSvaluesFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, SSvalueParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_s=prod(n_s);

%% 
for ii=1:length(GEPriceParamNames)
    Parameters.(GEPriceParamNames{ii})=p(ii);
end

[~, Policy]=ValueFnIter_Case2(V0Kron, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s, Phi_aprimeKron, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);


%Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
StationaryDistKron=StationaryDist_Case2(Policy,Phi_aprimeKron,Case2_Type,n_d,n_a,n_s,pi_s,simoptions);
SSvalues_AggVars=SSvalues_AggVars_Case2(StationaryDistKron, Policy, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_s, d_grid, a_grid, s_grid,2); % The 2 is for Parallel (use GPU) % The 2 is for Parallel (use GPU)

% The following line is often a useful double-check if something is going wrong.
%    SSvalues_AggVars

% use of real() is a hack that could disguise errors, but I couldn't
% find why matlab was treating output as complex
GeneralEqmConditionsVec=real(GeneralEqmConditions_Case2(SSvalues_AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));

if heteroagentoptions.multiGEcriterion==0 
    GeneralEqmConditions=sum(abs(GeneralEqmConditionsVec));
elseif heteroagentoptions.multiGEcriterion==1 %the measure of general eqm is to take the sum of squares of each of the general eqm conditions being satisfied 
    GeneralEqmConditions=sum(GeneralEqmConditionsVec.^2);                                                                                                         
end

GeneralEqmConditions=gather(GeneralEqmConditions);

if heteroagentoptions.verbose==1
    fprintf('Current Aggregates: \n')
    SSvalues_AggVars
    fprintf('Current GE prices and GeneralEqmConditionsVec. \n')
    p
    GeneralEqmConditionsVec
end

end