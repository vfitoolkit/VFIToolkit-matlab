function GeneralEqmConditions=HeteroAgentStationaryEqm_Case2_FHorz_subfn(p, jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, pi_z, d_grid, a_grid, z_grid, Phi_aprimeFn, Case2_Type, ReturnFn, FnsToEvaluateFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);

%%
for ii=1:length(GEPriceParamNames)
    Parameters.(GEPriceParamNames{ii})=p(ii);
end

[~, Policy]=ValueFnIter_Case2_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, Phi_aprimeFn, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);

%Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
StationaryDistKron=StationaryDist_FHorz_Case2(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames,simoptions);
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case2(StationaryDistKron, Policy, FnsToEvaluateFn, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z,N_j, d_grid, a_grid, z_grid, simoptions, pi_z); % If using Age Dependent Variables then 'pi_z' is actually AgeDependentGridParamNames. If not then it will anyway be ignored.

% The following line is often a useful double-check if something is going wrong.
%    SSvalues_AggVars

% use of real() is a hack that could disguise errors, but I couldn't
% find why matlab was treating output as complex
GeneralEqmConditionsVec=real(GeneralEqmConditions_Case2(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));

if heteroagentoptions.multiGEcriterion==0 
    GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    GeneralEqmConditions=sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2));                                                                                                         
end

GeneralEqmConditions=gather(GeneralEqmConditions);

if heteroagentoptions.verbose==1
    fprintf('Current GE prices and GeneralEqmConditionsVec. \n')
    p
    GeneralEqmConditionsVec
end