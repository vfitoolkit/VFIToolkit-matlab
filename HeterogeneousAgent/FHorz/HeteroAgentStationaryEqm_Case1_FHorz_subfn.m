function GeneralEqmConditions=HeteroAgentStationaryEqm_Case1_FHorz_subfn(p, jequaloneDist,AgeWeights, n_d, n_a, n_z, N_j, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, SSvaluesFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, SSvalueParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_p=prod(n_p);

l_p=length(n_p);

%% 
for ii=1:l_p
    Parameters.(GEPriceParamNames{ii})=p(ii);
end

[~, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

%Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
StationaryDistKron=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeights,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
SSvalues_AggVars=SSvalues_AggVars_FHorz_Case1(StationaryDistKron, Policy, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_z,N_j, d_grid, a_grid, z_grid,2); % The 2 is for Parallel (use GPU)

% The following line is often a useful double-check if something is going wrong.
%    SSvalues_AggVars

% use of real() is a hack that could disguise errors, but I couldn't
% find why matlab was treating output as complex
GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1(SSvalues_AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));

if heteroagentoptions.multiGEcriterion==0  
    GeneralEqmConditions=sum(abs(GeneralEqmConditionsVec));
elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    GeneralEqmConditions=sqrt(sum(GeneralEqmConditionsVec.^2));                                                                                                         
end

GeneralEqmConditions=gather(GeneralEqmConditions);

if heteroagentoptions.verbose==1
    [p,GeneralEqmConditions]
end

end