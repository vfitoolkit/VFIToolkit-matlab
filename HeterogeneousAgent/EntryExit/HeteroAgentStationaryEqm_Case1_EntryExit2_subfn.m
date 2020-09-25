function GeneralEqmConditions=HeteroAgentStationaryEqm_Case1_EntryExit2_subfn(p, n_d, n_a, n_z, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% These next few lines should be outside the subfn
% To create initial guess use ('middle' of) the newborns distribution for seed point and do no burnin and short simulations (ignoring exit).
EntryDist=reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[N_a*N_z,1]);
[~,seedpoint_index]=max(abs(cumsum(EntryDist)-0.5));
simoptions.seedpoint=ind2sub_homemade([N_a,N_z],seedpoint_index); % Would obviously be better initial guess to do a bunch of different simulations for variety of points in the 'EntryDist'.
simoptions.simperiods=10^3;
simoptions.burnin=0;
if simoptions.verbose==1
    fprintf('Note: simoptions.iterate=1 is imposed/required when using simoptions.agententryandexit=2 \n')
end
ExitProb=Parameters.(EntryExitParamNames.ProbOfDeath{1});

%% 
for ii=1:length(GEPriceParamNames)
    Parameters.(GEPriceParamNames{ii})=p(ii);
end

[V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

%Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
PolicyKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z);%,simoptions);
if simoptions.parallel~=2 % To cover the case when using gpu to solve value fn, but cpu to solve agent dist
    PolicyKron=gather(PolicyKron);
end
StationaryDistKron=StationaryDist_Case1_Simulation_raw(PolicyKron,N_d,N_a,N_z,pi_z, simoptions);
StationaryDistKron=StationaryDist_Case1_Iteration_EntryExit2_raw(StationaryDistKron,PolicyKron,N_d,N_a,N_z,pi_z,ExitProb,EntryDist,simoptions);

AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDistKron, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions.parallel,simoptions,EntryExitParamNames);

% The following line is often a useful double-check if something is going wrong.
%    AggVars

% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
% GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames, simoptions.parallel));
% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
GeneralEqmConditionsVec=gather(real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames, simoptions.parallel)));

if heteroagentoptions.multiGEcriterion==0 %only used when there is only one price 
    GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    GeneralEqmConditions=sqrt(sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2)));                                                                                                         
end

GeneralEqmConditions=gather(GeneralEqmConditions);

if heteroagentoptions.verbose==1
    fprintf('Current Aggregates: \n')
    AggVars
    fprintf('Current GE prices and GeneralEqmConditionsVec: \n')
    p
    GeneralEqmConditionsVec
end

end
