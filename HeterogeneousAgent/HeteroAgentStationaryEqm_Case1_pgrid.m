function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_pgrid(n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluateFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)
% Solve for the stationary general equilbirium. Evaluates the general
% equilibrium conditions at the points on the price grid.

N_p=prod(n_p);
l_p=length(n_p);

p_grid=heteroagentoptions.pgrid;

%% 

if simoptions.parallel==2 || simoptions.parallel==4
    GeneralEqmConditionsKron=ones(N_p,l_p,'gpuArray');
else
    GeneralEqmConditionsKron=ones(N_p,l_p);
end

for p_c=1:N_p
    if heteroagentoptions.verbose==1
        fprintf('Stationary eqm: using price grid element %i out of %i \n', p_c, N_p)
    end

    % If z (and e) are determined in GE
    if heteroagentoptions.gridsinGE==1
        % Some of the shock grids depend on parameters that are determined in general eqm
        [z_grid, pi_z, vfoptions]=ExogShockSetup(n_z,z_grid,pi_z,Parameters,vfoptions,3);
        % Note: these are actually z_gridvals and pi_z
        simoptions.e_gridvals=vfoptions.e_gridvals; % Note, will be [] if no e
        simoptions.pi_e=vfoptions.pi_e; % Note, will be [] if no e
    end

    %Step 1: Solve the value fn iteration problem (given this price, indexed by p_c)
    %Calculate the price vector associated with p_c
    p_index=ind2sub_homemade(n_p,p_c);
    GEprices=nan(l_p,1);
    for ii=1:l_p
        if ii==1
            GEprices(ii)=p_grid(p_index(1));
        else
            GEprices(ii)=p_grid(sum(n_p(1:ii-1))+p_index(ii));
        end
        Parameters.(GEPriceParamNames{ii})=GEprices(ii);
    end
    
    [~,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn,Parameters, DiscountFactorParamNames,ReturnFnParamNames,vfoptions);

    %Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
    StationaryDistKron=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z,simoptions);
    
    AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDistKron, Policy, FnsToEvaluateFn, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions.parallel);
    
    % The following line is often a useful double-check if something is going wrong.
    %    AggVars
    
    % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
    if isstruct(GeneralEqmEqns)
        AggVarNames=fieldnames(AggVars); % Using GeneralEqmEqns as a struct presupposes using FnsToEvaluate (and hence AggVars) as a stuct
        for ii=1:length(AggVarNames)
            Parameters.(AggVarNames{ii})=AggVars.(AggVarNames{ii}).Mean;
        end
        GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns, Parameters));
    else
        GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1(AggVars,GEprices, GeneralEqmEqns, Parameters,GeneralEqmEqnInputNames, simoptions.parallel));
    end
    GeneralEqmConditionsKron(p_c,:)=GeneralEqmConditionsVec;
end

multiGEweightsKron=ones(N_p,1)*heteroagentoptions.multiGEweights;
if simoptions.parallel==2 || simoptions.parallel==4
    multiGEweightsKron=gpuArray(multiGEweightsKron);
end

if heteroagentoptions.multiGEcriterion==0 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(abs(multiGEweightsKron.*GeneralEqmConditionsKron),2));
elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(multiGEweightsKron.*(GeneralEqmConditionsKron.^2),2));                                                                                                         
end

%p_eqm_index=zeros(num_p,1);
if simoptions.parallel==2
    p_eqm_index=ind2sub_homemade_gpu(n_p,p_eqm_indexKron);
else
    p_eqm_index=ind2sub_homemade(n_p,p_eqm_indexKron);
end
if l_p>1
    if simoptions.parallel==2
        GeneralEqmConditions=nan(N_p,1+l_p,'gpuArray');
    else
        GeneralEqmConditions=nan(N_p,1+l_p);
    end
    if heteroagentoptions.multiGEcriterion==0
        GeneralEqmConditions(:,1)=sum(abs(multiGEweightsKron.*GeneralEqmConditionsKron),2);
    elseif heteroagentoptions.multiGEcriterion==1 %the measure of general eqm is to take the sum of squares of each of the general eqm conditions holding 
        GeneralEqmConditions(:,1)=sum(multiGEweightsKron.*(GeneralEqmConditionsKron.^2),2);
    end
    GeneralEqmConditions(:,2:end)=GeneralEqmConditionsKron;
    GeneralEqmConditions=reshape(multiGEweightsKron.*GeneralEqmConditions,[n_p,1+l_p]);
else
    GeneralEqmConditions=reshape(multiGEweightsKron.*GeneralEqmConditionsKron,[n_p,1]);
end

%Calculate the price associated with p_eqm_index
p_eqm=zeros(l_p,1);
for ii=1:l_p
    if ii==1
        p_eqm(ii)=p_grid(p_eqm_index(1));
    else
        p_eqm(ii)=p_grid(sum(n_p(1:ii-1))+p_eqm_index(ii));
    end
end

% Move results from gpu to cpu before returning them
p_eqm=gather(p_eqm);
p_eqm_index=gather(p_eqm_index);
GeneralEqmConditions=gather(GeneralEqmConditions);

end
