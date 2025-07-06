function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_FHorz_pgrid(jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, n_p, pi_z_J, d_grid, a_grid, z_gridvals_J, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)
% Solve for the stationary general eqm on p_grid (a grid on the GEPriceParams)

N_p=prod(n_p);
l_p=length(n_p);
p_grid=heteroagentoptions.pgrid;

%% 

if vfoptions.parallel==2
    GeneralEqmConditionsVec=ones(N_p,l_p,'gpuArray');
else
    GeneralEqmConditionsVec=ones(N_p,l_p);
end

for p_c=1:N_p
    if heteroagentoptions.verbose==1
        sprintf('Evaluating price vector %i of %i \n',p_c,N_p)
    end
        
    %Step 1: Solve the value fn iteration problem (given this price, indexed by p_c)
    %Calculate the price vector associated with p_c
    p_index=ind2sub_homemade(n_p,p_c);
    p=nan(l_p,1);
    for ii=1:l_p
        if ii==1
            p(ii)=p_grid(p_index(1));
        else
            p(ii)=p_grid(sum(n_p(1:ii-1))+p_index(ii));
        end
        Parameters.(GEPriceParamNames{ii})=p(ii);
    end
    
    if heteroagentoptions.gridsinGE==1
        % Some of the shock grids depend on parameters that are determined in general eqm
        [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_gridvals_J,pi_z_J,N_j,Parameters,vfoptions,3);
        % Convert z and e to age-dependent joint-grids and transtion matrix
        % Note: Ignores which, just redoes both z and e
        simoptions.e_gridvals_J=vfoptions.e_gridvals_J; % if no e, this is just empty anyway
        simoptions.pi_e_J=vfoptions.pi_e_J;
    end
    
    [~, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

    %Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
    StationaryDistKron=StationaryDist_FHorz_Case1(jequaloneDist, AgeWeightParamNames, Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDistKron, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, simoptions);
    
    % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
    GeneralEqmConditionsVec(p_c,:)=real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));
end

multiGEweightsKron=ones(N_p,1)*heteroagentoptions.multiGEweights;
if simoptions.parallel==2 || simoptions.parallel==4
    multiGEweightsKron=gpuArray(multiGEweightsKron);
end

if heteroagentoptions.multiGEcriterion==0 % the measure of market clearance is to take the sum of absolute distance in each market 
    [~,p_eqm_indexKron]=min(sum(abs(multiGEweightsKron.*GeneralEqmConditionsVec),2));
elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(multiGEweightsKron.*(GeneralEqmConditionsVec.^2),2));                                                                                                         
end

%p_eqm_index=zeros(num_p,1);
p_eqm_index=ind2sub_homemade_gpu(n_p,p_eqm_indexKron);
if l_p>1
    GeneralEqmConditions=nan(N_p,1+l_p,'gpuArray');
    if heteroagentoptions.multiGEcriterion==0
        GeneralEqmConditions(:,1)=sum(abs(multiGEweightsKron.*GeneralEqmConditionsVec),2);
    elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
        GeneralEqmConditions(:,1)=sum(multiGEweightsKron.*(GeneralEqmConditionsVec.^2),2);
    end
    GeneralEqmConditions(:,2:end)=multiGEweightsKron.*GeneralEqmConditionsVec;
    GeneralEqmConditions=reshape(GeneralEqmConditions,[n_p,1+l_p]);
else
    GeneralEqmConditions=reshape(multiGEweightsKron.*GeneralEqmConditionsVec,[n_p,1]);
end

p_eqm_index=round(p_eqm_index); % Was having rounding errors of order of numerical accuracy.

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
