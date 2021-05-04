function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case2_FHorz_pgrid(jequaloneDist,AgeWeightParamNames,n_d, n_a, n_z, N_j, n_p, pi_z, d_grid, a_grid, z_grid, Phi_aprimeFn, Case2_Type, ReturnFn, FnsToEvaluateFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);
N_p=prod(n_p);

l_p=length(n_p);

p_grid=heteroagentoptions.p_grid;

%%

if vfoptions.parallel==2
    GeneralEqmConditionsKron=ones(N_p,l_p,'gpuArray');
else
    GeneralEqmConditionsKron=ones(N_p,l_p);
end
%V0Kron=reshape(V0,[N_a,N_s]);

for p_c=1:N_p
    if heteroagentoptions.verbose==1
        sprintf('Evaluating price vector %i of %i',p_c,N_p)
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
        Parameters.(GEPriceParamNames{ii})=gather(p(ii));
    end
    
    [~, Policy]=ValueFnIter_Case2_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, Phi_aprimeFn, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
    
    %Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
    StationaryDist=StationaryDist_FHorz_Case2(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames,simoptions);
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case2(StationaryDist, Policy, FnsToEvaluateFn, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z,N_j, d_grid, a_grid, z_grid, simoptions, pi_z); % If using Age Dependent Variables then 'pi_z' is actually AgeDependentGridParamNames. If not then it will anyway be ignored.

    % The following line is often a useful double-check if something is going wrong.
%    AggVars
    
    % use of real() is a hack that could disguise errors, but I couldn't
    % find why matlab was treating output as complex
    GeneralEqmConditionsKron(p_c,:)=real(GeneralEqmConditions_Case2(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));
end

multiGEweightsKron=ones(N_p,1)*heteroagentoptions.multiGEweights;
if simoptions.parallel==2 || simoptions.parallel==4
    multiGEweightsKron=gpuArray(multiGEweightsKron);
end

if heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(multiGEweightsKron.*(GeneralEqmConditionsKron.^2),2));                                                                                                         
end

p_eqm_index=ind2sub_homemade_gpu(n_p,p_eqm_indexKron);
if l_p>1
    GeneralEqmConditions=nan(N_p,1+l_p,'gpuArray');
    if heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
        GeneralEqmConditions(:,1)=sum(multiGEweightsKron.*(GeneralEqmConditionsKron.^2),2);
    end
    GeneralEqmConditions(:,2:end)=multiGEweightsKron.*GeneralEqmConditionsKron;
    GeneralEqmConditions=reshape(GeneralEqmConditions,[n_p,1+l_p]);
else
    GeneralEqmConditions=reshape(multiGEweightsKron.*GeneralEqmConditionsKron,[n_p,1]);
end

%Calculate the price associated with p_eqm_index
p_eqm=zeros(l_p,1);
for i=1:l_p
    if i==1
        p_eqm(i)=p_grid(p_eqm_index(1));
    else
        p_eqm(i)=p_grid(sum(n_p(1:i-1))+p_eqm_index(i));
    end
end

% Move results from gpu to cpu before returning them
p_eqm=gather(p_eqm);
p_eqm_index=gather(p_eqm_index);
GeneralEqmConditions=gather(GeneralEqmConditions);

end