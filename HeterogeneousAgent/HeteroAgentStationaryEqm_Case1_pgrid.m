function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_pgrid(n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluateFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
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
        p_c
    end
    
%     V0Kron(~isfinite(V0Kron))=0; %Since we loop through with V0Kron from previous p_c this is necessary to avoid contamination by -Inf's
    
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
    
    [~,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn,Parameters, DiscountFactorParamNames,ReturnFnParamNames,vfoptions);

    %Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
    StationaryDistKron=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z,simoptions);
    
    AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDistKron, Policy, FnsToEvaluateFn, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions.parallel);
    
    % The following line is often a useful double-check if something is going wrong.
%    AggVars
    
    % use of real() is a hack that could disguise errors, but I couldn't
    % find why matlab was treating output as complex
    GeneralEqmConditionsKron(p_c,:)=real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames, simoptions.parallel));
end

if heteroagentoptions.multiGEcriterion==0 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(abs(GeneralEqmConditionsKron),2));
elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(GeneralEqmConditionsKron.^2,2));                                                                                                         
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
        GeneralEqmConditions(:,1)=sum(abs(GeneralEqmConditionsKron),2);
    elseif heteroagentoptions.multiGEcriterion==1 %the measure of general eqm is to take the sum of squares of each of the general eqm conditions holding 
        GeneralEqmConditions(:,1)=sum(GeneralEqmConditionsKron.^2,2);
    end
    GeneralEqmConditions(:,2:end)=GeneralEqmConditionsKron;
    GeneralEqmConditions=reshape(GeneralEqmConditions,[n_p,1+l_p]);
else
    GeneralEqmConditions=reshape(GeneralEqmConditionsKron,[n_p,1]);
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
