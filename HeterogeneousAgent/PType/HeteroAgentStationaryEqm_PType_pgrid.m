function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_PType_pgrid(n_p, PTypeStructure, Parameters, GeneralEqmEqns, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions)

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);
N_p=prod(n_p);

l_p=length(n_p);

p_grid=heteroagentoptions.p_grid;

%%

if heteroagentoptions.parallel==2
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
    for pp=1:l_p
        if pp==1
            p(pp)=p_grid(p_index(1));
        else
            p(pp)=p_grid(sum(n_p(1:pp-1))+p_index(pp));
        end
        Parameters.(GEPriceParamNames{pp})=gather(p(pp));
    end
    
    
    for ii=1:PTypeStructure.N_i
    
        iistr=PTypeStructure.iistr{ii};
        for pp=1:length(GEPriceParamNames)
            PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=p(pp);
        end
        
        if PTypeStructure.(iistr).finitehorz==0  % Infinite horizon
            % Infinite Horizon requires an initial guess of value function. For
            % the present I simply don't let this feature be used when using
            % permanent types. WOULD BE GOOD TO CHANGE THIS IN FUTURE SOMEHOW.
            if PTypeStructure.(iistr).Case1orCase2==1
                [~, Policy_ii]=ValueFnIter_Case1(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
                StationaryDist_ii=StationaryDist_Case1(Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).simoptions);
                StatsFromDist_AggVars_ii=SSvalues_AggVars_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluateFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).simoptions.parallel);
            elseif PTypeStructure.(iistr).Case1orCase2==2
                [~, Policy_ii]=ValueFnIter_Case2(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).Phi_aprime, PTypeStructure.(iistr).Case2_Type, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).PhiaprimeParamNames, PTypeStructure.(iistr).vfoptions);
                StationaryDist_ii=StationaryDist_Case2(Policy_ii,PTypeStructure.(iistr).Phi_aprime,PTypeStructure.(iistr).Case2_Type,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).simoptions);
                StatsFromDist_AggVars_ii=SSvalues_AggVars_Case2(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluateFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).simoptions.parallel);
            end
        elseif PTypeStructure.(iistr).finitehorz==1 % Finite horizon
            % Check for some relevant vfoptions that may depend on permanent type
            % dynasty, agedependentgrids, lowmemory, (parallel??)
            if PTypeStructure.(iistr).Case1orCase2==1
                [~, Policy_ii]=ValueFnIter_Case1_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
                StationaryDist_ii=StationaryDist_FHorz_Case1(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).simoptions);
                StatsFromDist_AggVars_ii=SSvalues_AggVars_FHorz_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluateFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).simoptions.parallel);
            elseif PTypeStructure.(iistr).Case1orCase2==2
                [~, Policy_ii]=ValueFnIter_Case2_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).Phi_aprime, PTypeStructure.(iistr).Case2_Type, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).PhiaprimeParamNames, PTypeStructure.(iistr).vfoptions);
                StationaryDist_ii=StationaryDist_FHorz_Case2(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Phi_aprime,PTypeStructure.(iistr).Case2_Type,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).PhiaprimeParamNames,PTypeStructure.(iistr).simoptions);
                StatsFromDist_AggVars_ii=SSvalues_AggVars_FHorz_Case2(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluateFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).simoptions, PTypeStructure.(iistr).pi_z); % If using Age Dependent Variables then 'pi_z' is actually AgeDependentGridParamNames. If not then it will anyway be ignored.
            end
        end
        
        StatsFromDist_AggVars=zeros(PTypeStructure.(iistr).numFnsToEvaluate,1,'gpuArray');
        for kk=1:PTypeStructure.(iistr).numFnsToEvaluate
            jj=PTypeStructure.(iistr).WhichFnsForCurrentPType(kk);
            if jj>0
                StatsFromDist_AggVars(kk,:)=StatsFromDist_AggVars(kk,:)+PTypeStructure.(iistr).PTypeWeight*StatsFromDist_AggVars_ii(jj,:);
            end
        end
    
    end
    
    GeneralEqmConditionsKron(p_c,:)=real(GeneralEqmConditionsFn(StatsFromDist_AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));

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