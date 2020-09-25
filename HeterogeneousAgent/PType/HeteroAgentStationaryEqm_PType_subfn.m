function GeneralEqmCondition=HeteroAgentStationaryEqm_PType_subfn(p, PTypeStructure, Parameters, GeneralEqmEqns, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions)

%%
for pp=1:length(GEPriceParamNames) % Not sure this is needed, have it just in case they are used when calling 'GeneralEqmConditionsFn', but I am pretty sure they never would be.
    Parameters.(GEPriceParamNames{pp})=gather(p(pp));
end

StatsFromDist_AggVars=zeros(PTypeStructure.ptype001.numFnsToEvaluate,1,'gpuArray'); % numFnsToEvaluate should be independent of the ptype so just take it from the first.

for ii=1:PTypeStructure.N_i
    
    iistr=PTypeStructure.iistr{ii};
    for pp=1:length(GEPriceParamNames)
        PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=p(pp);
    end
    
    if PTypeStructure.(iistr).finitehorz==0  % Infinite horizon
        % Infinite Horizon requires an initial guess of value function. For
        % the present I simply don't let this feature be used when using
        % permanent types. WOULD BE GOOD TO CHANGE THIS IN FUTURE SOMEHOW.
        V_ii=zeros(prod(PTypeStructure.(iistr).n_a),prod(PTypeStructure.(iistr).n_z)); % The initial guess (note that its value is 'irrelevant' in the sense that global uniform convergence is anyway known to occour for VFI).
        if PTypeStructure.(iistr).Case1orCase2==1
            if isfield(PTypeStructure.(iistr),'vfoptions')
                [~, Policy_ii]=ValueFnIter_Case1(V_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
            else
                [~, Policy_ii]=ValueFnIter_Case1(V_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames);
            end
            if isfield(PTypeStructure.(iistr),'simoptions')
                StationaryDist_ii=StationaryDist_Case1(Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).simoptions);
            else
                StationaryDist_ii=StationaryDist_Case1(Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z);
            end
            StatsFromDist_AggVars_ii=SSvalues_AggVars_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluateFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).simoptions.parallel);
        elseif PTypeStructure.(iistr).Case1orCase2==2
            if isfield(PTypeStructure.(iistr),'vfoptions')
                [~, Policy_ii]=ValueFnIter_Case2(V_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).Phi_aprime, PTypeStructure.(iistr).Case2_Type, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).PhiaprimeParamNames, PTypeStructure.(iistr).vfoptions);
            else
                [~, Policy_ii]=ValueFnIter_Case2(V_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).Phi_aprime, PTypeStructure.(iistr).Case2_Type, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).PhiaprimeParamNames);
            end
            if isfield(PTypeStructure.(iistr),'simoptions')
                StationaryDist_ii=StationaryDist_Case2(Policy_ii,PTypeStructure.(iistr).Phi_aprime,PTypeStructure.(iistr).Case2_Type,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).simoptions);
            else
                StationaryDist_ii=StationaryDist_Case2(Policy_ii,PTypeStructure.(iistr).Phi_aprime,PTypeStructure.(iistr).Case2_Type,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z);
            end
            StatsFromDist_AggVars_ii=SSvalues_AggVars_Case2(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluateFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).simoptions.parallel);
        end
    elseif PTypeStructure.(iistr).finitehorz==1 % Finite horizon
        % Check for some relevant vfoptions that may depend on permanent type
        % dynasty, agedependentgrids, lowmemory, (parallel??)
        if PTypeStructure.(iistr).Case1orCase2==1
            if isfield(PTypeStructure.(iistr),'vfoptions')
                [~, Policy_ii]=ValueFnIter_Case1_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
            else
                [~, Policy_ii]=ValueFnIter_Case1_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames);
            end
            if isfield(PTypeStructure.(iistr),'simoptions')
                StationaryDist_ii=StationaryDist_FHorz_Case1(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).simoptions);
            else
                StationaryDist_ii=StationaryDist_FHorz_Case1(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters);
            end
            StatsFromDist_AggVars_ii=SSvalues_AggVars_FHorz_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluateFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).simoptions.parallel);
        elseif PTypeStructure.(iistr).Case1orCase2==2
            if isfield(PTypeStructure.(iistr),'vfoptions')
                [~, Policy_ii]=ValueFnIter_Case2_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).Phi_aprime, PTypeStructure.(iistr).Case2_Type, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).PhiaprimeParamNames, PTypeStructure.(iistr).vfoptions);
            else
                [~, Policy_ii]=ValueFnIter_Case2_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).Phi_aprime, PTypeStructure.(iistr).Case2_Type, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).PhiaprimeParamNames);
            end
            if isfield(PTypeStructure.(iistr),'simoptions')
                StationaryDist_ii=StationaryDist_FHorz_Case2(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Phi_aprime,PTypeStructure.(iistr).Case2_Type,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).PhiaprimeParamNames,PTypeStructure.(iistr).simoptions);
                StatsFromDist_AggVars_ii=SSvalues_AggVars_FHorz_Case2(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluateFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).simoptions, PTypeStructure.(iistr).pi_z); % If using Age Dependent Variables then 'pi_z' is actually AgeDependentGridParamNames. If not then it will anyway be ignored.
            else
                StationaryDist_ii=StationaryDist_FHorz_Case2(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Phi_aprime,PTypeStructure.(iistr).Case2_Type,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).PhiaprimeParamNames);
                StatsFromDist_AggVars_ii=SSvalues_AggVars_FHorz_Case2(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluateFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid);
            end           
        end
    end
    
    
%     [~, Policy]=ValueFnIter_Case2_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, Phi_aprimeFn, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);

%     %Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
%     StationaryDistKron=StationaryDist_FHorz_Case2(jequaloneDist,AgeWeights,Policy,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames,simoptions);
%     SSvalues_AggVars=SSvalues_AggVars_FHorz_Case2(StationaryDistKron, Policy, FnsToEvaluateFn, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z,N_j, d_grid, a_grid, z_grid, simoptions, pi_z); % If using Age Dependent Variables then 'pi_z' is actually AgeDependentGridParamNames. If not then it will anyway be ignored.

    
%     StatsFromDist_AggVars=zeros(PTypeStructure.(iistr).numFnsToEvaluate,1,'gpuArray');     % MOVED OUTSIDE THE ii loop
    for kk=1:PTypeStructure.(iistr).numFnsToEvaluate
        jj=PTypeStructure.(iistr).WhichFnsForCurrentPType(kk);
        if jj>0
            StatsFromDist_AggVars(kk,:)=StatsFromDist_AggVars(kk,:)+PTypeStructure.(iistr).PTypeWeight*StatsFromDist_AggVars_ii(jj,:);
        end
    end

end

% The following line is often a useful double-check if something is going wrong.
%    SSvalues_AggVars

% use of real() is a hack that could disguise errors, but I couldn't
% find why matlab was treating output as complex
GeneralEqmConditionsVec=real(GeneralEqmConditionsFn(StatsFromDist_AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));



if heteroagentoptions.multiGEcriterion==0 
    GeneralEqmCondition=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    GeneralEqmCondition=sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2));                                                                                                         
end

GeneralEqmCondition=gather(GeneralEqmCondition);

if heteroagentoptions.verbose==1
    fprintf('Current GE prices and GeneralEqmConditionsVec. \n')
    p
    GeneralEqmConditionsVec
end