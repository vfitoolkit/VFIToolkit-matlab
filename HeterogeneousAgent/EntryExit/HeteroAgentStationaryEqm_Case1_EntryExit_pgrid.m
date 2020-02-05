function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_EntryExit_pgrid(V0Kron, n_d, n_a, n_z, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluateFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, EntryExitParamNames, heteroagentoptions, simoptions, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_p=prod(n_p);

l_p=length(n_p);

p_grid=heteroagentoptions.pgrid;

if simoptions.parallel==2 || simoptions.parallel==4
    GeneralEqmConditionsKron=ones(N_p,length(GeneralEqmEqns),'gpuArray');
else
    GeneralEqmConditionsKron=ones(N_p,length(GeneralEqmEqns));
end

%% Figure out which general eqm conditions are normal
standardgeneqmcondnsused=0;
specialgeneqmcondnsused=0;
entrycondnexists=0; condlentrycondnexists=0;
if ~isfield(heteroagentoptions,'specialgeneqmcondn')
    standardgeneqmcondnindex=1:1:length(GeneralEqmEqns);
else
    standardgeneqmcondnindex=zeros(1,length(GeneralEqmEqns));
    jj=1;
    GeneralEqmEqnParamNames_Full=GeneralEqmEqnParamNames;
    clear GeneralEqmEqnParamNames
    for ii=1:length(GeneralEqmEqns)
        if isnumeric(heteroagentoptions.specialgeneqmcondn{ii}) % numeric means equal to zero and is a standard GEqm
            standardgeneqmcondnsused=1;
            standardgeneqmcondnindex(jj)=ii;
            GeneralEqmEqnParamNames(jj).Names=GeneralEqmEqnParamNames_Full(ii).Names;
            jj=jj+1;
        elseif strcmp(heteroagentoptions.specialgeneqmcondn{ii},'entry')
            specialgeneqmcondnsused=1;
            entrycondnexists=1;
            entrygeneqmcondnindex=ii;
            EntryCondnEqn=GeneralEqmEqns(ii);
            EntryCondnEqnParamNames(1).Names=GeneralEqmEqnParamNames_Full(ii).Names;
        elseif strcmp(heteroagentoptions.specialgeneqmcondn{ii},'condlentry')
            specialgeneqmcondnsused=1;
            condlentrycondnexists=1;
            condlentrygeneqmcondnindex=ii;
            CondlEntryCondnEqn=GeneralEqmEqns(ii);
            CondlEntryCondnEqnParamNames(1).Names=GeneralEqmEqnParamNames_Full(ii).Names;
        end
    end
    standardgeneqmcondnindex=standardgeneqmcondnindex(standardgeneqmcondnindex>0); % get rid of zeros at the end
    GeneralEqmEqns=GeneralEqmEqns(standardgeneqmcondnindex);
end


%% 

for p_c=1:N_p
    if heteroagentoptions.verbose==1
        p_c
    end
    
    V0Kron(~isfinite(V0Kron))=0; %Since we loop through with V0Kron from previous p_c this is necessary to avoid contamination by -Inf's
    
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
    
    %     ReturnFnParams(IndexesForPricesInReturnFnParams)=p;
%     [V,Policy]=ValueFnIter_Case1(V0Kron, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn,Parameters, DiscountFactorParamNames,ReturnFnParamNames,vfoptions);
    if simoptions.endogenousexit==1
        [V,Policy,ExitPolicy]=ValueFnIter_Case1(V0Kron, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % With entry-exit will need to keep the value function to be able to evaluate entry condition.
        Parameters.(EntryExitParamNames.CondlProbOfSurvival{1})=1-ExitPolicy;
    else
        [V,Policy]=ValueFnIter_Case1(V0Kron, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % With entry-exit will need to keep the value function to be able to evaluate entry condition.
    end
    
    %Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
%     StationaryDistKron=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z,simoptions);
    StationaryDistKron=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z, simoptions,Parameters,EntryExitParamNames);
%     Parameters.(EntryExitParamNames.MassOfExistingAgents{1})=MassOfExistingAgents;

    AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDistKron, Policy, FnsToEvaluateFn, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions.parallel,simoptions,EntryExitParamNames);
    
    % The following line is often a useful double-check if something is going wrong.
%    SSvalues_AggVars
    
    if standardgeneqmcondnsused==1
        % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
        GeneralEqmConditionsKron(p_c,standardgeneqmcondnindex)=real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames, simoptions.parallel));
    end
    % Now fill in the 'non-standard' cases
    if specialgeneqmcondnsused==1
        if condlentrycondnexists==1
            % Evaluate the conditional equilibrium condition on the (potential entrants) grid,
            % and where it is >=0 use this to set new values for the
            % EntryExitParamNames.CondlEntryDecisions parameter.
%             PotentialEntrantDist=Parameters.(EntryExitParamNames.DistOfNewAgents{1});
            CondlEntryCondnEqnParamsVec=CreateVectorFromParams(Parameters, CondlEntryCondnEqnParamNames(1).Names);
            CondlEntryCondnEqnParamsCell=cell(length(CondlEntryCondnEqnParamsVec),1);
            for jj=1:length(CondlEntryCondnEqnParamsVec)
                CondlEntryCondnEqnParamsCell(jj,1)={CondlEntryCondnEqnParamsVec(jj)};
            end
            Parameters.(EntryExitParamNames.CondlEntryDecisions{1})=(CondlEntryCondnEqn{1}(V,p,CondlEntryCondnEqnParamsCell{:}) >=0);
            GeneralEqmConditionsKron(p_c,condlentrygeneqmcondnindex)=0; % Because the EntryExitParamNames.CondlEntryDecisions is set to hold exactly we can consider this as contributing 0
            if entrycondnexists==1
                % Calculate the expected (based on entrants distn) value fn (note, DistOfNewAgents is the pdf, so this is already 'normalized' EValueFn.
                EValueFn=sum(reshape(V,[numel(V),1]).*reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[numel(V),1]).*reshape(Parameters.(EntryExitParamNames.CondlEntryDecisions{1}),[numel(V),1]));
                % @(EValueFn,ce)
                % And use entrants distribution, not the stationary distn
                GeneralEqmConditionsKron(p_c,entrygeneqmcondnindex)=real(GeneralEqmConditions_Case1(EValueFn,p, EntryCondnEqn, Parameters,EntryCondnEqnParamNames, simoptions.parallel));
            end
        else
            if entrycondnexists==1
                % Calculate the expected (based on entrants distn) value fn (note, DistOfNewAgents is the pdf, so this is already 'normalized' EValueFn.
                EValueFn=sum(reshape(V,[numel(V),1]).*reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[numel(V),1]));
                % @(EValueFn,ce)
                % And use entrants distribution, not the stationary distn
                GeneralEqmConditionsKron(p_c,entrygeneqmcondnindex)=real(GeneralEqmConditions_Case1(EValueFn,p, EntryCondnEqn, Parameters,EntryCondnEqnParamNames, simoptions.parallel));
            end
        end
    end
end

if heteroagentoptions.multiGEcriterion==0 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(abs(GeneralEqmConditionsKron),2));
elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(GeneralEqmConditionsKron.^2,2));                                                                                                         
end

%p_eqm_index=zeros(num_p,1);
p_eqm_index=ind2sub_homemade_gpu(n_p,p_eqm_indexKron);
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
p_eqm_vec=zeros(l_p,1);
for ii=1:l_p
    if ii==1
        p_eqm_vec(ii)=p_grid(p_eqm_index(1));
    else
        p_eqm_vec(ii)=p_grid(sum(n_p(1:ii-1))+p_eqm_index(ii));
    end
end

% Move results from gpu to cpu before returning them
p_eqm_vec=gather(p_eqm_vec);
p_eqm=struct();
for ii=1:length(GEPriceParamNames)
    p_eqm.(GEPriceParamNames{ii})=p_eqm_vec(ii);
end
if specialgeneqmcondnsused==1
    if condlentrycondnexists==1
        p_eqm.(EntryExitParamNames.CondlEntryDecisions{1})=Parameters.(EntryExitParamNames.CondlEntryDecisions{1});
    end
end
p_eqm_index=gather(p_eqm_index);
GeneralEqmConditions=gather(GeneralEqmConditions);

end
