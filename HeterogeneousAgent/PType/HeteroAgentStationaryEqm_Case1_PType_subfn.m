function GeneralEqmConditions=HeteroAgentStationaryEqm_Case1_PType_subfn(GEprices, PTypeStructure, Parameters, GeneralEqmEqns, GEPriceParamNames, AggVarNames, nGEprices, heteroagentoptions)

%% Do any transformations of parameters before we say what they are
penalty=zeros(length(GEprices),1); % Used to apply penalty to objective function when parameters try to leave restricted ranges
nGEprices=length(GEprices);
for pp=1:nGEprices
    if heteroagentoptions.constrainpositive(pp)==1 % Forcing this parameter to be positive
        temp=GEprices(pp);
        penalty(pp)=abs(temp/50).*(temp<-51); % 1 if out of range [Note: 51, rather than 50, so penalty only hits once genuinely out of range]
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        GEprices(pp)=exp(GEprices(pp));
    elseif heteroagentoptions.constrain0to1(pp)==1
        temp=GEprices(pp);
        penalty(pp)=abs(temp/50).*((temp>51)+(temp<-51)); % 1 if out of range [Note: 51, rather than 50, so penalty only hits once genuinely out of range]
        % Constrain parameter to be 0 to 1 (be working with x=log(p/(1-p)), where p is parameter) then always take 1/(1+exp(-x)) before inputting to model
        GEprices(pp)=1/(1+exp(-GEprices(pp)));
        % Note: This does not include the endpoints of 0 and 1 as 1/(1+exp(-x)) maps from the Real line into the open interval (0,1)
        %       R is not compact, and [0,1] is compact, so cannot have a continuous bijection (one-to-one and onto) function from R into [0,1].
        %       So I settle for a function from R to (0,1) and then trim ends of R to give 0 and 1, like I do for constrainpositive I use +-50 as the cutoffs
        GEprices(pp)=GEprices(pp).*(temp>-50); % set values less than -50 to zero
        GEprices(pp)=GEprices(pp).*(1-(temp>50))+(temp>50); % set values greater than 50 to one
    end
    % Note: sometimes, need to do both of constrainAtoB and constrain0to1, so cannot use elseif
    if heteroagentoptions.constrainAtoB(pp)==1
        % Constrain parameter to be A to B
        GEprices(pp)=heteroagentoptions.constrainAtoBlimits(pp,1)+(heteroagentoptions.constrainAtoBlimits(pp,2)-heteroagentoptions.constrainAtoBlimits(pp,1))*GEprices(pp);
        % Note, this parameter will have first been converted to 0 to 1 already, so just need to further make it A to B
        % y=A+(B-A)*x, converts 0-to-1 x, into A-to-B y
    end
end
if sum(penalty)>0
    penalty=1/prod(1./penalty(penalty>0)); % Turn into a scalar penalty [I try to do opposite of geometric mean, and penalize more when one gets extreme]
else
    penalty=0;
end
% NOTE: penalty has not been used here

%%
for pp=1:nGEprices % Not sure this is needed, have it just in case they are used when calling 'GeneralEqmConditionsFn', but I am pretty sure they never would be.
    Parameters.(GEPriceParamNames{pp})=gather(GEprices(pp));
end

if heteroagentoptions.parallel==2
    AggVars=zeros(PTypeStructure.numFnsToEvaluate,1,'gpuArray'); % numFnsToEvaluate is independent of the ptype
else
    AggVars=zeros(PTypeStructure.numFnsToEvaluate,1); % numFnsToEvaluate is independent of the ptype
end

for ii=1:PTypeStructure.N_i
    
    iistr=PTypeStructure.iistr{ii};
    for pp=1:length(GEPriceParamNames)
        PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEprices(pp);
    end

    if heteroagentoptions.gridsinGE(ii)==1
        % Some of the shock grids depend on parameters that are determined in general eqm
        [PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).vfoptions]=ExogShockSetup(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
        % Convert z and e to age-dependent joint-grids and transtion matrix
        % Note: Ignores which, just redoes both z and e
        PTypeStructure.(iistr).simoptions.e_gridvals=PTypeStructure.(iistr).vfoptions.e_gridvals; % if no e, this is just empty anyway
        PTypeStructure.(iistr).simoptions.pi_e=PTypeStructure.(iistr).vfoptions.pi_e;
    end

    
    [~, Policy_ii]=ValueFnIter_Case1(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
    StationaryDist_ii=StationaryDist_Case1(Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).simoptions,PTypeStructure.(iistr).Parameters);
    % PTypeStructure.(iistr).simoptions.outputasstructure=0; % Want AggVars_ii as matrix to make it easier to add them across the PTypes (is set outside this script)
    AggVars_ii=EvalFnOnAgentDist_AggVars_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluate, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, [], PTypeStructure.(iistr).simoptions);
    
    for kk=1:PTypeStructure.numFnsToEvaluate
        jj=PTypeStructure.(iistr).WhichFnsForCurrentPType(kk);
        if jj>0
            AggVars(kk)=AggVars(kk)+PTypeStructure.(iistr).PTypeWeight*AggVars_ii(jj);
        end
    end

end

% Note: AggVars is a matrix

% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
if isstruct(GeneralEqmEqns)
    for ii=1:length(AggVarNames)
        Parameters.(AggVarNames{ii})=AggVars(ii);
    end
    GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns, Parameters));
end

% We might want to output GE conditions as a vector or structure
if heteroagentoptions.outputGEform==0 % scalar
    if heteroagentoptions.multiGEcriterion==0
        GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
    elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market
        GeneralEqmConditions=sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2));
    end
    if heteroagentoptions.outputgather==1
        GeneralEqmConditions=gather(GeneralEqmConditions);
    end
elseif heteroagentoptions.outputGEform==1 % vector
    GeneralEqmConditions=GeneralEqmConditionsVec;
    if heteroagentoptions.outputgather==1
        GeneralEqmConditions=gather(GeneralEqmConditions);
    end
elseif heteroagentoptions.outputGEform==2 % structure
    clear GeneralEqmConditions
    GeneralEqmEqnsNames=fieldnames(GeneralEqmEqns);
    for ii=1:length(GeneralEqmEqnsNames)
        GeneralEqmConditions.(GeneralEqmEqnsNames{ii})=GeneralEqmConditionsVec(ii);
    end
end

if heteroagentoptions.verbose==1
    fprintf(' \n')
    fprintf('Current GE prices: \n')
    for ii=1:nGEprices
        fprintf('	%s: %8.4f \n',GEPriceParamNames{ii},GEprices(ii))
    end
    fprintf('Current aggregate variables: \n')
    for ii=1:length(AggVarNames)
        fprintf('	%s: %8.4f \n',AggVarNames{ii},Parameters.(AggVarNames{ii})) % Note, this is done differently here because AggVars itself has been set as a matrix
    end
    fprintf('Current GeneralEqmEqns: \n')
    GeneralEqmEqnsNames=fieldnames(GeneralEqmEqns);
    for ii=1:length(GeneralEqmEqnsNames)
        fprintf('	%s: %8.4f \n',GeneralEqmEqnsNames{ii},GeneralEqmConditionsVec(ii))
    end
end

end