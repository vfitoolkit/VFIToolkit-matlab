function GeneralEqmConditions=HeteroAgentStationaryEqm_Case1_FHorz_subfn(GEprices, jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, l_p, pi_z_J, d_grid, a_grid, z_gridvals_J, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions)

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
for ii=1:l_p
    Parameters.(GEPriceParamNames{ii})=GEprices(ii);
end

if heteroagentoptions.gridsinGE==1
    % Some of the shock grids depend on parameters that are determined in general eqm
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_gridvals_J,pi_z_J,N_j,Parameters,vfoptions,3);
    % Convert z and e to age-dependent joint-grids and transtion matrix
    % Note: Ignores which, just redoes both z and e
    simoptions.e_gridvals_J=vfoptions.e_gridvals_J; % if no e, this is just empty anyway
    simoptions.pi_e_J=vfoptions.pi_e_J;
end

%%
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

%Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z,N_j, d_grid, a_grid, z_gridvals_J,simoptions);

if heteroagentoptions.useCustomModelStats==1
    CustomStats=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,pi_z_J,heteroagentoptions,vfoptions,simoptions);
    % Note: anything else you want, just 'hide' it in heteroagentoptions
    customstatnames=fieldnames(CustomStats);
    for pp=1:length(customstatnames)
        Parameters.(customstatnames{pp})=CustomStats.(customstatnames{pp});
    end
end

% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
if isstruct(GeneralEqmEqns)
    AggVarNames=fieldnames(AggVars); % Using GeneralEqmEqns as a struct presupposes using FnsToEvaluate (and hence AggVars) as a stuct
    for ii=1:length(AggVarNames)
        Parameters.(AggVarNames{ii})=AggVars.(AggVarNames{ii}).Mean;
    end
    GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns, Parameters));
else
    GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1(AggVars,GEprices, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));
end

% We might want to output GE conditions as a vector or structure
if heteroagentoptions.outputGEform==0 % scalar
    if heteroagentoptions.multiGEcriterion==0
        GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
    elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market
        GeneralEqmConditions=sqrt(sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2)));
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
    for ii=1:l_p
        fprintf('	%s: %8.4f \n',GEPriceParamNames{ii},GEprices(ii))
    end
    fprintf('Current aggregate variables: \n')
    if ~isstruct(AggVars)
        AggVars
    else
        for ii=1:length(AggVarNames)
            fprintf('	%s: %8.4f \n',AggVarNames{ii},AggVars.(AggVarNames{ii}).Mean)
        end
        if heteroagentoptions.useCustomModelStats==1
            for ii=1:length(customstatnames)
                fprintf('	%s: %8.4f \n',customstatnames{ii},CustomStats.(customstatnames{ii}))
            end
        end
    end
    fprintf('Current GeneralEqmEqns: \n')
    if ~isstruct(GeneralEqmEqns)
        GeneralEqmConditionsVec
    else
        GeneralEqmEqnsNames=fieldnames(GeneralEqmEqns);
        for ii=1:length(GeneralEqmEqnsNames)
            fprintf('	%s: %8.4f \n',GeneralEqmEqnsNames{ii},GeneralEqmConditionsVec(ii))
        end
    end
end


end