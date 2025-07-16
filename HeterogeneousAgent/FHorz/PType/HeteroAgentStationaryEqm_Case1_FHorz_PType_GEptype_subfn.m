function GeneralEqmConditions=HeteroAgentStationaryEqm_Case1_FHorz_PType_GEptype_subfn(GEprices, PTypeStructure, Parameters, GeneralEqmEqns,  GeneralEqmEqnsCell, GeneralEqmEqnParamNames,GEPriceParamNames, AggVarNames, nGEprices, GEpriceindexes, GEprice_ptype, heteroagentoptions)

% nGEprices=length(GEPriceParamNames);

%% Do any transformations of parameters before we say what they are
penalty=zeros(length(GEprices),1); % Used to apply penalty to objective function when parameters try to leave restricted ranges
for pp=1:nGEprices
    if heteroagentoptions.constrainpositive(pp)==1 % Forcing this parameter to be positive
        temp=GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
        penalty(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=abs(temp/50).*(temp<-51); % 1 if out of range [Note: 51, rather than 50, so penalty only hits once genuinely out of range]
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=exp(GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2)));
    elseif heteroagentoptions.constrain0to1(pp)==1
        temp=GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
        penalty(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=abs(temp/50).*((temp>51)+(temp<-51)); % 1 if out of range [Note: 51, rather than 50, so penalty only hits once genuinely out of range]
        % Constrain parameter to be 0 to 1 (be working with x=log(p/(1-p)), where p is parameter) then always take 1/(1+exp(-x)) before inputting to model
        GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=1/(1+exp(-GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2))));
        % Note: This does not include the endpoints of 0 and 1 as 1/(1+exp(-x)) maps from the Real line into the open interval (0,1)
        %       R is not compact, and [0,1] is compact, so cannot have a continuous bijection (one-to-one and onto) function from R into [0,1].
        %       So I settle for a function from R to (0,1) and then trim ends of R to give 0 and 1, like I do for constrainpositive I use +-50 as the cutoffs
        GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2)).*(temp>-50); % set values less than -50 to zero
        GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2)).*(1-(temp>50))+(temp>50); % set values greater than 50 to one
    end
    % Note: sometimes, need to do both of constrainAtoB and constrain0to1, so cannot use elseif
    if heteroagentoptions.constrainAtoB(pp)==1
        % Constrain parameter to be A to B
        GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=heteroagentoptions.constrainAtoBlimits(pp,1)+(heteroagentoptions.constrainAtoBlimits(pp,2)-heteroagentoptions.constrainAtoBlimits(pp,1))*GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
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
    Parameters.(GEPriceParamNames{pp})=GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
end

AggVars_ConditionalOnPType=zeros(PTypeStructure.numFnsToEvaluate,PTypeStructure.N_i,'gpuArray'); % Create AggVars conditional on ptype.

for ii=1:PTypeStructure.N_i
    
    iistr=PTypeStructure.iistr{ii};
    for pp=1:length(GEPriceParamNames)
        if GEprice_ptype(pp)==0
            PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEprices(GEpriceindexes(pp,1));
        else
            PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEprices(GEpriceindexes(pp,1)+ii-1);
            % I also create an '_name' version that can be used for GE eqns that are cross-ptype
        end
    end

    if heteroagentoptions.gridsinGE(ii)==1
        % Some of the shock grids depend on parameters that are determined in general eqm
        [PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).vfoptions]=ExogShockSetup_FHorz(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_gridvals_J,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
        % Convert z and e to age-dependent joint-grids and transtion matrix
        % Note: Ignores which, just redoes both z and e
        PTypeStructure.(iistr).simoptions.e_gridvals_J=PTypeStructure.(iistr).vfoptions.e_gridvals_J; % if no e, this is just empty anyway
        PTypeStructure.(iistr).simoptions.pi_e_J=PTypeStructure.(iistr).vfoptions.pi_e_J;
    end
    
    if isfinite(PTypeStructure.(iistr).N_j)
        [~, Policy_ii]=ValueFnIter_Case1_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
        StationaryDist_ii=StationaryDist_FHorz_Case1(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).simoptions);
        % PTypeStructure.(iistr).simoptions.outputasstructure=0; % Want AggVars_ii as matrix to make it easier to add them across the PTypes (is set outside this script)
        AggVars_ii=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluate, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).simoptions);
    else  % PType actually allows for infinite horizon as well
        [~, Policy_ii]=ValueFnIter_Case1(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
        StationaryDist_ii=StationaryDist_Case1(Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).simoptions,PTypeStructure.(iistr).Parameters);
        % PTypeStructure.(iistr).simoptions.outputasstructure=0; % Want AggVars_ii as matrix to make it easier to add them across the PTypes (is set outside this script)
        AggVars_ii=EvalFnOnAgentDist_AggVars_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluate, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).simoptions);     
    end

    AggVars_ConditionalOnPType(PTypeStructure.(iistr).FnsAndPTypeIndicator_ii,ii)=AggVars_ii;
end
AggVars=sum(AggVars_ConditionalOnPType.*PTypeStructure.ptweights,2);
% Note: AggVars is a vector



%% Put GE parameters  and AggVars in structure, so they can be used for intermediateEqns and GeneralEqmEqns
% already did the basic GE params
% for pp=1:nGEprices
%     Parameters.(GEPriceParamNames{pp})=GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
% end
for aa=1:length(AggVarNames)
    Parameters.(AggVarNames{aa})=AggVars(aa);
end

% Do the general eqm parameters that depend on ptype
for pp=1:length(GEPriceParamNames)
    if GEprice_ptype(pp)==1
        for ii=1:PTypeStructure.N_i
            Parameters.([GEPriceParamNames{pp},'_',PTypeStructure.Names_i{ii}])=GEprices(GEpriceindexes(pp,1)+ii-1);
        end
    end
end
% And do all the AggVars as well
for aa=1:length(AggVarNames)
    for ii=1:PTypeStructure.N_i
        Parameters.([AggVarNames{aa},'_',PTypeStructure.Names_i{ii}])=AggVars_ConditionalOnPType(aa,ii); % Note: this will create AggVar values of zero, even where they 'dont exist', but I don't think the user will get that wrong
    end
end


%% Intermediate Eqns
if isfield(heteroagentoptions,'intermediateEqns')
    % Note: intermediateEqns just take in things from the Parameters structure, as do GeneralEqmEqns (AggVars get put into structure), hence just use the GeneralEqmConditions_Case1_v3g().
    
    intEqnnames=fieldnames(heteroagentoptions.intermediateEqns);
    intermediateEqnsVec=zeros(1,sum(heteroagentoptions.intermediateEqnsptype==0)+PTypeStructure.N_i*sum(heteroagentoptions.intermediateEqnsptype==1));

    % Do the intermediateEqns, in order
    gg_c=0;
    for gg=1:length(intEqnnames)
        if heteroagentoptions.intermediateEqnsptype(gg)==0 % standard intermediateEqn
            gg_c=gg_c+1;
            intermediateEqnsVec(gg_c)=real(GeneralEqmConditions_Case1_v3g(heteroagentoptions.intermediateEqnsCell{gg}, heteroagentoptions.intermediateEqnParamNames(gg_c).Names, Parameters));
            Parameters.(intEqnnames{gg})=intermediateEqnsVec(gg_c);

            % if the intermediateEqn is using '_name', then put it into Params as a structure with name
            intEqnnames_gg=intEqnnames{gg};
            if contains(intEqnnames_gg,'_') % potential uses '_name'
                for ii=1:PTypeStructure.N_i
                    lname=length(PTypeStructure.Names_i{ii});
                    if length(intEqnnames_gg)>lname+1 % only check if intEqnnames_gg is long enough to be possible
                        if strcmp(intEqnnames_gg(end-lname:end),['_',PTypeStructure.Names_i{ii}])
                            Parameters.(intEqnnames_gg(1:end-lname-1)).(PTypeStructure.Names_i{ii})=Parameters.(intEqnnames{gg});
                            % E.g., creates Parameters.r.ptype001 from Parameters.r_ptype001
                        end
                    end
                end
            end
        elseif heteroagentoptions.intermediateEqnsptype(gg)==1 % Do this intermediateEqn condition conditional on ptype
            for ii=1:PTypeStructure.N_i % This General eqm condition has to hold conditional on each ptype
                gg_c=gg_c+1;
                intermediateEqnsVec(gg_c)=real(GeneralEqmConditions_Case1_v3g_ptype(heteroagentoptions.intermediateEqnsCell{gg}, heteroagentoptions.intermediateEqnParamNames(gg_c).Names, Parameters));
                Parameters.(intEqnnames{gg}).(PTypeStructure.Names_i{ii})=intermediateEqnsVec(gg_c);
                % also, just in case they need to be used again, add the _name version
                Parameters.([intEqnnames{gg},'_',PTypeStructure.Names_i{ii}])=intermediateEqnsVec(gg_c);
            end
        end
    end
end


%% Evaluate the General Eqm Eqns
% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
GEeqnnames=fieldnames(GeneralEqmEqns);
GeneralEqmConditionsVec=zeros(1,sum(heteroagentoptions.GEptype==0)+PTypeStructure.N_i*sum(heteroagentoptions.GEptype==1));
% Some general eqm conditions are conditional on ptype, so go through one by one
gg_c=0;
for gg=1:length(GEeqnnames)
    if heteroagentoptions.GEptype(gg)==0 % Standard general eqm condition
        gg_c=gg_c+1;
        GeneralEqmConditionsVec(gg_c)=real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg_c).Names, Parameters));
    elseif heteroagentoptions.GEptype(gg)==1 % Do this general eqm condition conditional on ptype
        for ii=1:PTypeStructure.N_i % This General eqm condition has to hold conditional on each ptype
            gg_c=gg_c+1;
            GeneralEqmConditionsVec(gg_c)=real(GeneralEqmConditions_Case1_v3g_ptype(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg_c).Names, Parameters));
        end
    end
end


%% We might want to output GE conditions as a vector or structure
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
    gg_c=1;
    for gg=1:length(GEeqnnames)
        if heteroagentoptions.GEptype(gg)==0 % Standard general eqm condition
            GeneralEqmConditions.(GEeqnnames{gg})=GeneralEqmConditionsVec(gg_c);
            gg_c=gg_c+1;
        elseif heteroagentoptions.GEptype(gg)==1 % Do this general eqm condition conditional on ptype
            GeneralEqmConditions.(GEeqnnames{gg})=GeneralEqmConditionsVec(gg_c:gg_c+PTypeStructure.N_i-1);
            gg_c=gg_c+PTypeStructure.N_i;
        end
    end
end



%% Give feedback on current iteration
if heteroagentoptions.verbose==1
    if all(heteroagentoptions.GEptype==0)
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
    else
        numberstr=repmat(' %8.4f',1,PTypeStructure.N_i);
        % Adjust output for the fact there are multiple ptypes
        fprintf(' \n')
        fprintf('Current GE prices: \n')
        for pp=1:nGEprices
            if GEprice_ptype(pp)==1
                fprintf(['	%s:',numberstr,' \n'],GEPriceParamNames{pp},GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2)))
            else
                fprintf(['	%s: %8.4f \n'],GEPriceParamNames{pp},GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2)))
            end
        end
        fprintf('Current aggregate variables: \n')
        for aa=1:length(AggVarNames)
            fprintf('	%s: %8.4f \n',AggVarNames{aa},AggVars(aa)) % Note, this is done differently here because AggVars itself has been set as a matrix
        end
        fprintf('Current aggregate variables, conditional on ptype: \n')
        for aa=1:length(AggVarNames)
            fprintf(['	%s:',numberstr,' \n'],AggVarNames{aa},AggVars_ConditionalOnPType(aa,:)) % Note, this is done differently here because AggVars itself has been set as a matrix
        end
        if isfield(heteroagentoptions,'intermediateEqns')
            fprintf('Current intermediateEqns: \n')
            ggindex=ones(length(intEqnnames),1)+heteroagentoptions.intermediateEqnsptype'*(PTypeStructure.N_i-1);
            ggindex=[[1; cumsum(ggindex(1:end-1))+1],cumsum(ggindex)];
            for gg=1:length(intEqnnames)
                if heteroagentoptions.intermediateEqnsptype(gg)==1
                    fprintf(['	%s:',numberstr,' \n'],intEqnnames{gg},intermediateEqnsVec(ggindex(gg,1):ggindex(gg,2)))
                else
                    fprintf('	%s: %8.4f \n',intEqnnames{gg},intermediateEqnsVec(ggindex(gg,1):ggindex(gg,2)))
                end
            end
        end
        fprintf('Current GeneralEqmEqns: \n')
        GeneralEqmEqnsNames=fieldnames(GeneralEqmEqns);
        ggindex=ones(length(GeneralEqmEqnsNames),1)+heteroagentoptions.GEptype'*(PTypeStructure.N_i-1);
        ggindex=[[1; cumsum(ggindex(1:end-1))+1],cumsum(ggindex)];
        for gg=1:length(GeneralEqmEqnsNames)
            if heteroagentoptions.GEptype(gg)==1
                fprintf(['	%s:',numberstr,' \n'],GeneralEqmEqnsNames{gg},GeneralEqmConditionsVec(ggindex(gg,1):ggindex(gg,2)))
            else
                fprintf('	%s: %8.4f \n',GeneralEqmEqnsNames{gg},GeneralEqmConditionsVec(ggindex(gg,1):ggindex(gg,2)))            
            end
        end
    end
end



if heteroagentoptions.saveprogresseachiter==1
    save HeterAgentEqm_internal.mat GEprices Parameters GeneralEqmConditionsVec
end

end
