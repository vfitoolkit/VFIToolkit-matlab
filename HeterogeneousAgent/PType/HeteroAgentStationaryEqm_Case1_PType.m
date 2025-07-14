function varargout=HeteroAgentStationaryEqm_Case1_PType(n_d, n_a, n_z, Names_i, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, PTypeDistParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)
% Outputs: [p_eqm, GeneralEqmConditions]
% Unless you use n_p and p_grid, in which case [p_eq, p_eqm_index, GeneralEqmConditions]
%
% Allows for different permanent (fixed) types of agent. 
% See ValueFnIter_Case1_PType for general idea.
%
% How exactly to handle these differences between permanent (fixed) types
% is to some extent left to the user. You can, for example, input
% parameters that differ by permanent type as a vector with different rows f
% for each type, or as a structure with different fields for each type.
%
% Any input that does not depend on the permanent type is just passed in
% exactly the same form as normal.
%
% Names_i can either be a cell containing the 'names' of the different
% permanent types, or if there are no structures used (just parameters that
% depend on permanent type and inputted as vectors or matrices as appropriate) 
% then Names_i can just be the number of permanent types (but does not have to be, can still be names).


%% Check 'double fminalgo'
if exist('heteroagentoptions','var')
    if isfield(heteroagentoptions,'fminalgo')
        if length(heteroagentoptions.fminalgo)>1
            if isfield(heteroagentoptions,'toleranceGEcondns')
                heteroagentoptions.toleranceGEcondns=heteroagentoptions.toleranceGEcondns.*ones(1,length(heteroagentoptions.fminalgo));
            else
                heteroagentoptions.toleranceGEcondns=10^(-4).*ones(1,length(heteroagentoptions.fminalgo)); % Accuracy of general eqm prices
            end
            if isfield(heteroagentoptions,'toleranceGEprices')
                heteroagentoptions.toleranceGEprices=heteroagentoptions.toleranceGEprices.*ones(1,length(heteroagentoptions.fminalgo));
            else
                heteroagentoptions.toleranceGEprices=10^(-4).*ones(1,length(heteroagentoptions.fminalgo)); % Accuracy of general eqm prices
            end
            temp=heteroagentoptions.fminalgo;
            temp2=heteroagentoptions.toleranceGEcondns;
            temp3=heteroagentoptions.toleranceGEprices;
            heteroagentoptions.fminalgo=heteroagentoptions.fminalgo(1:end-1);
            heteroagentoptions.toleranceGEcondns=heteroagentoptions.toleranceGEcondns(1:end-1);
            heteroagentoptions.toleranceGEprices=heteroagentoptions.toleranceGEprices(1:end-1);
            p_eqm_previous=HeteroAgentStationaryEqm_Case1_PType(n_d, n_a, n_z, Names_i, n_p, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, PTypeDistParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
            for pp=1:length(GEPriceParamNames)
                Parameters.(GEPriceParamNames{pp})=p_eqm_previous.(GEPriceParamNames{pp});
            end
            heteroagentoptions.fminalgo=temp(end);
            heteroagentoptions.toleranceGEcondns=temp2(end);
            heteroagentoptions.toleranceGEprices=temp3(end);
        end
    end
end


%% Check which options have been used, set all others to defaults 
N_p=prod(n_p);
if isempty(n_p)
    N_p=0;
end

if exist('heteroagentoptions','var')==0
    heteroagentoptions.multiGEcriterion=1;
    heteroagentoptions.multiGEweights=ones(1,length(fieldnames(GeneralEqmEqns)));
    heteroagentoptions.toleranceGEprices=10^(-4); % Accuracy of general eqm prices
    heteroagentoptions.toleranceGEcondns=10^(-4); % Accuracy of general eqm eqns
    heteroagentoptions.verbose=0;
    heteroagentoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    heteroagentoptions.fminalgo=1; % use fminsearch
    heteroagentoptions.maxiter=1000; % maximum iterations of optimization routine
    heteroagentoptions.GEptype={}; % zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type' [input should be a cell of names; it gets reformatted internally to be this form]
    % Constrain parameters
    heteroagentoptions.constrainpositive={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    heteroagentoptions.constrain0to1={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    heteroagentoptions.constrainAtoB={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    % heteroagentoptions.outputGEform=0; % For internal use only
    heteroagentoptions.outputGEstruct=1; % output GE conditions as a structure (=2 will output as a vector)
    heteroagentoptions.outputgather=1; % output GE conditions on CPU [some optimization routines only work on CPU, some can handle GPU]
else
    if isfield(heteroagentoptions,'multiGEcriterion')==0
        heteroagentoptions.multiGEcriterion=1;
    end
    if isfield(heteroagentoptions,'multiGEweights')==0
        heteroagentoptions.multiGEweights=ones(1,length(fieldnames(GeneralEqmEqns)));
    end
    if N_p~=0
        if isfield(heteroagentoptions,'p_grid')==0
            disp('ERROR: you have set n_p to a non-zero value, but not declared heteroagentoptions.pgrid')
            dbstack
        end
    end
    if isfield(heteroagentoptions,'toleranceGEprices')==0
        heteroagentoptions.toleranceGEprices=10^(-4); % Accuracy of general eqm prices
    end
    if isfield(heteroagentoptions,'toleranceGEcondns')==0
        heteroagentoptions.toleranceGEcondns=10^(-4); % Accuracy of general eqm prices
    end
    if isfield(heteroagentoptions,'verbose')==0
        heteroagentoptions.verbose=0;
    end
    if isfield(heteroagentoptions,'fminalgo')==0
        heteroagentoptions.fminalgo=1; % use fminsearch
    end
    if isfield(heteroagentoptions,'parallel')==0
        heteroagentoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if isfield(heteroagentoptions,'maxiter')==0
        heteroagentoptions.maxiter=1000; % maximum iterations of optimization routine
    end
    if ~isfield(heteroagentoptions,'GEptype')
        heteroagentoptions.GEptype={}; % zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type' [input should be a cell of names; it gets reformatted internally to be this form]
    end
    % Constrain parameters
    if ~isfield(heteroagentoptions,'constrainpositive')
        heteroagentoptions.constrainpositive={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
        % Convert constrained positive p into x=log(p) which is unconstrained.
        % Then use p=exp(x) in the model.
    end
    if ~isfield(heteroagentoptions,'constrain0to1')
        heteroagentoptions.constrain0to1={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
        % Handle 0 to 1 constraints by using log-odds function to switch parameter p into unconstrained x, so x=log(p/(1-p))
        % Then use the logistic-sigmoid p=1/(1+exp(-x)) when evaluating model.
    end
    if ~isfield(heteroagentoptions,'constrainAtoB')
        heteroagentoptions.constrainAtoB={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
        % Handle A to B constraints by converting y=(p-A)/(B-A) which is 0 to 1, and then treating as constrained 0 to 1 y (so convert to unconstrained x using log-odds function)
        % Once we have the 0 to 1 y (by converting unconstrained x with the logistic sigmoid function), we convert to p=A+(B-a)*y
    else
        if ~isfield(heteroagentoptions,'constrainAtoBlimits')
            error('You have used heteroagentoptions.constrainAtoB, but are missing heteroagentoptions.constrainAtoBlimits')
        end
    end
    % heteroagentoptions.outputGEform=0; % For internal use only
    if ~isfield(heteroagentoptions,'outputGEstruct')
        heteroagentoptions.outputGEstruct=1; % output GE conditions as a structure (=2 will output as a vector)
    end
    if ~isfield(heteroagentoptions,'outputgather')
        heteroagentoptions.outputgather=1; % output GE conditions on CPU [some optimization routines only work on CPU, some can handle GPU]
    end
end

heteroagentoptions.useCustomModelStats=0;
if isfield(heteroagentoptions,'CustomModelStats')
    heteroagentoptions.useCustomModelStats=1;
end

if heteroagentoptions.fminalgo==0
    heteroagentoptions.outputGEform=1;
elseif heteroagentoptions.fminalgo==5
    heteroagentoptions.outputGEform=1; % Need to output GE condns as a vector when using fminalgo=5
    heteroagentoptions.outputgather=0; % leave GE condns vector on GPU
elseif heteroagentoptions.fminalgo==7
    heteroagentoptions.outputGEform=1; % Need to output GE condns as a vector when using fminalgo=7
else
    heteroagentoptions.outputGEform=0;
end

temp=size(heteroagentoptions.multiGEweights);
if temp(2)==1 % probably column vector
    heteroagentoptions.multiGEweights=heteroagentoptions.multiGEweights'; % make row vector
end
if length(heteroagentoptions.multiGEweights)~=length(fieldnames(GeneralEqmEqns))
    error('length(heteroagentoptions.multiGEweights)~=length(fieldnames(GeneralEqmEqns)) (the length of the GE weights is not equal to the number of general eqm equations')
end

AggVarNames=fieldnames(FnsToEvaluate);
nGEprices=length(GEPriceParamNames);

PTypeStructure.numFnsToEvaluate=length(fieldnames(FnsToEvaluate)); % Total number of functions to evaluate

%%
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i;
    Names_i={'ptype001'};
    for ii=2:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

%% Reformat heteroagentoptions.GEptype from cell of names into vector of 1s and 0s
if isempty(heteroagentoptions.GEptype)
    heteroagentoptions.GEptype=zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type'
else
    temp=heteroagentoptions.GEptype;
    heteroagentoptions.GEptype=zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type'
    GEeqnNames=fieldnames(GeneralEqmEqns);
    for gg1=1:length(temp)
        for gg2=1:length(GEeqnNames)
            if strcmp(temp{gg1},GEeqnNames{gg2})
                heteroagentoptions.GEptype(gg2)=1;
            end
        end
    end
end
if any(heteroagentoptions.GEptype==1)
    % Adjust the size of weights for the dependence on ptype
    temp=heteroagentoptions.multiGEweights;
    heteroagentoptions.multiGEweights=zeros(1,sum(heteroagentoptions.GEptype==0)+N_i*sum(heteroagentoptions.GEptype==1));
    gg_c=0;
    for gg=1:length(fieldnames(GeneralEqmEqns))
        if heteroagentoptions.GEptype(gg)==0
            gg_c=gg_c+1;
            heteroagentoptions.multiGEweights(gg_c)=temp(gg);
        elseif heteroagentoptions.GEptype(gg)==1
            for ii=1:N_i
                gg_c=gg_c+1;
                heteroagentoptions.multiGEweights(gg_c)=temp(gg);
            end
        end
    end
    if ~isfield(heteroagentoptions,'GEptype_vectoroutput')
        heteroagentoptions.GEptype_vectoroutput=0;
    end
elseif length(heteroagentoptions.multiGEweights)~=length(fieldnames(GeneralEqmEqns))
    error('length(heteroagentoptions.multiGEweights)~=length(fieldnames(GeneralEqmEqns)) (the length of the GE weights is not equal to the number of general eqm equations')
end


%%
% PTypeStructure contains everything for the different permanent types.
% There are two fields, PTypeStructure.Names_i, and PTypeStructure.N_i.
% Then everything else is stored in, eg., PTypeStructure.ptype004, for the
% 4th permanent type.
%
% Code implicitly assumes that simoptions.agedependentgrids contains the same as
% vfoptions.agedependentgrids. Seems likely that you would always want this
% to be the case anyway.

% PTypeStructure.Names_i never really gets used. Just makes things easier
% to read when you are looking at PTypeStructure (which only ever exists
% internally to the VFI Toolkit)
if iscell(Names_i)
    PTypeStructure.Names_i=Names_i;
    PTypeStructure.N_i=length(Names_i);
else
    PTypeStructure.N_i=Names_i;
    PTypeStructure.Names_i={'ptype001'};
    for ii=2:PTypeStructure.N_i
        if ii<10
            PTypeStructure.Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            PTypeStructure.Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            PTypeStructure.Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

for ii=1:PTypeStructure.N_i

    iistr=PTypeStructure.Names_i{ii};
    PTypeStructure.iistr{ii}=iistr;
    
    if exist('vfoptions','var') % vfoptions.verbose (allowed to depend on permanent type)
        if ~isempty(vfoptions)
            PTypeStructure.(iistr).vfoptions=PType_Options(vfoptions,Names_i,ii); % some vfoptions will differ by permanent type, will clean these up as we go before they are passed
        else
            PTypeStructure.(iistr).simoptions.verbose=0;
        end
    else
        PTypeStructure.(iistr).vfoptions.verbose=0;
    end
    
    if exist('simoptions','var') % vfoptions.verbose (allowed to depend on permanent type)
        if ~isempty(simoptions)
            PTypeStructure.(iistr).simoptions=PType_Options(simoptions,Names_i,ii); % some vfoptions will differ by permanent type, will clean these up as we go before they are passed
        else
            PTypeStructure.(iistr).simoptions.verbose=0;
        end
    else
        PTypeStructure.(iistr).simoptions.verbose=0;
    end
    
    PTypeStructure.(iistr).simoptions.outputasstructure=0; % Used by AggVars (in heteroagent subfn)

    if heteroagentoptions.verbose==1
        fprintf('Setting up, Permanent type: %i of %i \n',ii, PTypeStructure.N_i)
    end

    % Go through everything which might be dependent on permanent type (PType)
    % Notice that the way this is coded the grids (etc.) could be either
    % fixed, or a function (that depends on age, and possibly on permanent
    % type), or they could be a structure. Only in the case where they are
    % a structure is there a need to take just a specific part and send
    % only that to the 'non-PType' version of the command.
    
    % Start with those that determine whether the current permanent type is finite or
    % infinite horizon, and whether it is Case 1 or Case 2
    % Figure out which case is relevant to the current PType. This is done
    % using N_j which for the current type will evaluate to 'Inf' if it is
    % infinite horizon and a finite number for any other finite horizon.
    % First, check if it is a structure, and otherwise just get the
    % relevant value.
    
    PTypeStructure.(iistr).n_d=n_d;
    if isa(n_d,'struct')
        PTypeStructure.(iistr).n_d=n_d.(Names_i{ii});
    else
        temp=size(n_d);
        if temp(1)>1 % n_d depends on fixed type
            PTypeStructure.(iistr).n_d=n_d(ii,:);
        elseif temp(2)==PTypeStructure.N_i % If there is one row, but number of elements in n_d happens to coincide with number of permanent types, then just let user know
            warning('Possible Warning: Number of columns of n_d is the same as the number of permanent types. \n This may just be coincidence as number of d variables is equal to number of permanent types. \n If they are intended to be permanent types then n_d should have them as different rows (not columns). \n')
        end
    end
    PTypeStructure.(iistr).n_a=n_a;
    if isa(n_a,'struct')
        PTypeStructure.(iistr).n_a=n_a.(Names_i{ii});
    else
        temp=size(n_a);
        if temp(1)>1 % n_a depends on fixed type
            PTypeStructure.(iistr).n_a=n_a(ii,:);
        elseif temp(2)==PTypeStructure.N_i % If there is one row, but number of elements in n_a happens to coincide with number of permanent types, then just let user know
            warning('Possible Warning: Number of columns of n_a is the same as the number of permanent types. \n This may just be coincidence as number of a variables is equal to number of permanent types. \n If they are intended to be permanent types then n_a should have them as different rows (not columns). \n')
        end
    end
    PTypeStructure.(iistr).n_z=n_z;
    if isa(n_z,'struct')
        PTypeStructure.(iistr).n_z=n_z.(Names_i{ii});
    else
        temp=size(n_z);
        if temp(1)>1 % n_z depends on fixed type
            PTypeStructure.(iistr).n_z=n_z(ii,:);
        elseif temp(2)==PTypeStructure.N_i % If there is one row, but number of elements in n_d happens to coincide with number of permanent types, then just let user know
            warning('Possible Warning: Number of columns of n_z is the same as the number of permanent types. \n This may just be coincidence as number of z variables is equal to number of permanent types. \n If they are intended to be permanent types then n_z should have them as different rows (not columns). \n')
        end
    end
    
    if PTypeStructure.(iistr).n_d(1)==0
        PTypeStructure.(iistr).l_d=0;
    else
        PTypeStructure.(iistr).l_d=length(PTypeStructure.(iistr).n_d);
    end
    PTypeStructure.(iistr).l_a=length(PTypeStructure.(iistr).n_a);
    PTypeStructure.(iistr).l_z=length(PTypeStructure.(iistr).n_z);
    if isfield(PTypeStructure.(iistr).simoptions,'n_e')
        PTypeStructure.(iistr).l_e=length(PTypeStructure.(iistr).simoptions.n_e);
    else
        PTypeStructure.(iistr).l_e=0;
    end
    
    PTypeStructure.(iistr).d_grid=d_grid;
    if isa(d_grid,'struct')
        PTypeStructure.(iistr).d_grid=d_grid.(Names_i{ii});
    end
    PTypeStructure.(iistr).a_grid=a_grid;
    if isa(a_grid,'struct')
        PTypeStructure.(iistr).a_grid=a_grid.(Names_i{ii});
    end
    PTypeStructure.(iistr).z_grid=z_grid;
    if isa(z_grid,'struct')
        PTypeStructure.(iistr).z_grid=z_grid.(Names_i{ii});
    end
    PTypeStructure.(iistr).pi_z=pi_z;
    if isa(pi_z,'struct')
        PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii});
    end

    %%
    PTypeStructure.(iistr).ReturnFn=ReturnFn;
    if isa(ReturnFn,'struct')
        PTypeStructure.(iistr).ReturnFn=ReturnFn.(Names_i{ii});
    end
    temp=getAnonymousFnInputNames(PTypeStructure.(iistr).ReturnFn);
    if length(temp)>(PTypeStructure.(iistr).l_d+PTypeStructure.(iistr).l_a+PTypeStructure.(iistr).l_a+PTypeStructure.(iistr).l_z+PTypeStructure.(iistr).l_e) % This is largely pointless, the ReturnFn is always going to have some parameters
        PTypeStructure.(iistr).ReturnFnParamNames={temp{PTypeStructure.(iistr).l_d+PTypeStructure.(iistr).l_a+PTypeStructure.(iistr).l_a+PTypeStructure.(iistr).l_z+PTypeStructure.(iistr).l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        PTypeStructure.(iistr).ReturnFnParamNames={};
    end
    
    % Parameters are allowed to be given as structure, or as vector/matrix
    % (in terms of their dependence on permanent type). So go through each of
    % these in term.
    % ie. Parameters.alpha=[0;1]; or Parameters.alpha.ptype1=0; Parameters.alpha.ptype2=1;
    PTypeStructure.(iistr).Parameters=Parameters;
    FullParamNames=fieldnames(Parameters); % all the different parameters
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check the current parameter for permanent type in structure form
            % Check if this parameter is used for the current permanent type (it may or may not be, some parameters are only used be a subset of permanent types)
            if isfield(Parameters.(FullParamNames{kField}),Names_i{ii})
                PTypeStructure.(iistr).Parameters.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(Names_i{ii});
            end
        elseif sum(size(Parameters.(FullParamNames{kField}))==PTypeStructure.N_i)>=1 % Check for permanent type in vector/matrix form.
            temp=Parameters.(FullParamNames{kField});
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==PTypeStructure.N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType, it should be the row dimension, if it is not then give a warning.
            if ptypedim==1
                PTypeStructure.(iistr).Parameters.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                PTypeStructure.(iistr).Parameters.(FullParamNames{kField})=temp(:,ii);
            end
        end
    end
    % THIS TREATMENT OF PARAMETERS COULD BE IMPROVED TO BETTER DETECT INPUT SHAPE ERRORS.
    
    %% Set up exogenous shock grids now (so they can then just be reused every time)
    % Check if using ExogShockFn or EiidShockFn, and if so, do these use a
    % parameter that is being determined in general eqm
    heteroagentoptions.gridsinGE(ii)=0;
    if isfield(PTypeStructure.(iistr).vfoptions,'ExogShockFn')
        tempExogShockFnParamNames=getAnonymousFnInputNames(PTypeStructure.(iistr).vfoptions.ExogShockFn);
        % can just leave action space in here as we only use it to see if GEPriceParamNames is part of it
        if ~isempty(intersect(tempExogShockFnParamNames,GEPriceParamNames))
            heteroagentoptions.gridsinGE(ii)=1;
        end
    end
    if isfield(PTypeStructure.(iistr).vfoptions,'EiidShockFn')
        tempEiidShockFnParamNames=getAnonymousFnInputNames(PTypeStructure.(iistr).vfoptions.EiidShockFn);
        % can just leave action space in here as we only use it to see if GEPriceParamNames is part of it
        if ~isempty(intersect(tempEiidShockFnParamNames,GEPriceParamNames))
            heteroagentoptions.gridsinGE(ii)=1;
        end
    end
    % If z (and e) are not determined in GE, then compute z_gridvals_J and pi_z_J now (and e_gridvals_J and pi_e_J)
    if heteroagentoptions.gridsinGE(ii)==0
        % Some of the shock grids depend on parameters that are determined in general eqm
        [PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).vfoptions]=ExogShockSetup(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
        % Note: these are actually z_gridvals and pi_z
        PTypeStructure.(iistr).simoptions.e_gridvals=PTypeStructure.(iistr).vfoptions.e_gridvals; % Note, will be [] if no e
        PTypeStructure.(iistr).simoptions.pi_e=PTypeStructure.(iistr).vfoptions.pi_e; % Note, will be [] if no e
    end
    % Regardless of whether they are done here of in _subfn, they will be
    % precomputed by the time we get to the value fn, staty dist, etc. So
    PTypeStructure.(iistr).vfoptions.alreadygridvals=1;
    PTypeStructure.(iistr).simoptions.alreadygridvals=1;
       

    %%
    % The parameter names can be made to depend on the permanent-type
    PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames.(Names_i{ii});
    end

    % Figure out which functions are actually relevant to the present PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluate and FnsToEvaluateFnParamNames are necessarily the same.
    % Allows for FnsToEvaluate as structure.
    if PTypeStructure.(iistr).n_d(1)==0
        l_d_temp=0;
    else
        l_d_temp=1;
    end
    l_a_temp=length(PTypeStructure.(iistr).n_a);
    l_z_temp=length(PTypeStructure.(iistr).n_z);  
    [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,~]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    PTypeStructure.(iistr).FnsToEvaluate=FnsToEvaluate_temp;
    PTypeStructure.(iistr).FnsToEvaluateParamNames=FnsToEvaluateParamNames_temp;
    PTypeStructure.(iistr).WhichFnsForCurrentPType=WhichFnsForCurrentPType;
    
    if isa(PTypeDistParamNames, 'array')
        PTypeStructure.(iistr).PTypeWeight=PTypeDistParamNames(ii);
    else
        PTypeStructure.(iistr).PTypeWeight=PTypeStructure.(iistr).Parameters.(PTypeDistParamNames{1}); % Don't need '.(Names_i{ii}' as this was already done when putting it into PTypeStrucutre, and here I take it straing from PTypeStructure.(iistr).Parameters rather than from Parameters itself.
    end
end




%% If using intermediateEqns, switch from structure to cell setup
AggVarNames_mod=AggVarNames; % AggVarNames_mod is used to check which inputs need to depend on ptype for things that are done by ptype

heteroagentoptions.useintermediateEqns=0;
if isfield(heteroagentoptions,'intermediateEqns')
    heteroagentoptions.useintermediateEqns=1;
    intEqnNames=fieldnames(heteroagentoptions.intermediateEqns);
    nIntEqns=length(intEqnNames);

    if isfield(heteroagentoptions,'intermediateEqnsptype')
        temp=heteroagentoptions.intermediateEqnsptype;
        heteroagentoptions.intermediateEqnsptype=zeros(1,nIntEqns); % 1 indicates that this intermediate eqn is 'conditional on permanent type'
        for gg1=1:length(temp)
            for gg2=1:length(intEqnNames)
                if strcmp(temp{gg1},intEqnNames{gg2})
                    heteroagentoptions.intermediateEqnsptype(gg2)=1;
                end
            end
        end
    else
        heteroagentoptions.intermediateEqnsptype=zeros(1,nIntEqns); % 1 indicates that this intermediate eqn is 'conditional on permanent type'
    end
    
    heteroagentoptions.intermediateEqnsCell=cell(1,nIntEqns);
    gg_c=0;
    for gg=1:nIntEqns
        intEqnnames_gg=intEqnNames{gg};
        temp=getAnonymousFnInputNames(heteroagentoptions.intermediateEqns.(intEqnNames{gg}));
        if heteroagentoptions.intermediateEqnsptype(gg)==1
            AggVarNames_mod{length(AggVarNames_mod)+1}=intEqnNames{gg}; % add to AggVarNames_mod in case used as input later

            for ii=1:N_i
                temp_ii=temp;
                gg_c=gg_c+1;
                for tt=1:length(temp_ii)
                    % Need to check if it is in AggVarNames_mod, in which case use '_name'
                    if any(strcmp(AggVarNames_mod,temp_ii{tt}))
                        for aa=1:length(AggVarNames_mod)
                            if strcmp(AggVarNames_mod{aa},temp_ii{tt})
                                temp_ii{tt}=[temp_ii{tt},'_',Names_i{ii}]; % use the '_name' for the AggVar inputs
                            end
                        end
                    end
                    % Need to check if it a parameter that depends on ptype, in which case use '_name'
                    if any(strcmp(paramnamesptype,temp_ii{tt}))
                        for pp=1:length(paramnamesptype)
                            if strcmp(paramnamesptype{pp},temp_ii{tt})
                                temp_ii{tt}=[temp_ii{tt},'_',Names_i{ii}]; % use the '_name' for the AggVar inputs
                            end
                        end
                    end
                end
                heteroagentoptions.intermediateEqnParamNames(gg_c).Names=temp_ii;
            end
        else
            gg_c=gg_c+1;
            heteroagentoptions.intermediateEqnParamNames(gg_c).Names=temp;

            % check if it is an _name, in which case need to put it into AggVarNames_mod so that it gets handled correctly if it is used as an input later
            checkunderscorename=0;
            for ii=1:N_i
                lname=length(Names_i{ii});
                if length(intEqnnames_gg)>lname+1 % only check if intEqnnames_gg is long enough to be possible
                    if strcmp(intEqnnames_gg(end-lname:end),['_',Names_i{ii}])
                        % E.g., creates Parameters.r.ptype001 from Parameters.r_ptype001
                        checkunderscorename=checkunderscorename+1;
                        intEqnNames_ggmod=intEqnnames_gg(1:end-lname-1);
                    end
                end
            end
            if checkunderscorename==1
                if ~any(strcmp(AggVarNames_mod,intEqnNames_ggmod))
                    AggVarNames_mod{length(AggVarNames_mod)+1}=intEqnNames_ggmod; % add to AggVarNames_mod in case used as input later
                end
            end
        end
        heteroagentoptions.intermediateEqnsCell{gg}=heteroagentoptions.intermediateEqns.(intEqnNames{gg});        
    end
    % Now:
    %  heteroagentoptions.intermediateEqns is still the structure
    %  heteroagentoptions.intermediateEqnsCell is cell
    %  heteroagentoptions.intermediateEqnParamNames(gg_c).Names contains the names
    % Note: 
    % intermediateEqnParamNames is based on gg_c, so that they can differ when using by-ptype in which case they use '_name'

end


%% GE eqns, switch from structure to cell setup
GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);

GeneralEqmEqnsCell=cell(1,nGeneralEqmEqns);
gg_c=0;
for gg=1:nGeneralEqmEqns
    temp=getAnonymousFnInputNames(GeneralEqmEqns.(GEeqnNames{gg}));
    if heteroagentoptions.GEptype(gg)==1
        for ii=1:N_i
            temp_ii=temp;
            gg_c=gg_c+1;
            for tt=1:length(temp_ii)
                % Need to check if it is in AggVarNames_mod, in which case use '_name'
                if any(strcmp(AggVarNames_mod,temp_ii{tt}))
                    for aa=1:length(AggVarNames_mod)
                        if strcmp(AggVarNames_mod{aa},temp_ii{tt})
                            temp_ii{tt}=[temp_ii{tt},'_',Names_i{ii}]; % use the '_name' for the AggVar inputs
                        end
                    end
                end
                % Need to check if it a parameter that depends on ptype, in which case use '_name'
                if any(strcmp(paramnamesptype,temp_ii{tt}))
                    for pp=1:length(paramnamesptype)
                        if strcmp(paramnamesptype{pp},temp_ii{tt})
                            temp_ii{tt}=[temp_ii{tt},'_',Names_i{ii}]; % use the '_name' for the AggVar inputs
                        end
                    end
                end
            end
            GeneralEqmEqnParamNames(gg_c).Names=temp_ii;
        end
    else
        gg_c=gg_c+1;
        GeneralEqmEqnParamNames(gg_c).Names=temp;
    end
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns.(GEeqnNames{gg});
end
% Now: 
%  GeneralEqmEqns is still the structure
%  GeneralEqmEqnsCell is cell
%  GeneralEqmEqnParamNames(gg_c).Names contains the names
% Note: 
% GeneralEqmEqnParamNames is based on gg_c, so that they can differ when using by-ptype in which case they use '_name'


%% Permit that some GEPriceParamNames might depend on PType
GEparamsvec0=[]; % column vector
GEpriceindexes=zeros(nGEprices,1);
GEprice_ptype=zeros(nGEprices,1);
for pp=1:nGEprices
    if isstruct(Parameters.(GEPriceParamNames{pp}))
        for ii=1:PTypeStructure.N_i
            iistr=PTypeStructure.Names_i{ii};
            GEparamsvec0=[GEparamsvec0; gather(Parameters.(GEPriceParamNames{pp}).(iistr))]; % reshape()' is making sure it is a row vector
        end
        GEpriceindexes(pp)=PTypeStructure.N_i;
        GEprice_ptype(pp)=1;
    else
        GEparamsvec0=[GEparamsvec0;reshape(gather(Parameters.(GEPriceParamNames{pp})),[],1)]; % reshape() is making sure it is a column vector
        GEpriceindexes(pp)=length(Parameters.(GEPriceParamNames{pp}));
        if length(Parameters.(GEPriceParamNames{pp}))>1
            GEprice_ptype(pp)=1;
        end
    end
end
GEpriceindexes=[[1; 1+cumsum(GEpriceindexes(1:end-1))],cumsum(GEpriceindexes)];


%% Set up GEparamsvec0 and parameter constraints
% Backup the parameter constraint names, so I can replace them with vectors
heteroagentoptions.constrainpositivenames=heteroagentoptions.constrainpositive;
heteroagentoptions.constrainpositive=zeros(nGEprices,1); % if equal 1, then that parameter is constrained to be positive
heteroagentoptions.constrain0to1names=heteroagentoptions.constrain0to1;
heteroagentoptions.constrain0to1=zeros(nGEprices,1); % if equal 1, then that parameter is constrained to be 0 to 1
heteroagentoptions.constrainAtoBnames=heteroagentoptions.constrainAtoB;
heteroagentoptions.constrainAtoB=zeros(nGEprices,1); % if equal 1, then that parameter is constrained to be 0 to 1
if ~isempty(heteroagentoptions.constrainAtoBnames)
    heteroagentoptions.constrainAtoBlimitsnames=heteroagentoptions.constrainAtoBlimits;
    heteroagentoptions.constrainAtoBlimits=zeros(nGEprices,2); % rows are parameters, column is lower (A) and upper (B) bounds [row will be [0,0] is unconstrained]
end
for pp=1:nGEprices
    % First, check the name, and convert it if relevant
    if any(strcmp(heteroagentoptions.constrainpositivenames,GEPriceParamNames{pp}))
        heteroagentoptions.constrainpositive(pp)=1;
    end
    if any(strcmp(heteroagentoptions.constrain0to1names,GEPriceParamNames{pp}))
        heteroagentoptions.constrain0to1(pp)=1;
    end
    if any(strcmp(heteroagentoptions.constrainAtoBnames,GEPriceParamNames{pp}))
        % For parameters A to B, I convert via 0 to 1
        heteroagentoptions.constrain0to1(pp)=1;
        heteroagentoptions.constrainAtoB(pp)=1;
        heteroagentoptions.constrainAtoBlimits(pp,:)=heteroagentoptions.constrainAtoBlimitsnames.(GEPriceParamNames{pp});
    end
    if heteroagentoptions.constrainpositive(pp)==1
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        GEparamsvec0(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=max(log(GEparamsvec0(GEpriceindexes(pp,1):GEpriceindexes(pp,2))),-49.99);
        % Note, the max() is because otherwise p=0 returns -Inf. [Matlab evaluates exp(-50) as about 10^-22, I overrule and use exp(-50) as zero, so I set -49.99 here so solver can realise the boundary is there; not sure if this setting -49.99 instead of my -50 cutoff actually helps, but seems like it might so I have done it here].
    end
    if heteroagentoptions.constrainAtoB(pp)==1
        % Constraint parameter to be A to B (by first converting to 0 to 1, and then treating it as contraint 0 to 1)
        GEparamsvec0(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=(GEparamsvec0(GEpriceindexes(pp,1):GEpriceindexes(pp,2))-caliboptions.constrainAtoBlimits(pp,1))/(caliboptions.constrainAtoBlimits(pp,2)-caliboptions.constrainAtoBlimits(pp,1));
        % x=(y-A)/(B-A), converts A-to-B y, into 0-to-1 x
        % And then the next if-statement converts this 0-to-1 into unconstrained
    end
    if heteroagentoptions.constrain0to1(pp)==1
        % Constrain parameter to be 0 to 1 (be working with log(p/(1-p)), where p is parameter) then always take exp()/(1+exp()) before inputting to model
        GEparamsvec0(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=min(49.99,max(-49.99,  log(GEparamsvec0(GEpriceindexes(pp,1):GEpriceindexes(pp,2))/(1-GEparamsvec0(GEpriceindexes(pp,1):GEpriceindexes(pp,2)))) ));
        % Note: the max() and min() are because otherwise p=0 or 1 returns -Inf or Inf [Matlab evaluates 1/(1+exp(-50)) as one, and 1/(1+exp(50)) as about 10^-22, so I overrule them as 1 and 0, so I set -49.99 here so solver can realise the boundary is there; not sure if this setting -49.99 instead of my -50 cutoff actually helps, but seems like it might so I have done it here].
    end
    if heteroagentoptions.constrainpositive(pp)==1 && heteroagentoptions.constrain0to1(pp)==1 % Double check of inputs
        fprinf(['Relating to following error message: Parameter ',num2str(pp),' of ',num2str(length(GEPriceParamNames))])
        error('You cannot constrain parameter twice (you are constraining one of the parameters using both heteroagentoptions.constrainpositive and in one of heteroagentoptions.constrain0to1 and heteroagentoptions.constrainAtoB')
    end
end


%% Have now finished creating PTypeStructure. Time to do the actual finding the HeteroAgentStationaryEqm:


%% If using fminalgo=5, then need some further setup

if heteroagentoptions.fminalgo==5
    heteroagentoptions.weightscheme=0; % Don't do any weightscheme, is already taken care of by GEnewprice=3
    
    if isstruct(GeneralEqmEqns) 
        % Need to make sure that order of rows in transpathoptions.GEnewprice3.howtoupdate
        % Is same as order of fields in GeneralEqmEqns
        % I do this by just reordering rows of transpathoptions.GEnewprice3.howtoupdate
        temp=heteroagentoptions.fminalgo5.howtoupdate;
        GEeqnNames=fieldnames(GeneralEqmEqns);
        for tt=1:length(GEeqnNames)
            for jj=1:size(temp,1)
                if strcmp(temp{jj,1},GEeqnNames{tt}) % Names match
                    heteroagentoptions.fminalgo5.howtoupdate{tt,1}=temp{jj,1};
                    heteroagentoptions.fminalgo5.howtoupdate{tt,2}=temp{jj,2};
                    heteroagentoptions.fminalgo5.howtoupdate{tt,3}=temp{jj,3};
                    heteroagentoptions.fminalgo5.howtoupdate{tt,4}=temp{jj,4};
                end
            end
        end
        nGeneralEqmEqns=length(GEeqnNames);
    else
        nGeneralEqmEqns=length(GeneralEqmEqns);
    end
    heteroagentoptions.fminalgo5.add=[heteroagentoptions.fminalgo5.howtoupdate{:,3}];
    heteroagentoptions.fminalgo5.factor=[heteroagentoptions.fminalgo5.howtoupdate{:,4}];
    heteroagentoptions.fminalgo5.keepold=ones(size(heteroagentoptions.fminalgo5.factor));
    heteroagentoptions.fminalgo5.keepold=ones(size(heteroagentoptions.fminalgo5.factor));
    
    if size(heteroagentoptions.fminalgo5.howtoupdate,1)==nGeneralEqmEqns && nGeneralEqmEqns==length(GEPriceParamNames)
        % do nothing, this is how things should be
    else
        fprintf('ERROR: heteroagentoptions.fminalgo5..howtoupdate does not fit with GeneralEqmEqns (different number of conditions/prices) \n')
    end
    heteroagentoptions.fminalgo5.permute=zeros(size(heteroagentoptions.fminalgo5.howtoupdate,1),1);
    for tt=1:size(heteroagentoptions.fminalgo5.howtoupdate,1) % number of rows is the number of prices (and number of GE conditions)
        for jj=1:length(GEPriceParamNames)
            if strcmp(heteroagentoptions.fminalgo5.howtoupdate{tt,2},GEPriceParamNames{jj})
                heteroagentoptions.fminalgo5.permute(tt)=jj;
            end
        end
    end
    if isfield(heteroagentoptions,'updateaccuracycutoff')==0
        heteroagentoptions.updateaccuracycutoff=0; % No cut-off (only changes in the price larger in magnitude that this will be made (can be set to, e.g., 10^(-6) to help avoid changes at overly high precision))
    end
end



%%
if heteroagentoptions.maxiter>0 % Can use heteroagentoptions.maxiter=0 to just evaluate the current general eqm eqns
    
    %%  Otherwise, use fminsearch to find the general equilibrium
    if all(heteroagentoptions.GEptype==0)
        if heteroagentoptions.fminalgo~=8 && heteroagentoptions.fminalgo~=3
            GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_PType_subfn(p, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices,heteroagentoptions);
        elseif heteroagentoptions.fminalgo==3
            heteroagentoptions.outputGEform=1; % vector
            GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_PType_subfn(p, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions);
        elseif heteroagentoptions.fminalgo==8
            heteroagentoptions.outputGEform=1; % vector
            weightsbackup=heteroagentoptions.multiGEweights;
            heteroagentoptions.multiGEweights=sqrt(heteroagentoptions.multiGEweights); % To use a weighting matrix in lsqnonlin(), we work with the square-roots of the weights
            GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_PType_subfn(p, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions);
            heteroagentoptions.multiGEweights=weightsbackup; % change it back now that we have set up CalibrateLifeCycleModel_objectivefn()
        end
    else
        error('Have not actually yet implemented GEptype for infinite horizon, contact me and I will do so')
    end

    GEparamsvec0=nan(nGEprices,1);
    for ii=1:nGEprices
        GEparamsvec0(ii)=Parameters.(GEPriceParamNames{ii});
    end

    % Choosing algorithm for the optimization problem
    % https://au.mathworks.com/help/optim/ug/choosing-the-algorithm.html#bscj42s
    minoptions = optimset('TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns);
    p_eqm_index=nan; % If not using p_grid then this is irrelevant/useless
    if N_p~=0 % Solving on p_grid
        GeneralEqmConditions=zeros(size(heteroagentoptions.p_grid));
        for pp_c=1:size(heteroagentoptions.p_grid,1)
            pvec=heteroagentoptions.p_grid(pp_c,:);
            GeneralEqmConditions(pp_c,:)=GeneralEqmConditionsFnOpt(pvec);
        end
        [~,p_eqm_index]=max(sum(GeneralEqmConditions.^2,2));
        p_eqm=heteroagentoptions.p_grid(p_eqm_index,:);
    elseif heteroagentoptions.fminalgo==0 % fzero, is based on root-finding so it needs just the vector of GEcondns, not the sum-of-squares (it is not a minimization routine)
        [p_eqm_vec,GeneralEqmConditions]=fzero(GeneralEqmConditionsFnOpt,GEparamsvec0,minoptions);
    elseif heteroagentoptions.fminalgo==1
        [p_eqm_vec,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFnOpt,GEparamsvec0,minoptions);
    elseif heteroagentoptions.fminalgo==2
        % Use the optimization toolbox so as to take advantage of automatic differentiation
        z=optimvar('z',length(GEparamsvec0));
        optimfun=fcn2optimexpr(GeneralEqmConditionsFnOpt, z);
        prob = optimproblem("Objective",optimfun);
        z0.z=GEparamsvec0;
        [sol,GeneralEqmConditions]=solve(prob,z0);
        p_eqm_vec=sol.z;
        % Note, doesn't really work as automattic differentiation is only for
        % supported functions, and the objective here is not a supported function
    elseif heteroagentoptions.fminalgo==3
        goal=zeros(length(GEparamsvec0),1);
        weight=ones(length(GEparamsvec0),1); % I already implement weights via heteroagentoptions
        [p_eqm_vec,GeneralEqmConditionsVec] = fgoalattain(GeneralEqmConditionsFnOpt,GEparamsvec0,goal,weight);
        GeneralEqmConditions=sum(abs(GeneralEqmConditionsVec));
    elseif heteroagentoptions.fminalgo==4 % CMA-ES algorithm (Covariance-Matrix adaptation - Evolutionary Stategy)
        % https://en.wikipedia.org/wiki/CMA-ES
        % https://cma-es.github.io/
        % Code is cmaes.m from: https://cma-es.github.io/cmaes_sourcecode_page.html#matlab
        if ~isfield(heteroagentoptions,'insigma')
            % insigma: initial coordinate wise standard deviation(s)
            heteroagentoptions.insigma=0.3*abs(GEparamsvec0)+0.1*(GEparamsvec0==0); % Set standard deviation to 30% of the initial parameter value itself (cannot input zero, so add 0.1 to any zeros)
        end
        if ~isfield(heteroagentoptions,'inopts')
            % inopts: options struct, see defopts below
            heteroagentoptions.inopts=[];
        end
        if isfield(heteroagentoptions,'toleranceGEcondns')
            heteroagentoptions.inopts.StopFitness=heteroagentoptions.toleranceGEcondns;
        end
        % varargin (unused): arguments passed to objective function
        if heteroagentoptions.verbose==1
            disp('VFI Toolkit is using the CMA-ES algorithm, consider giving a cite to: Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution Strategy on Multimodal Test Functions' )
        end
    	% This is a minor edit of cmaes, because I want to use 'GeneralEqmConditionsFnOpt' as a function_handle, but the original cmaes code only allows for 'GeneralEqmConditionsFnOpt' as a string
        [p_eqm_vec,GeneralEqmConditions,counteval,stopflag,out,bestever] = cmaes_vfitoolkit(GeneralEqmConditionsFnOpt,GEparamsvec0,heteroagentoptions.insigma,heteroagentoptions.inopts); % ,varargin);
    elseif heteroagentoptions.fminalgo==5
        % Update based on rules in heteroagentoptions.fminalgo5.howtoupdate
        % Get initial prices, p
        p=nan(1,length(GEPriceParamNames));
        for ii=1:length(GEPriceParamNames)
            p(ii)=Parameters.(GEPriceParamNames{ii});
        end
        % Given current prices solve the model to get the general equilibrium conditions as a structure
        itercounter=0;
        p_change=Inf;
        GeneralEqmConditions=Inf;
        while (any(p_change>heteroagentoptions.toleranceGEprices) || GeneralEqmConditions>heteroagentoptions.toleranceGEcondns) && itercounter<heteroagentoptions.maxiter

            p_i=GeneralEqmConditionsFnOpt(p); % using heteroagentoptions.outputGEform=1, so this is a vector

            GeneralEqmConditionsVec=p_i; % Need later to look at convergence

            % Update prices based on GEstruct following the howtoupdate rules
            p_i=p_i(heteroagentoptions.fminalgo5.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
            I_makescutoff=(abs(p_i)>heteroagentoptions.updateaccuracycutoff);
            p_i=I_makescutoff.*p_i;
            p_new=(p.*heteroagentoptions.fminalgo5.keepold)+heteroagentoptions.fminalgo5.add.*heteroagentoptions.fminalgo5.factor.*p_i-(1-heteroagentoptions.fminalgo5.add).*heteroagentoptions.fminalgo5.factor.*p_i;

            % Calculate GeneralEqmConditions which measures convergence
            if heteroagentoptions.multiGEcriterion==0
                GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
            elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market
                GeneralEqmConditions=sqrt(sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2)));
            end

            % Put new prices into Parameters
            for ii=1:length(GEPriceParamNames)
                Parameters.(GEPriceParamNames{ii})=p_new(ii);
            end

            p_change=abs(p_new-p); % note, this is a vector
            % p_percentchange=max(abs(p_new-p)./abs(p));
            % p_percentchange(p==0)=abs(p_new(p==0)); %-p(p==0)); but this is just zero anyway
            % Update p for next iteration
            p=p_new;
            itercounter=itercounter+1; % increment iteration counter
        end
        p_eqm_vec=p_new; % Need to put it in p_eqm_vec so that it can be used to create the final output
    elseif heteroagentoptions.fminalgo==6
        if ~isfield(heteroagentoptions,'lb') || ~isfield(heteroagentoptions,'ub')
            error('When using constrained optimization (heteroagentoptions.fminalgo=6) you must set the lower and upper bounds of the GE price parameters using heteroagentoptions.lb and heteroagentoptions.ub')
        end
        [p_eqm_vec,GeneralEqmConditions]=fmincon(GeneralEqmConditionsFnOpt,GEparamsvec0,[],[],[],[],heteroagentoptions.lb,heteroagentoptions.ub,[],minoptions);
    elseif heteroagentoptions.fminalgo==7 % Matlab fsolve()
        heteroagentoptions.multiGEcriterion=0;
        [p_eqm_vec,GeneralEqmConditions]=fsolve(GeneralEqmConditionsFnOpt,GEparamsvec0,minoptions);
    elseif heteroagentoptions.fminalgo==8 % Matlab lsqnonlin()
        minoptions = optimoptions('lsqnonlin','FiniteDifferenceStepSize',1e-2,'TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns,'MaxFunEvals',heteroagentoptions.maxiter,'MaxIter',heteroagentoptions.maxiter);
        [p_eqm_vec,GeneralEqmConditions]=lsqnonlin(GeneralEqmConditionsFnOpt,GEparamsvec0,[],[],[],[],[],[],[],minoptions);
    end
    
    p_eqm_vec_untranformed=p_eqm_vec; % Use to get GE eqn values as structure/vector

    % Do any transformations of parameters before we say what they are
    for pp=1:nGEprices
        if heteroagentoptions.constrainpositive(pp)==1 % Forcing this parameter to be positive
            % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
            GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=exp(GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2)));
        elseif heteroagentoptions.constrain0to1(pp)==1
            temp=GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
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
    

    for pp=1:nGEprices
        if GEprice_ptype(pp)==0
            p_eqm.(GEPriceParamNames{pp})=p_eqm_vec(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
        else
            if heteroagentoptions.GEptype_vectoroutput==1
                p_eqm.(GEPriceParamNames{pp})=p_eqm_vec(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
            elseif heteroagentoptions.GEptype_vectoroutput==0
                temp=p_eqm_vec(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
                for ii=1:N_i
                    p_eqm.(GEPriceParamNames{pp}).(Names_i{ii})=temp(ii);
                end
            end
        end
    end

%%
elseif heteroagentoptions.maxiter==0 % Can use heteroagentoptions.maxiter=0 to just evaluate the current general eqm eqns
    % Just use the prices that are currently in Params
    p_eqm_vec_untranformed=zeros(length(GEparamsvec0),1);
    p_eqm=nan; % So user cannot misuse
    p_eqm_index=nan; % In case user asks for it
    for ii=1:length(GEPriceParamNames)
        p_eqm_vec_untranformed(ii)=Parameters.(GEPriceParamNames{ii});
    end

end



%% Add an evaluation of the general eqm eqns so that these can be output as a vectors/structure rather than just the sum of squares
if heteroagentoptions.outputGEstruct==1
    heteroagentoptions.outputGEform=2; % output as struct
elseif heteroagentoptions.outputGEstruct==2
    heteroagentoptions.outputGEform=1; % output as vector
end

if heteroagentoptions.outputGEstruct==1 || heteroagentoptions.outputGEstruct==2
    % Run once more to get the general eqm eqns in a nice form for output
    GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_PType_subfn(p, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions);
    GeneralEqmConditions=GeneralEqmConditionsFnOpt(p_eqm_vec_untranformed);
    % structure is much easier to read if it is on cpu
    if heteroagentoptions.outputGEstruct==1
        GENames=fieldnames(GeneralEqmEqns);
        for gg=1:length(GENames)
            GeneralEqmConditions.(GENames{gg})=gather(GeneralEqmConditions.(GENames{gg}));
        end
    end
end

if nargout==2
    varargout={p_eqm,GeneralEqmConditions};
elseif nargout==3
    varargout={p_eqm,p_eqm_index,GeneralEqmConditions};
end

end
