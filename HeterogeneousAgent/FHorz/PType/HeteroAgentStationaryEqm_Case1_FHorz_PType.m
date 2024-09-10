function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_FHorz_PType(n_d, n_a, n_z, N_j, Names_i, n_p, pi_z, d_grid, a_grid, z_grid,jequaloneDist, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, PTypeDistParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)
% Inputting vfoptions and simoptions is optional (they are not required inputs)
%
% Allows for different permanent (fixed) types of agent. 
% See ValueFnIter_Case1_FHorz_PType for general idea.
%
% Rest of this description describes how those inputs not used for
% ValueFnIter_Case1_FHorz_PType should be set up.
%
% jequaloneDist can either be same for all permanent types, or must be passed as a structure.
% AgeWeightParamNames is either same for all permanent types, or must be passed as a structure.
%
%
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



%%
if ~isempty(n_p)
    N_p=prod(n_p);
else
    N_p=0;
end
l_p=length(n_p);

if exist('heteroagentoptions','var')==0
    heteroagentoptions.multiGEcriterion=1;
    heteroagentoptions.multiGEweights=ones(1,length(GeneralEqmEqns));
    heteroagentoptions.toleranceGEprices=10^(-4); % Accuracy of general eqm prices
    heteroagentoptions.toleranceGEcondns=10^(-4); % Accuracy of general eqm eqns
    heteroagentoptions.maxiter=200*length(GEPriceParamNames); % this is roughly the matlab default (for fminsearch(), or would be if all GEPriceParamNames are scalar)
    heteroagentoptions.verbose=0;
    heteroagentoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    heteroagentoptions.fminalgo=1; % use fminsearch
    heteroagentoptions.saveprogresseachiter=0;
    heteroagentoptions.GEptype=zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type'
    % heteroagentoptions.outputGEform=0; % For internal use only
    heteroagentoptions.outputGEstruct=1; % output GE conditions as a structure (=2 will output as a vector)
else
    if ~isfield(heteroagentoptions,'multiGEcriterion')
        heteroagentoptions.multiGEcriterion=1;
    end
    if ~isfield(heteroagentoptions,'multiGEweights')
        heteroagentoptions.multiGEweights=ones(1,length(GeneralEqmEqns));
    end
    if ~isfield(heteroagentoptions,'maxiter')
        heteroagentoptions.maxiter=200*length(GEPriceParamNames); % this is roughly the matlab default (for fminsearch(), or would be if all GEPriceParamNames are scalar)
    end
    if N_p~=0
        if ~isfield(heteroagentoptions,'p_grid')
            disp('ERROR: you have set n_p to a non-zero value, but not declared heteroagentoptions.pgrid')
            dbstack
        end
    end
    if ~isfield(heteroagentoptions,'toleranceGEprices')
        heteroagentoptions.toleranceGEprices=10^(-4); % Accuracy of general eqm prices
    end
    if ~isfield(heteroagentoptions,'toleranceGEcondns')
        heteroagentoptions.toleranceGEcondns=10^(-4); % Accuracy of general eqm prices
    end
    if ~isfield(heteroagentoptions,'verbose')
        heteroagentoptions.verbose=0;
    end
    if ~isfield(heteroagentoptions,'fminalgo')
        heteroagentoptions.fminalgo=1; % use fminsearch
    end
    if ~isfield(heteroagentoptions,'parallel')
        heteroagentoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if ~isfield(heteroagentoptions,'saveprogresseachiter')
        heteroagentoptions.saveprogresseachiter=0;
    end
    if ~isfield(heteroagentoptions,'GEptype')
        heteroagentoptions.GEptype=zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type'
    end
    % heteroagentoptions.outputGEform=0; % For internal use only
    if ~isfield(heteroagentoptions,'outputGEstruct')
        heteroagentoptions.outputGEstruct=1; % output GE conditions as a structure (=2 will output as a vector)
    end
end

if heteroagentoptions.fminalgo==5
    if isfield(heteroagentoptions,'toleranceGEprices_percent')==0
        heteroagentoptions.toleranceGEprices_percent=10^(-3); % one-tenth of one percent
    end
    heteroagentoptions.outputGEform=1; % Need to output GE condns as a vector when using fminalgo=5
else
    heteroagentoptions.outputGEform=0;
end

AggVarNames=fieldnames(FnsToEvaluate);
nGEprices=length(GEPriceParamNames);

PTypeStructure.numFnsToEvaluate=length(fieldnames(FnsToEvaluate)); % Total number of functions to evaluate

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

% If jequaloneDist is not a structure, then we deal with it here (as it may be dependent on PType, and we will turn it into a structure if it is)
if ~isstruct(jequaloneDist)
    [jequaloneDist,~,Parameters]=jequaloneDist_PType(jequaloneDist,Parameters,simoptions,n_a,n_z,PTypeStructure.N_i,PTypeStructure.Names_i,PTypeDistParamNames,1);
end

%%
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
    
    % Horizon is determined via N_j
    if isstruct(N_j)
        PTypeStructure.(iistr).N_j=N_j.(Names_i{ii});
    elseif isscalar(N_j)
        PTypeStructure.(iistr).N_j=N_j;
    else
        PTypeStructure.(iistr).N_j=N_j(ii);
    end
    
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
    if PTypeStructure.(iistr).n_z(1)==0
        PTypeStructure.(iistr).l_z=0;
    else
        PTypeStructure.(iistr).l_z=length(PTypeStructure.(iistr).n_z);
    end
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
    % If using 'agedependentgrids' then pi_z will actually be the AgeDependentGridParamNames, which is a structure. 
    % Following gets complicated as pi_z being a structure could be because
    % it depends just on age, or on permanent type, or on both.
    if exist('vfoptions','var')
        if isfield(vfoptions,'agedependentgrids')
            if isa(vfoptions.agedependentgrids, 'struct')
                if isfield(vfoptions.agedependentgrids, Names_i{ii})
                    PTypeStructure.(iistr).vfoptions.agedependentgrids=vfoptions.agedependentgrids.(Names_i{ii});
                    PTypeStructure.(iistr).simoptions.agedependentgrids=simoptions.agedependentgrids.(Names_i{ii});
                    % In this case AgeDependentGridParamNames must be set up as, e.g., AgeDependentGridParamNames.ptype1.d_grid
                    PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii});
                else
                    % The current permanent type does not use age dependent grids.
                    PTypeStructure.(iistr).vfoptions=rmfield(PTypeStructure.(iistr).vfoptions,'agedependentgrids');
                    PTypeStructure.(iistr).simoptions=rmfield(PTypeStructure.(iistr).simoptions,'agedependentgrids');
                    % Different grids by permanent type (some of them must be using agedependentgrids even though not the current permanent type), but not depending on age.
                    PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii});
                end
            else
                temp=size(vfoptions.agedependentgrids);
                if temp(1)>1 % So different permanent types use different settings for age dependent grids
                    if prod(temp(ii,:))>0
                        PTypeStructure.(iistr).vfoptions.agedependentgrids=vfoptions.agedependentgrids(ii,:);
                        PTypeStructure.(iistr).simoptions.agedependentgrids=simoptions.agedependentgrids(ii,:);
                    else
                        PTypeStructure.(iistr).vfoptions=rmfield(PTypeStructure.(iistr).vfoptions,'agedependentgrids');
                        PTypeStructure.(iistr).simoptions=rmfield(PTypeStructure.(iistr).simoptions,'agedependentgrids');
                    end
                    % In this case AgeDependentGridParamNames must be set up as, e.g., AgeDependentGridParamNames.ptype1.d_grid
                    PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii});
                else % Grids depend on age, but not on permanent type (at least the function does not, you could set it up so that this is handled by the same function but a parameter whose value differs by permanent type
                    PTypeStructure.(iistr).pi_z=pi_z;
                end
            end
        elseif isa(pi_z,'struct')
            PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age.
        end
    elseif isa(pi_z,'struct')
        PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age. (same as the case just above; this case can occour with or without the existence of vfoptions, as long as there is no vfoptions.agedependentgrids)
    end
    
    PTypeStructure.(iistr).ReturnFn=ReturnFn;
    if isa(ReturnFn,'struct')
        PTypeStructure.(iistr).ReturnFn=ReturnFn.(Names_i{ii});
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
    
    if isstruct(jequaloneDist)
        if isfield(jequaloneDist,PTypeStructure.Names_i{ii})
            if isa(jequaloneDist, 'function_handle')
                [PTypeStructure.(iistr).jequaloneDist,~,PTypeStructure.(iistr).Parameters]=jequaloneDist_PType(jequaloneDist.(iistr),PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).simoptions,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_i,PTypeStructure.(iistr).PTypeDistParamNames,0);
            else
                PTypeStructure.(iistr).jequaloneDist=jequaloneDist.(PTypeStructure.Names_i{ii});
            end
        else
            if isfinite(PTypeStructure.(iistr).N_j)
                sprintf(['ERROR: You must input jequaloneDist for permanent type ', PTypeStructure.Names_i{ii}, ' \n'])
                dbstack
            end
        end
    else
        PTypeStructure.(iistr).jequaloneDist=jequaloneDist;
    end
    
    % The parameter names can be made to depend on the permanent-type
    PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames.(Names_i{ii});
    end
    PTypeStructure.(iistr).AgeWeightParamNames=AgeWeightsParamNames;
    if isa(AgeWeightsParamNames,'struct')
        if isfield(AgeWeightsParamNames,Names_i{ii})
            PTypeStructure.(iistr).AgeWeightParamNames=AgeWeightsParamNames.(Names_i{ii});
        else
            if isfinite(PTypeStructure.(iistr).N_j)
                sprintf(['ERROR: You must input AgeWeightParamNames for permanent type ', Names_i{ii}, ' \n'])
                dbstack
            end
        end
    end
        
    PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNamesFn(PTypeStructure.(iistr).ReturnFn,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).Parameters);
    
    % Figure out which functions are actually relevant to the present PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluate and FnsToEvaluateFnParamNames are necessarily the same.
    % Allows for FnsToEvaluate as structure.
    l_d_temp=PTypeStructure.(iistr).l_d;
    l_a_temp=PTypeStructure.(iistr).l_a;
    l_z_temp=PTypeStructure.(iistr).l_z+PTypeStructure.(iistr).l_e;  
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


%% Have now finished creating PTypeStructure. Time to do the actual finding the HeteroAgentStationaryEqm:

%%
if N_p~=0
    error('NOTE: HeteroAgentStationaryEqm_Case1_FHorz_PType with p_grid does not yet exist so will throw an error. Contact robertdkirkby@gmail.com if you actually want to use it and I will set it up. \n')
%     [p_eqm_vec,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_Case1_FHorz_PType_pgrid(n_p, PTypeStructure, Parameters, GeneralEqmEqns, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions);
%     for ii=1:length(GEPriceParamNames)
%         p_eqm.(GEPriceParamNames{ii})=p_eqm_vec;
%     end
%     return
end


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

%% Permit that some GEPriceParamNames might depend on PType
p0=[]; % column vector
GEpriceindexes=zeros(nGEprices,1);
GEprice_ptype=zeros(nGEprices,1);
for pp=1:nGEprices
    if isstruct(Parameters.(GEPriceParamNames{pp}))
        for ii=1:PTypeStructure.N_i
            iistr=PTypeStructure.Names_i{ii};
            p0=[p0; gather(Parameters.(GEPriceParamNames{pp}).(iistr))]; % reshape()' is making sure it is a row vector
        end
        GEpriceindexes(pp)=PTypeStructure.N_i;
        GEprice_ptype(pp)=1;
    else
        p0=[p0;reshape(gather(Parameters.(GEPriceParamNames{pp})),[],1)]; % reshape() is making sure it is a column vector
        GEpriceindexes(pp)=length(Parameters.(GEPriceParamNames{pp}));
        if length(Parameters.(GEPriceParamNames{pp}))>1
            GEprice_ptype(pp)=1;
        end
    end
end
GEpriceindexes=[[1; 1+cumsum(GEpriceindexes(1:end-1))],cumsum(GEpriceindexes)];

%%
if heteroagentoptions.maxiter>0 % Can use heteroagentoptions.maxiter=0 to just evaluate the current general eqm eqns

    %%  Otherwise, use fminsearch to find the general equilibrium
    if all(heteroagentoptions.GEptype==0)
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_FHorz_PType_subfn(p, PTypeStructure, Parameters, GeneralEqmEqns, GEPriceParamNames,AggVarNames,nGEprices,heteroagentoptions);
    else
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_FHorz_PType_GEptype_subfn(p, PTypeStructure, Parameters, GeneralEqmEqns, GEPriceParamNames,AggVarNames,nGEprices,GEpriceindexes,GEprice_ptype,heteroagentoptions);
    end



    % Choosing algorithm for the optimization problem
    % https://au.mathworks.com/help/optim/ug/choosing-the-algorithm.html#bscj42s
    minoptions = optimset('TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns,'MaxFunEvals',heteroagentoptions.maxiter);
    if heteroagentoptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
        heteroagentoptions.multimarketcriterion=0;
        [p_eqm_vec,GeneralEqmConditions]=fzero(GeneralEqmConditionsFnOpt,p0,minoptions);
    elseif heteroagentoptions.fminalgo==1
        [p_eqm_vec,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFnOpt,p0,minoptions);
    elseif heteroagentoptions.fminalgo==2
        % Use the optimization toolbox so as to take advantage of automatic differentiation
        z=optimvar('z',length(p0));
        optimfun=fcn2optimexpr(GeneralEqmConditionsFnOpt, z);
        prob = optimproblem("Objective",optimfun);
        z0.z=p0;
        [sol,GeneralEqmConditions]=solve(prob,z0);
        p_eqm_vec=sol.z;
        % Note, doesn't really work as automattic differentiation is only for
        % supported functions, and the objective here is not a supported function
    elseif heteroagentoptions.fminalgo==3
        goal=zeros(length(p0),1);
        weight=ones(length(p0),1); % I already implement weights via heteroagentoptions
        [p_eqm_vec,GeneralEqmConditionsVec] = fgoalattain(GeneralEqmConditionsFnOpt,p0,goal,weight);
        GeneralEqmConditions=sum(abs(GeneralEqmConditionsVec));
    elseif heteroagentoptions.fminalgo==4 % CMA-ES algorithm (Covariance-Matrix adaptation - Evolutionary Stategy)
        % https://en.wikipedia.org/wiki/CMA-ES
        % https://cma-es.github.io/
        % Code is cmaes.m from: https://cma-es.github.io/cmaes_sourcecode_page.html#matlab
        if ~isfield(heteroagentoptions,'insigma')
            % insigma: initial coordinate wise standard deviation(s)
            heteroagentoptions.insigma=0.3*abs(p0)+0.1*(p0==0); % Set standard deviation to 30% of the initial parameter value itself (cannot input zero, so add 0.1 to any zeros)
        end
        if ~isfield(heteroagentoptions,'inopts')
            % inopts: options struct, see defopts below
            heteroagentoptions.inopts=[];
        end
        % varargin (unused): arguments passed to objective function
        if heteroagentoptions.verbose==1
            disp('VFI Toolkit is using the CMA-ES algorithm, consider giving a cite to: Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution Strategy on Multimodal Test Functions' )
        end
    	% This is a minor edit of cmaes, because I want to use 'GeneralEqmConditionsFnOpt' as a function_handle, but the original cmaes code only allows for 'GeneralEqmConditionsFnOpt' as a string
        [p_eqm_vec,GeneralEqmConditions,counteval,stopflag,out,bestever] = cmaes_vfitoolkit(GeneralEqmConditionsFnOpt,p0,heteroagentoptions.insigma,heteroagentoptions.inopts); % ,varargin);
    elseif heteroagentoptions.fminalgo==5
        % Update based on rules in heteroagentoptions.fminalgo5.howtoupdate
        GeneralEqmEqnsNames=fieldnames(GeneralEqmEqns);
        GeneralEqmConditions=Inf;
        % Get initial prices, p
        p=nan(1,length(GEPriceParamNames));
        for ii=1:length(GEPriceParamNames)
            p(ii)=Parameters.(GEPriceParamNames{ii});
        end
        % Given current prices solve the model to get the general equilibrium conditions as a structure
        p_percentchange=Inf;
        itercount=0;
        while any(p_percentchange>heteroagentoptions.toleranceGEprices_percent) % GeneralEqmConditions>heteroagentoptions.toleranceGEcondns

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

            fprintf('Current iteration \n')
            p_percentchange
            p_new
            p
            p_i

            p_percentchange=max(abs(p_new-p)./abs(p));
            p_percentchange(p==0)=abs(p_new(p==0)); %-p(p==0)); but this is just zero anyway
            % Update p for next iteration
            p=p_new;

            if heteroagentoptions.saveprogresseachiter==1
                itercount=itercount+1;
                save HeterAgentEqm_internal2.mat p_percentchange GeneralEqmConditionsVec itercount
            end
        end
        p_eqm_vec=p_new; % Need to put it in p_eqm_vec so that it can be used to create the final output
    elseif heteroagentoptions.fminalgo==6
        if ~isfield(heteroagentoptions,'lb') || ~isfield(heteroagentoptions,'ub')
            error('When using constrained optimization (heteroagentoptions.fminalgo=6) you must set the lower and upper bounds of the GE price parameters using heteroagentoptions.lb and heteroagentoptions.ub')
        end
        [p_eqm_vec,GeneralEqmConditions]=fmincon(GeneralEqmConditionsFnOpt,p0,[],[],[],[],heteroagentoptions.lb,heteroagentoptions.ub,[],minoptions);
    elseif heteroagentoptions.fminalgo==7 % Matlab fsolve()
        heteroagentoptions.multiGEcriterion=0;
        [p_eqm_vec,GeneralEqmConditions]=fzero(GeneralEqmConditionsFnOpt,p0,minoptions);
    end

    p_eqm_index=nan; % If not using p_grid then this is irrelevant/useless

    for pp=1:nGEprices
        p_eqm.(GEPriceParamNames{pp})=p_eqm_vec(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
    end

    % vargout={p_eqm,p_eqm_index,GeneralEqmConditions};
    % if heteroagentoptions.fminalgo==3
    %     vargout={p_eqm,GeneralEqmConditions,counteval,stopflag,out,bestever};
    % end

%%
elseif heteroagentoptions.maxiter==0 % Can use heteroagentoptions.maxiter=0 to just evaluate the current general eqm eqns
    % Just use the prices that are currently in Params
    p_eqm=nan; % So user cannot misuse
    for ii=1:length(GEPriceParamNames)
        p_eqm_vec(ii)=Parameters.(GEPriceParamNames{ii});
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
    if all(heteroagentoptions.GEptype==0)
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_FHorz_PType_subfn(p, PTypeStructure, Parameters, GeneralEqmEqns, GEPriceParamNames,AggVarNames,nGEprices,heteroagentoptions);
    else
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_Case1_FHorz_PType_GEptype_subfn(p, PTypeStructure, Parameters, GeneralEqmEqns, GEPriceParamNames,AggVarNames,nGEprices,GEpriceindexes,GEprice_ptype,heteroagentoptions);
    end
    GeneralEqmConditions=GeneralEqmConditionsFnOpt(p_eqm_vec);
end



end
