function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_PType(n_d, n_a, n_z, N_j, Names_i, n_p, pi_z, d_grid, a_grid, z_grid,jequaloneDist, Phi_aprime, Case2_Type, ReturnFn, FnsToEvaluateFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, AgeWeightParamNames, PTypeDistNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)
% Inputting vfoptions and simoptions is optional (they are not required inputs)
%
% Allows for different permanent (fixed) types of agent. 
% See ValueFnIter_PType for general idea.
%
% Rest of this description describes how those inputs not used for
% ValueFnIter_PType should be set up.
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
N_p=prod(n_p);
l_p=length(n_p);

if exist('heteroagentoptions','var')==0
    heteroagentoptions.multiGEcritereon=1;
    heteroagentoptions.multiGEweights=ones(1,length(GeneralEqmEqns));
    heteroagentoptions.toleranceGEprices=10^(-4); % Accuracy of general eqm prices
    heteroagentoptions.toleranceGEcondns=10^(-4); % Accuracy of general eqm eqns
    heteroagentoptions.verbose=0;
    heteroagentoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    heteroagentoptions.fminalgo=1; % use fminsearch
else
    if isfield(heteroagentoptions,'multiGEcriterion')==0
        heteroagentoptions.multiGEcriterion=1;
    end
    if isfield(heteroagentoptions,'multiGEweights')==0
        heteroagentoptions.multiGEweights=ones(1,length(GeneralEqmEqns));
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

    % Create all the things specific for each Permanent type and store them all in PTypeStructure.
    if ii<10 % one digit
        iistr=['ptype00',num2str(ii)];
    elseif ii<100 % two digit
        iistr=['ptype0',num2str(ii)];
    elseif ii<1000 % three digit
        iistr=['ptype',num2str(ii)];
    end
    PTypeStructure.iistr{ii}=iistr;
    
    if exist('vfoptions','var')
        PTypeStructure.(iistr).vfoptions=PType_Options(vfoptions,Names_i,ii);
    end
    if vfoptions.verbose==1
        sprintf('Permanent type: %i of %i',ii, PTypeStructure.N_i)
    end
     
    if exist('simoptions','var')
        PTypeStructure.(iistr).simoptions=PType_Options(simoptions,Names_i,ii);
    end
    if simoptions.verbose==1
        sprintf('Permanent type: %i of %i',ii, PTypeStructure.N_i)
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
    PTypeStructure.(iistr).finitehorz=0;
    if isstruct(N_j)
        if isfield(N_j, Names_i{ii})
            if isfinite(N_j.(Names_i{ii}))
                PTypeStructure.(iistr).finitehorz=1;
                PTypeStructure.(iistr).N_j=N_j.(Names_i{ii});
                % else
                % % do nothing: PTypeStructure.(iistr).finitehorz=0
            end
            % else
                % % do nothing: PTypeStructure.(iistr).finitehorz=0
        end
    elseif ~isempty(N_j)
        if isfinite(N_j(ii))
            PTypeStructure.(iistr).finitehorz=1;
            PTypeStructure.(iistr).N_j=N_j(ii);
%         else
%             % do nothing: PTypeStructure.(iistr).finitehorz=0
        end
    % else % in situtation of isempty(N_j)
        % do nothing: PTypeStructure.(iistr).finitehorz=0
    end
    
    % Case 1 or Case 2 is determined via Phi_aprime
    if exist('Phi_aprime','var') % If all the Permanent Types are 'Case 1' then there will be no Phi_aprime
        if isa(Phi_aprime,'struct')
            if isfield(Phi_aprime,Names_i{ii})==1 % Check if it exists for the current permanent type
                %         names=fieldnames(Phi_aprime);
                PTypeStructure.(iistr).Case1orCase2=2;
                PTypeStructure.(iistr).Case2_Type=Case2_Type.(Names_i{ii});
                PTypeStructure.(iistr).Phi_aprime=Phi_aprime.(Names_i{ii});
            else
                PTypeStructure.(iistr).Case1orCase2=1;
                PTypeStructure.(iistr).Case2_Type=Case2_Type;
                PTypeStructure.(iistr).Phi_aprime=Phi_aprime;
            end
        elseif isempty(Phi_aprime)
            PTypeStructure.(iistr).Case1orCase2=1;
        else
            % if Phi_aprime is not a structure then it must be relevant for all permanent types
            PTypeStructure.(iistr).Case1orCase2=2;
            PTypeStructure.(iistr).Case2_Type=Case2_Type;
            PTypeStructure.(iistr).Phi_aprime=Phi_aprime;
        end
    else
        PTypeStructure.(iistr).Case1orCase2=1;
    end
    
    % Now that we have PTypeStructure.(iistr).finitehorz and PTypeStructure.(iistr).Case1orCase2, do everything else for the current permanent type.

    PTypeStructure.(iistr).n_d=n_d;
    if isa(n_d,'struct')
        PTypeStructure.(iistr).n_d=n_d.(Names_i{ii});
    else
        temp=size(n_d);
        if temp(1)>1 % n_d depends on fixed type
            PTypeStructure.(iistr).n_d=n_d(ii,:);
        elseif temp(2)==PTypeStructure.N_i % If there is one row, but number of elements in n_d happens to coincide with number of permanent types, then just let user know
            sprintf('Possible Warning: Number of columns of n_d is the same as the number of permanent types. \n This may just be coincidence as number of d variables is equal to number of permanent types. \n If they are intended to be permanent types then n_d should have them as different rows (not columns). \n')
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
            sprintf('Possible Warning: Number of columns of n_a is the same as the number of permanent types. \n This may just be coincidence as number of a variables is equal to number of permanent types. \n If they are intended to be permanent types then n_a should have them as different rows (not columns). \n')
            dbstack
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
            sprintf('Possible Warning: Number of columns of n_z is the same as the number of permanent types. \n This may just be coincidence as number of z variables is equal to number of permanent types. \n If they are intended to be permanent types then n_z should have them as different rows (not columns). \n')
            dbstack
        end
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

    if isa(pi_z,'struct')
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
                sprintf('Possible Warning: some parameters appear to have been imputted with dependence on permanent type indexed by column rather than row \n')
                sprintf(['Specifically, parameter: ', FullParamNames{kField}, ' \n'])
                sprintf('(it is possible this is just a coincidence of number of columns) \n')
                dbstack
            end
        end
    end
    % THIS TREATMENT OF PARAMETERS COULD BE IMPROVED TO BETTER DETECT INPUT SHAPE ERRORS.
    
    if PTypeStructure.(iistr).finitehorz==1
        PTypeStructure.(iistr).jequaloneDist=jequaloneDist;
        if isa(jequaloneDist,'struct')
            if isfield(jequaloneDist,Names_i{ii})
                PTypeStructure.(iistr).jequaloneDist=jequaloneDist.(Names_i{ii});
            else
                if isfinite(PTypeStructure.(iistr).N_j)
                    sprintf(['ERROR: You must input jequaloneDist for permanent type ', Names_i{ii}, ' \n'])
                    dbstack
                end
            end
        end
    end
    
    % The parameter names can be made to depend on the permanent-type
    PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames.(Names_i{ii});
    end
    PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNames;
    if isa(ReturnFnParamNames,'struct')
        PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNames.(Names_i{ii});
    end
    if PTypeStructure.(iistr).Case1orCase2==2
        PTypeStructure.(iistr).PhiaprimeParamNames=PhiaprimeParamNames;
        if isa(PhiaprimeParamNames,'struct')
            PTypeStructure.(iistr).PhiaprimeParamNames=PhiaprimeParamNames.(Names_i{ii});
        end
    end
    PTypeStructure.(iistr).AgeWeightParamNames=AgeWeightParamNames;
    if isa(AgeWeightParamNames,'struct')
        if isfield(AgeWeightParamNames,Names_i{ii})
            PTypeStructure.(iistr).AgeWeightParamNames=AgeWeightParamNames.(Names_i{ii});
        else
            if isfinite(PTypeStructure.(iistr).N_j)
                sprintf(['ERROR: You must input AgeWeightParamNames for permanent type ', Names_i{ii}, ' \n'])
                dbstack
            end
        end
    end
    
    
    % AgeDependentGridParamNames will be inputted in place of pi_z, so deal
    % with this when dealing with pi_z
%     % Now that we have figured out if we are using agedependentgrids
%     % and stored this in PTypeStructure.(iistr).vfoptions we can use this to figure out if
%     % we need PTypeStructure.(iistr).AgeDependentGridParamNames
%     if isfield(PTypeStructure.(iistr).vfoptions,'agedependentgrids')
%         if isa(AgeDependentGridParamNames.d_grid,'struct')
%             PTypeStructure.(iistr).AgeDependentGridParamNames.d_grid=AgeDependentGridParamNames.d_grid.(Names_i{ii}); % Different grids by permanent type
%         else
%             PTypeStructure.(iistr).AgeDependentGridParamNames.d_grid=AgeDependentGridParamNames.d_grid;
%         end
%         if isa(AgeDependentGridParamNames.a_grid,'struct')
%             PTypeStructure.(iistr).AgeDependentGridParamNames.a_grid=AgeDependentGridParamNames.a_grid.(Names_i{ii}); % Different grids by permanent type
%         else
%             PTypeStructure.(iistr).AgeDependentGridParamNames.a_grid=AgeDependentGridParamNames.a_grid;
%         end
%         if isa(AgeDependentGridParamNames.z_grid,'struct')
%             PTypeStructure.(iistr).AgeDependentGridParamNames.z_grid=AgeDependentGridParamNames.z_grid.(Names_i{ii}); % Different grids by permanent type
%         else
%             PTypeStructure.(iistr).AgeDependentGridParamNames.z_grid=AgeDependentGridParamNames.z_grid;
%         end
%     end
    
    % Figure out which functions are actually relevant to the present
    % PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are
    % necessarily the same.
    PTypeStructure.(iistr).FnsToEvaluateFn={};
    PTypeStructure.(iistr).FnsToEvaluateParamNames=struct(); %(1).Names={}; % This is just an initialization value and will be overwritten
    PTypeStructure.(iistr).numFnsToEvaluate=length(FnsToEvaluateFn);
    PTypeStructure.(iistr).WhichFnsForCurrentPType=zeros(PTypeStructure.(iistr).numFnsToEvaluate,1);
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for kk=1:PTypeStructure.(iistr).numFnsToEvaluate
        if isa(FnsToEvaluateFn{kk},'struct')
            if isfield(FnsToEvaluateFn{kk}, Names_i{ii})
                PTypeStructure.(iistr).FnsToEvaluateFn{jj}=FnsToEvaluateFn{kk}.(Names_i{ii});
                if isa(FnsToEvaluateParamNames(kk).Names,'struct')
                    PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names=FnsToEvaluateParamNames(kk).Names.(Names_i{ii});
                else
                    PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names=FnsToEvaluateParamNames(kk).Names;
                end
                PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
                % else
                %  % do nothing as this FnToEvaluate is not relevant for the current PType
                % % Implicitly, PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=0
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            PTypeStructure.(iistr).FnsToEvaluateFn{jj}=FnsToEvaluateFn{kk};
            PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names=FnsToEvaluateParamNames(kk).Names;
            PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
        end
    end
    
    
%     PTypeStructure.(iistr).PTypeWeight=nan;
    if isa(PTypeDistNames, 'array')
        PTypeStructure.(iistr).PTypeWeight=PTypeDistNames(ii);
    else
%         PTypeStructure.(iistr).PTypeWeight;
%         PTypeStructure.(iistr).Parameters
%         iistr
%         PTypeDistNames{1}
%         PTypeStructure.(iistr).Parameters.(PTypeDistNames{1}); % {1} as I simply assume there is only a single parameter (name) that contains the distribution (weights) of the PTypes.
%         PTypeStructure.(iistr).Parameters.(PTypeDistNames{1})
%         PTypeStructure.(iistr).Parameters.(PTypeDistNames{1}).(Names_i{ii});
        PTypeStructure.(iistr).PTypeWeight=PTypeStructure.(iistr).Parameters.(PTypeDistNames{1}); % Don't need '.(Names_i{ii}' as this was already done when putting it into PTypeStrucutre, and here I take it straing from PTypeStructure.(iistr).Parameters rather than from Parameters itself.
    end
end

%%
% Have now finished creating PTypeStructure. Time to do the actual finding the HeteroAgentStationaryEqm:
if heteroagentoptions.verbose==1
    for ii=1:PTypeStructure.N_i
        % Create all the things specific for each Permanent type and store them all in PTypeStructure.
        if ii<10 % one digit
            iistr=['ptype00',num2str(ii)];
        elseif ii<100 % two digit
            iistr=['ptype0',num2str(ii)];
        elseif ii<1000 % three digit
            iistr=['ptype',num2str(ii)];
        end
        PTypeStructure.(iistr)
    end
end


%%
if N_p~=0
    [p_eqm_vec,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_PType_pgrid(n_p, PTypeStructure, Parameters, GeneralEqmEqns, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions);
    for ii=1:length(GEPriceParamNames)
        p_eqm.(GEPriceParamNames{ii})=p_eqm_vec;
    end
    return
end

%%  Otherwise, use fminsearch to find the general equilibrium

GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_PType_subfn(p, PTypeStructure, Parameters, GeneralEqmEqns, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions)

p0=nan(length(GEPriceParamNames),1);
for ii=1:length(GEPriceParamNames)
    p0(ii)=Parameters.(GEPriceParamNames{ii});
end

% Choosing algorithm for the optimization problem
% https://au.mathworks.com/help/optim/ug/choosing-the-algorithm.html#bscj42s
if heteroagentoptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
    heteroagentoptions.multiGEcriterion=0;
    [p_eqm_vec,GeneralEqmConditions]=fzero(GeneralEqmConditionsFnOpt,p0);    
elseif heteroagentoptions.fminalgo==1
    [p_eqm_vec,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFnOpt,p0);
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
end

p_eqm_index=nan; % If not using p_grid then this is irrelevant/useless

for ii=1:length(GEPriceParamNames)
    p_eqm.(GEPriceParamNames{ii})=p_eqm_vec(ii);
end

end
