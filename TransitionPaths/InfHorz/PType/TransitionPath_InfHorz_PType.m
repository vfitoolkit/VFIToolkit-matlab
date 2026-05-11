function varargout=TransitionPath_InfHorz_PType(PricePath0, ParamPath, T, V_final, StationaryDist_init, n_d,n_a,n_z, Names_i, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions)
% PricePathOld is a structure with fields names being the Prices and each field containing a T-by-1 path.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

% PricePathOld, ParamPath, T, V_final, StationaryDist_init, GeneralEqmEqns, GeneralEqmEqnParamNames

%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-4);
    transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiterations=500; % Based on personal experience anything that hasn't converged well before this is just hung-up on trying to get the 4th decimal place (typically because the number of grid points was not large enough to allow this level of accuracy).
    transpathoptions.verbose=0;
    transpathoptions.verbosegraphs=0;
    transpathoptions.graphpricepath=0; % 1: creates a graph of the 'current' price path which updates each iteration.
    transpathoptions.graphaggvarspath=0; % 1: creates a graph of the 'current' aggregate variables which updates each iteration.
    transpathoptions.graphGEcondns=0;  % 1: creates a graph of the 'current' general eqm conditions which updates each iteration.
    transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately);
                                   % 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up    transpathoptions.historyofpricepath=0;
    transpathoptions.usestockvars=0;
    transpathoptions.fastOLG=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'tolerance')
        transpathoptions.tolerance=10^(-4);
    end
    if ~isfield(transpathoptions,'parallel')
        transpathoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(transpathoptions,'oldpathweight')
        transpathoptions.oldpathweight=0.9;
    end
    if ~isfield(transpathoptions,'weightscheme')
        transpathoptions.weightscheme=1;
    end
    if ~isfield(transpathoptions,'Ttheta')
        transpathoptions.Ttheta=1;
    end
    if ~isfield(transpathoptions,'maxiterations')
        transpathoptions.maxiterations=500;
    end
    if ~isfield(transpathoptions,'verbose')
        transpathoptions.verbose=0;
    end
    if ~isfield(transpathoptions,'verbosegraphs')
        transpathoptions.verbosegraphs=0;
    end
    if ~isfield(transpathoptions,'graphpricepath')
        transpathoptions.graphpricepath=0; % 1: creates a graph of the 'current' price path which updates each iteration.
    end
    if ~isfield(transpathoptions,'graphaggvarspath')
        transpathoptions.graphaggvarspath=0; % 1: creates a graph of the 'current' aggregate variables which updates each iteration.
    end
    if ~isfield(transpathoptions,'graphGEcondns')
        transpathoptions.graphGEcondns=0;  % 1: creates a graph of the 'current' general eqm conditions which updates each iteration.
    end
    if ~isfield(transpathoptions,'GEnewprice')
        transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately);
                                   % 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up    
    end
    if ~isfield(transpathoptions,'fastOLG')
        transpathoptions.fastOLG=0;
    end
    if ~isfield(transpathoptions,'historyofpricepath')
        transpathoptions.historyofpricepath=0;
    end
    if ~isfield(transpathoptions,'usestockvars') % usestockvars is solely for internal use, the user does not need to set it
        if ~isfield(transpathoptions,'stockvarinit') && ~isfield(transpathoptions,'usestockvars') && ~isfield(transpathoptions,'usestockvars')
            transpathoptions.usestockvars=0;
        else
            transpathoptions.usestockvars=1; % If usestockvars has not itself been declared, but at least one of the stock variable options has then set usestockvars to 1.
        end
    end
    if transpathoptions.usestockvars==1 % Note: If this is not inputted then it is created by the above lines.
        if ~isfield(transpathoptions,'stockvarinit')
            error('transpathoptions includes some Stock Variable options but is missing stockvarinit \n')
        elseif ~isfield(transpathoptions,'stockvarpath0')
            error('transpathoptions includes some Stock Variable options but is missing stockvarpath0 \n')
        elseif ~isfield(transpathoptions,'stockvareqns')
            error('transpathoptions includes some Stock Variable options but is missing stockvareqns \n')
        end
    end
end


%% Some internal commands require a few vfoptions and simoptions to be set
if exist('vfoptions','var')==0
    vfoptions.exoticpreferences='none';
    vfoptions.divideandconquer=0;
    vfoptions.lowmemory=0;
else
    if ~isfield(vfoptions,'exoticpreferences')
        vfoptions.exoticpreferences='none';
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0;
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
end
if vfoptions.divideandconquer==1
    if isscalar(n_a)
        vfoptions.level1n=11;
    end
end

%% Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
[PricePath0,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec,PricePathSizeVec_ii,ParamPathSizeVec_ii]=PricePathParamPath_StructToMatrix(PricePath0,ParamPath,T, N_i);


%% Create PTypeStructure

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

PTypeStructure.FnsAndPTypeIndicator=zeros(length(FnsToEvaluate),PTypeStructure.N_i,'gpuArray');

if transpathoptions.verbose==1
    fprintf('Setting up the permanent types for transition \n')
end

for ii=1:PTypeStructure.N_i

    iistr=PTypeStructure.Names_i{ii};
    PTypeStructure.iistr{ii}=iistr;
    
    if exist('vfoptions','var') % vfoptions.verbose (allowed to depend on permanent type)
        PTypeStructure.(iistr).vfoptions=PType_Options(vfoptions,Names_i,ii); % some vfoptions will differ by permanent type, will clean these up as we go before they are passed
    end
    
    if exist('simoptions','var') % simoptions.verbose (allowed to depend on permanent type)
        PTypeStructure.(iistr).simoptions=PType_Options(simoptions,Names_i,ii); % some vfoptions will differ by permanent type, will clean these up as we go before they are passed
    end

    % if isfield(PTypeStructure.(iistr).vfoptions,'verbose')
    %     if PTypeStructure.(iistr).vfoptions.verbose==1
    %         sprintf('Permanent type: %i of %i',ii, PTypeStructure.N_i)
    %     elseif isfield(PTypeStructure.(iistr).simoptions,'verbose')
    %         if PTypeStructure.(iistr).simoptions.verbose==1
    %             sprintf('Permanent type: %i of %i',ii, PTypeStructure.N_i)
    %         end
    %     end
    % elseif isfield(PTypeStructure.(iistr).simoptions,'verbose')
    %     if PTypeStructure.(iistr).simoptions.verbose==1
    %         sprintf('Permanent type: %i of %i',ii, PTypeStructure.N_i)
    %     end
    % end
    % Need to fill in some defaults
    if ~isfield(PTypeStructure.(iistr).vfoptions,'parallel')
        PTypeStructure.(iistr).vfoptions.parallel=transpathoptions.parallel;
    end
    if ~isfield(PTypeStructure.(iistr).simoptions,'parallel')
        PTypeStructure.(iistr).simoptions.parallel=transpathoptions.parallel;
    end
    if ~isfield(PTypeStructure.(iistr).simoptions,'iterate')
        PTypeStructure.(iistr).simoptions.iterate=1;
    end

    % Go through everything which might be dependent on permanent type (PType)
    % Notice that the way this is coded the grids (etc.) could be either
    % fixed, or a function (that depends on age, and possibly on permanent
    % type), or they could be a structure. Only in the case where they are
    % a structure is there a need to take just a specific part and send
    % only that to the 'non-PType' version of the command.
    
    PTypeStructure.(iistr).n_d=n_d;
    if isa(n_d,'struct')
        PTypeStructure.(iistr).n_d=n_d.(Names_i{ii});
    end
    PTypeStructure.(iistr).N_d=prod(PTypeStructure.(iistr).n_d);
    PTypeStructure.(iistr).n_a=n_a;
    if isa(n_a,'struct')
        PTypeStructure.(iistr).n_a=n_a.(Names_i{ii});
    end
    PTypeStructure.(iistr).N_a=prod(PTypeStructure.(iistr).n_a);
    PTypeStructure.(iistr).n_z=n_z;
    if isa(n_z,'struct')
        PTypeStructure.(iistr).n_z=n_z.(Names_i{ii});
    end
    PTypeStructure.(iistr).N_z=prod(PTypeStructure.(iistr).n_z);
    
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
                PTypeStructure.(iistr).Parameters.(FullParamNames{kField})=temp(:,ii)';
            end
        end
    end
    % THIS TREATMENT OF PARAMETERS COULD BE IMPROVED TO BETTER DETECT INPUT SHAPE ERRORS.
    
    
    % The parameter names can be made to depend on the permanent-type
    PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames.(Names_i{ii});
    end

    % Implement new way of handling ReturnFn inputs (note l_d, l_a, l_z are just created for this and then not used for anything else later)
    if PTypeStructure.(iistr).n_d(1)==0
        l_d=0;
    else
        l_d=length(PTypeStructure.(iistr).n_d);
    end
    l_a=length(PTypeStructure.(iistr).n_a);
    l_z=length(PTypeStructure.(iistr).n_z);
    if PTypeStructure.(iistr).n_z(1)==0
        l_z=0;
    end
    if isfield(PTypeStructure.(iistr).vfoptions,'SemiExoStateFn')
        l_z=l_z+length(PTypeStructure.(iistr).vfoptions.n_semiz);
    end
    l_e=0;
    if isfield(PTypeStructure.(iistr).vfoptions,'n_e')
        if PTypeStructure.(iistr).vfoptions.n_e(1)~=0
            l_e=length(PTypeStructure.(iistr).vfoptions.n_e);
            using_e_var=1;
        else
            using_e_var=0;
        end
    else
        using_e_var=0;
    end
    % Figure out ReturnFnParamNames from ReturnFn
    temp=getAnonymousFnInputNames(PTypeStructure.(iistr).ReturnFn);
    if length(temp)>(l_d+l_a+l_a+l_z+l_e) % This is largely pointless, the ReturnFn is always going to have some parameters
        ReturnFnParamNames={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        ReturnFnParamNames={};
    end
    PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNames;

    
    % Figure out which functions are actually relevant to the present
    % PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are
    % necessarily the same.
    PTypeStructure.(iistr).FnsToEvaluate={};


    FnNames=fieldnames(FnsToEvaluate);
    PTypeStructure.numFnsToEvaluate=length(fieldnames(FnsToEvaluate));
    PTypeStructure.(iistr).WhichFnsForCurrentPType=zeros(PTypeStructure.numFnsToEvaluate,1);
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for kk=1:PTypeStructure.numFnsToEvaluate
        if isa(FnsToEvaluate.(FnNames{kk}),'struct')
            if isfield(FnsToEvaluate.(FnNames{kk}), Names_i{ii})
                PTypeStructure.(iistr).FnsToEvaluate{jj}=FnsToEvaluate.(FnNames{kk}).(Names_i{ii});
                % Figure out FnsToEvaluateParamNames
                temp=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{kk}).(Names_i{ii}));
                PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
                PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
                PTypeStructure.FnsAndPTypeIndicator(kk,ii)=1;
                % else
                %  % do nothing as this FnToEvaluate is not relevant for the current PType
                % % Implicitly, PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=0
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            PTypeStructure.(iistr).FnsToEvaluate{jj}=FnsToEvaluate.(FnNames{kk});
            % Figure out FnsToEvaluateParamNames
            temp=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{kk}));
            PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
            PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
            PTypeStructure.FnsAndPTypeIndicator(kk,ii)=1;
        end
    end



    %% Check if pi_z and z_grid can be precomputed
    % Note: cannot handle that whether not not they can be precomputed differs across ptypes
    transpathoptions.zpathprecomputed=1;
    transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
    if isfield(PTypeStructure.(iistr).vfoptions,'ExogShockFn')
        transpathoptions.zpathprecomputed=0;
        N_z=prod(PTypeStructure.(iistr).n_z);
        % Note: If ExogShockFn depends on the path, it must be done via a parameter
        % that depends on the path (i.e., via ParamPath or PricePath)
        PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames=getAnonymousFnInputNames(PTypeStructure.(iistr).vfoptions.ExogShockFn);
        overlap=0;
        for pp=1:length(PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames)
            if strcmp(PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames{pp},PricePathNames)
                overlap=1;
            end
        end
        if overlap==0
            transpathoptions.zpathprecomputed=1;
            % If ExogShockFn does not depend on any of the prices (in PricePath), then
            % we can simply create it now rather than within each 'subfn' or 'p_grid'

            % Check if it depends on the ParamPath
            transpathoptions.zpathtrivial=1;
            for pp=1:length(PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames)
                if strcmp(PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames{pp},ParamPathNames)
                    transpathoptions.zpathtrivial=0;
                end
            end
            if transpathoptions.zpathtrivial==1
                ExogShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for pp=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                end
                [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            elseif transpathoptions.zpathtrivial==0
                % z_grid and/or pi_z varies along the transition path (but only depending on ParamPath, not PricePath
                transpathoptions.(iistr).pi_z_T=zeros(N_z,N_z,T,'gpuArray');
                transpathoptions.(iistr).z_grid_T=zeros(sum(n_z),T,'gpuArray');
                for tt=1:T
                    for pp=1:length(ParamPathNames)
                        PTypeStructure.(iistr).Parameters.(ParamPathNames{pp})=ParamPathStruct.(ParamPathNames{pp});
                    end
                    % Note, we know the PricePath is irrelevant for the current purpose
                    ExogShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for pp=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                    end
                    [z_grid,pi_z]=PTypeStructure.(iistr).vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    transpathoptions.(iistr).pi_z_T(:,:,tt)=pi_z;
                    transpathoptions.(iistr).z_grid_T(:,tt)=z_grid;
                end
            end
        end
    end
    %% If used, check if pi_e and e_grid can be procomputed
    % Note: cannot handle that whether not not they can be precomputed differs across ptypes
    if using_e_var==1
        % Check if e_grid and/or pi_e depend on prices. If not then create pi_e_J and e_grid_J for the entire transition before we start
        transpathoptions.epathprecomputed=1;
        transpathoptions.epathtrivial=1; % e_grid_J and pi_e_J are not varying over the path
        if isfield(PTypeStructure.(iistr).vfoptions,'EiidShockFn')
            transpathoptions.epathprecomputed=0;
            N_e=prod(PTypeStructure.(iistr).vfoptions.n_e);
            % Note: If EiidShockFn depends on the path, it must be done via a parameter
            % that depends on the path (i.e., via ParamPath or PricePath)
            PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames=getAnonymousFnInputNames(PTypeStructure.(iistr).vfoptions.EiidShockFn);
            overlap=0;
            for pp=1:length(PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames)
                if strcmp(PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames{pp},PricePathNames)
                    overlap=1;
                end
            end
            if overlap==0
                transpathoptions.epathprecomputed=1;
                % If ExogShockFn does not depend on any of the prices (in PricePath), then
                % we can simply create it now rather than within each 'subfn' or 'p_grid'

                % Check if it depends on the ParamPath
                transpathoptions.epathtrivial=1;
                for pp=1:length(PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames)
                    if strcmp(PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames{pp},ParamPathNames)
                        transpathoptions.epathtrivial=0;
                    end
                end
                if transpathoptions.epathtrivial==1
                    EiidShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for pp=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                    end
                    [e_grid,pi_e]=PTypeStructure.(iistr).vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
                    % Now store them in vfoptions and simoptions
                    PTypeStructure.(iistr).vfoptions.pi_e=pi_e;
                    PTypeStructure.(iistr).vfoptions.e_grid=e_grid;
                    PTypeStructure.(iistr).simoptions.pi_e=pi_e;
                    PTypeStructure.(iistr).simoptions.e_grid=e_grid;
                elseif transpathoptions.epathtrivial==0
                    % e_grid_J and/or pi_e_J varies along the transition path (but only depending on ParamPath, not PricePath)
                    transpathoptions.(iistr).pi_e_T=zeros(N_e,T,'gpuArray');
                    transpathoptions.(iistr).e_grid_T=zeros(sum(PTypeStructure.(iistr).vfoptions.n_e),T,'gpuArray');
                    for tt=1:T
                        for pp=1:length(ParamPathNames)
                            PTypeStructure.(iistr).Parameters.(ParamPathNames{pp})=ParamPathStruct.(ParamPathNames{pp});
                        end
                        % Note, we know the PricePath is irrelevant for the current purpose
                        EiidShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, vPTypeStructure.(iistr).foptions.EiidShockFnParamNames);
                        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                        for pp=1:length(ExogShockFnParamsVec)
                            EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                        end
                        [e_grid,pi_e]=PTypeStructure.(iistr).vfoptions.ExogShockFn(EiidShockFnParamsCell{:});

                        transpathoptions.(iistr).pi_e_T(:,tt)=pi_e;
                        transpathoptions.(iistr).e_grid_T(:,tt)=e_grid;
                    end
                end
            end
        end
    end
   
end


%%
if transpathoptions.parallel==2 
   PricePathOld=gpuArray(PricePathOld);
end

% GeneralEqmEqnNames=fieldnames(GeneralEqmEqns);
% for gg=1:length(GeneralEqmEqnNames)
%     GeneralEqmEqnParamNames{gg}=getAnonymousFnInputNames(GeneralEqmEqns.(GeneralEqmEqnNames{gg}));
% end

%%
if transpathoptions.usestockvars==1 
    error('transpathoptions.usestockvars=1 not yet implemented with PType \n')
end


%% If using a shooting algorithm, set that up
transpathoptions=setupGEnewprice3_shooting(options,GeneralEqmEqns,PricePathNames,N_i,PricePathSizeVec);
GEeqnNames=fieldnames(GeneralEqmEqns);

%%
if transpathoptions.verbose==1
    transpathoptions
    fprintf('Completed setup, beginning transition computation \n')
end

%% GE eqns, switch from structure to cell setup
GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);
GeneralEqmEqnsCell=cell(1,nGeneralEqmEqns);
for gg=1:nGeneralEqmEqns
    temp=getAnonymousFnInputNames(GeneralEqmEqns.(GEeqnNames{gg}));
    GeneralEqmEqnParamNames(gg).Names=temp;
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns.(GEeqnNames{gg});
end

%%
if transpathoptions.GEnewprice~=2
    if transpathoptions.usestockvars==0
        if ~isfield(vfoptions,'n_e')
            [PricePath,GEcondnPathmatrix]=TransitionPath_InfHorz_PType_shooting(PricePath0, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, FnsToEvaluate, GeneralEqmEqns, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, nGeneralEqmEqns, transpathoptions, PTypeStructure);
        else
            error('Cannot use e variables with infinite horizon (contact me if you need this)')
            % PricePathOld=TransitionPath_InfHorz_PType_e_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, FnsToEvaluate, GeneralEqmEqns, transpathoptions, PTypeStructure);
        end
    end
    % Switch the solution into structure for output.
    for ii=1:length(PricePathNames)
        PricePathStruct.(PricePathNames{ii})=PricePath(:,ii);
    end
    for gg=1:length(GEeqnNames)
        GEcondnPath.(GEeqnNames{gg})=GEcondnPathmatrix(:,gg)';
    end


    if nargout==1
        varargout={PricePathStruct};
    elseif nargout==2
        varargout={PricePathStruct,GEcondnPath};
    end

    return
end

if transpathoptions.GEnewprice==2
    % NOT IMPLEMENTED


end

end
