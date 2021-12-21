function PricePath=TransitionPath_Case1_FHorz_PType(PricePathOld, ParamPath, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, Names_i, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, transpathoptions, simoptions, vfoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 
%
% PricePathOld is a structure with fields names being the Prices and each field containing a T-by-1 path.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

% Note: Internally PricePathOld is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
PricePathNames=fieldnames(PricePathOld);
PricePathStruct=PricePathOld; 
PricePathOld=zeros(T,length(PricePathNames));
for ii=1:length(PricePathNames)
    PricePathOld(:,ii)=PricePathStruct.(PricePathNames{ii});
end
ParamPathNames=fieldnames(ParamPath);
ParamPathStruct=ParamPath; 
ParamPath=zeros(T,length(ParamPathNames));
for ii=1:length(ParamPathNames)
    ParamPath(:,ii)=ParamPathStruct.(ParamPathNames{ii});
end

PricePath=struct();

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
    transpathoptions.fastOLG=0;
    transpathoptions.GEnewprice=2;
    transpathoptions.historyofpricepath=0;
    transpathoptions.stockvars=0;
    transpathoptions.fastOLG=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'tolerance')==0
        transpathoptions.tolerance=10^(-4);
    end
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(transpathoptions,'oldpathweight')==0
        transpathoptions.oldpathweight=0.9;
    end
    if isfield(transpathoptions,'weightscheme')==0
        transpathoptions.weightscheme=1;
    end
    if isfield(transpathoptions,'Ttheta')==0
        transpathoptions.Ttheta=1;
    end
    if isfield(transpathoptions,'maxiterations')==0
        transpathoptions.maxiterations=500;
    end
    if isfield(transpathoptions,'verbose')==0
        transpathoptions.verbose=0;
    end
    if isfield(transpathoptions,'verbosegraphs')==0
        transpathoptions.verbosegraphs=0;
    end
    if isfield(transpathoptions,'GEnewprice')==0
        transpathoptions.GEnewprice=2;
    end
    if isfield(transpathoptions,'fastOLG')==0
        transpathoptions.fastOLG=0;
    end
    if isfield(transpathoptions,'historyofpricepath')==0
        transpathoptions.historyofpricepath=0;
    end
    if isfield(transpathoptions,'usestockvars')==0 % usestockvars is solely for internal use, the user does not need to set it
        if isfield(transpathoptions,'stockvarinit')==0 && isfield(transpathoptions,'usestockvars')==0 && isfield(transpathoptions,'usestockvars')==0
            transpathoptions.usestockvars=0;
        else
            transpathoptions.usestockvars=1; % If usestockvars has not itself been declared, but at least one of the stock variable options has then set usestockvars to 1.
        end
    end
    if transpathoptions.usestockvars==1 % Note: If this is not inputted then it is created by the above lines.
        if isfield(transpathoptions,'stockvarinit')==0
            fprintf('ERROR: transpathoptions includes some Stock Variable options but is missing stockvarinit \n')
            dbstack
            return
        elseif isfield(transpathoptions,'stockvarpath0')==0
            fprintf('ERROR: transpathoptions includes some Stock Variable options but is missing stockvarpath0 \n')
            dbstack
            return
        elseif isfield(transpathoptions,'stockvareqns')==0
            fprintf('ERROR: transpathoptions includes some Stock Variable options but is missing stockvareqns \n')
            dbstack
            return
        end
    end
end


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

for ii=1:PTypeStructure.N_i

    iistr=PTypeStructure.Names_i{ii};
    PTypeStructure.iistr{ii}=iistr;
    
    if exist('vfoptions','var') % vfoptions.verbose (allowed to depend on permanent type)
        PTypeStructure.(iistr).vfoptions=PType_Options(vfoptions,Names_i,ii); % some vfoptions will differ by permanent type, will clean these up as we go before they are passed
    end
    
    if exist('simoptions','var') % vfoptions.verbose (allowed to depend on permanent type)
        PTypeStructure.(iistr).simoptions=PType_Options(simoptions,Names_i,ii); % some vfoptions will differ by permanent type, will clean these up as we go before they are passed
    end

    if isfield(PTypeStructure.(iistr).vfoptions,'verbose')
        if PTypeStructure.(iistr).vfoptions.verbose==1
            sprintf('Permanent type: %i of %i',ii, PTypeStructure.N_i)
        elseif isfield(PTypeStructure.(iistr).simoptions,'verbose')
            if PTypeStructure.(iistr).simoptions.verbose==1
                sprintf('Permanent type: %i of %i',ii, PTypeStructure.N_i)
            end
        end
    elseif isfield(PTypeStructure.(iistr).simoptions,'verbose')
        if PTypeStructure.(iistr).simoptions.verbose==1
            sprintf('Permanent type: %i of %i',ii, PTypeStructure.N_i)
        end
    end
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
            sprintf('Possible Warning: Number of columns of n_d is the same as the number of permanent types. \n This may just be coincidence as number of d variables is equal to number of permanent types. \n If they are intended to be permanent types then n_d should have them as different rows (not columns). \n')
        end
    end
    PTypeStructure.(iistr).N_d=prod(PTypeStructure.(iistr).n_d);
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
    PTypeStructure.(iistr).N_a=prod(PTypeStructure.(iistr).n_a);
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
                sprintf('Possible Warning: some parameters appear to have been imputted with dependence on permanent type indexed by column rather than row \n')
                sprintf(['Specifically, parameter: ', FullParamNames{kField}, ' \n'])
                sprintf('(it is possible this is just a coincidence of number of columns) \n')
                dbstack
            end
        end
    end
    % THIS TREATMENT OF PARAMETERS COULD BE IMPROVED TO BETTER DETECT INPUT SHAPE ERRORS.
    
    
    % The parameter names can be made to depend on the permanent-type
    PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames.(Names_i{ii});
    end
    PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNames;
    if isa(ReturnFnParamNames,'struct')
        PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNames.(Names_i{ii});
    end
    PTypeStructure.(iistr).AgeWeightsParamNames=AgeWeightsParamNames;
    if isa(AgeWeightsParamNames,'struct')
        if isfield(AgeWeightsParamNames,Names_i{ii})
            PTypeStructure.(iistr).AgeWeightsParamNames=AgeWeightsParamNames.(Names_i{ii});
        else
            if isfinite(PTypeStructure.(iistr).N_j)
                sprintf(['ERROR: You must input AgeWeightParamNames for permanent type ', Names_i{ii}, ' \n'])
                dbstack
            end
        end
    end
    
    % Figure out which functions are actually relevant to the present
    % PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are
    % necessarily the same.
    PTypeStructure.(iistr).FnsToEvaluate={};
    PTypeStructure.(iistr).FnsToEvaluateParamNames=struct(); %(1).Names={}; % This is just an initialization value and will be overwritten
    PTypeStructure.numFnsToEvaluate=length(FnsToEvaluate);
    PTypeStructure.(iistr).WhichFnsForCurrentPType=zeros(PTypeStructure.numFnsToEvaluate,1);
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for kk=1:PTypeStructure.numFnsToEvaluate
        if isa(FnsToEvaluate{kk},'struct')
            if isfield(FnsToEvaluate{kk}, Names_i{ii})
                PTypeStructure.(iistr).FnsToEvaluate{jj}=FnsToEvaluate{kk}.(Names_i{ii});
                if isa(FnsToEvaluateParamNames(kk).Names,'struct')
                    PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names=FnsToEvaluateParamNames(kk).Names.(Names_i{ii});
                else
                    PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names=FnsToEvaluateParamNames(kk).Names;
                end
                PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
                PTypeStructure.FnsAndPTypeIndicator(kk,ii)=1;
                % else
                %  % do nothing as this FnToEvaluate is not relevant for the current PType
                % % Implicitly, PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=0
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            PTypeStructure.(iistr).FnsToEvaluate{jj}=FnsToEvaluate{kk};
            PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names=FnsToEvaluateParamNames(kk).Names;
            PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
            PTypeStructure.FnsAndPTypeIndicator(kk,ii)=1;
        end
    end
    
%     if isa(PTypeDistParamNames, 'array')
%         PTypeStructure.(iistr).PTypeWeight=PTypeDistParamNames(ii);
%     else
%         PTypeStructure.(iistr).PTypeWeight=PTypeStructure.(iistr).Parameters.(PTypeDistParamNames{1}); % Don't need '.(Names_i{ii}' as this was already done when putting it into PTypeStrucutre, and here I take it straing from PTypeStructure.(iistr).Parameters rather than from Parameters itself.
%     end
end

%%
% Have now finished creating PTypeStructure. 







% 
% %% Check which vfoptions have been used, set all others to defaults 
% if exist('vfoptions','var')==0
%     disp('No vfoptions given, using defaults')
%     %If vfoptions is not given, just use all the defaults
% %     vfoptions.exoticpreferences=0;
%     vfoptions.parallel=transpathoptions.parallel;
%     vfoptions.returnmatrix=2;
%     vfoptions.verbose=0;
%     vfoptions.lowmemory=0;
%     vfoptions.exoticpreferences=0;
%     vfoptions.polindorval=1;
%     vfoptions.policy_forceintegertype=0;
% else
%     %Check vfoptions for missing fields, if there are some fill them with the defaults
%     if isfield(vfoptions,'parallel')==0
%         vfoptions.parallel=transpathoptions.parallel; % GPU where available, otherwise parallel CPU.
%     end
%     if vfoptions.parallel==2
%         vfoptions.returnmatrix=2; % On GPU, must use this option
%     end
%     if isfield(vfoptions,'lowmemory')==0
%         vfoptions.lowmemory=0;
%     end
%     if isfield(vfoptions,'verbose')==0
%         vfoptions.verbose=0;
%     end
%     if isfield(vfoptions,'returnmatrix')==0
%         if isa(ReturnFn,'function_handle')==1
%             vfoptions.returnmatrix=0;
%         else
%             vfoptions.returnmatrix=1;
%         end
%     end
%     if isfield(vfoptions,'exoticpreferences')==0
%         vfoptions.exoticpreferences=0;
%     end
%     if isfield(vfoptions,'polindorval')==0
%         vfoptions.polindorval=1;
%     end
%     if isfield(vfoptions,'policy_forceintegertype')==0
%         vfoptions.policy_forceintegertype=0;
%     end
% end
% 
% %% Check which simoptions have been used, set all others to defaults 
% if isfield(transpathoptions,'simoptions')==1
%     simoptions=transpathoptions.simoptions;
% end
% if exist('simoptions','var')==0
%     simoptions.nsims=10^4;
%     simoptions.parallel=transpathoptions.parallel; % GPU where available, otherwise parallel CPU.
%     simoptions.verbose=0;
%     try 
%         PoolDetails=gcp;
%         simoptions.ncores=PoolDetails.NumWorkers;
%     catch
%         simoptions.ncores=1;
%     end
%     simoptions.iterate=1;
%     simoptions.tolerance=10^(-9);
% else
%     %Check vfoptions for missing fields, if there are some fill them with
%     %the defaults
%     if isfield(simoptions,'tolerance')==0
%         simoptions.tolerance=10^(-9);
%     end
%     if isfield(simoptions,'nsims')==0
%         simoptions.nsims=10^4;
%     end
%     if isfield(simoptions,'parallel')==0
%         simoptions.parallel=transpathoptions.parallel;
%     end
%     if isfield(simoptions,'verbose')==0
%         simoptions.verbose=0;
%     end
%     if isfield(simoptions,'ncores')==0
%         try
%             PoolDetails=gcp;
%             simoptions.ncores=PoolDetails.NumWorkers;
%         catch
%             simoptions.ncores=1;
%         end
%     end
%     if isfield(simoptions,'iterate')==0
%         simoptions.iterate=1;
%     end
% end

%% Check the sizes of some of the inputs
% N_d=prod(n_d);
% N_z=prod(n_z);
% N_a=prod(n_a);
% 
% if N_d>0
%     if size(d_grid)~=[N_d, 1]
%         disp('ERROR: d_grid is not the correct shape (should be of size N_d-by-1)')
%         dbstack
%         return
%     end
% end
% if size(a_grid)~=[N_a, 1]
%     disp('ERROR: a_grid is not the correct shape (should be of size N_a-by-1)')
%     dbstack
%     return
% elseif size(z_grid)~=[N_z, 1]
%     disp('ERROR: z_grid is not the correct shape (should be of size N_z-by-1)')
%     dbstack
%     return
% elseif size(pi_z)~=[N_z, N_z]
%     disp('ERROR: pi is not of size N_z-by-N_z')
%     dbstack
%     return
% end
% if length(PricePathNames)~=length(GeneralEqmEqns)
%     disp('ERROR: Initial PricePath contains less variables than GeneralEqmEqns')
%     dbstack
%     return
% end

%%
if transpathoptions.parallel==2 
%    % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
%    pi_z=gpuArray(pi_z);
%    if exists('d_grid','var')
%        d_grid=gpuArray(d_grid);
%    end
%    a_grid=gpuArray(a_grid);
%    z_grid=gpuArray(z_grid);
   PricePathOld=gpuArray(PricePathOld);
% else
%    % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
%    % This may be completely unnecessary.
%    pi_z=gather(pi_z);
%    if exists('d_grid','var')
%        d_grid=gather(d_grid);
%    end
%    a_grid=gather(a_grid);
%    z_grid=gather(z_grid);
%    PricePathOld=gather(PricePathOld);
end


%%
if transpathoptions.usestockvars==1 
    fprinf('ERROR: transpathoptions.usestockvars=1 not yet implemented with PType \n')
    dbstack
%     % Get the stock variable objects out of transpathoptions.
%     StockVariable_init=transpathoptions.stockvarinit;
% %     StockVariableEqns=transpathoptions.stockvareqns.lawofmotion;
%     % and switch from structure/names to matrix for VFI Toolkit internals.
%     StockVarsPathNames=fieldnames(transpathoptions.stockvarpath0);
%     StockVarsStruct=transpathoptions.stockvarpath0;
%     StockVarsPathOld=zeros(T,length(StockVarsPathNames));
%     StockVariableEqns=cell(1,length(StockVarsPathNames));
%     for ii=1:length(StockVarsPathNames)
%         StockVarsPathOld(:,ii)=StockVarsStruct.(StockVarsPathNames{ii});
%         StockVariableEqnParamNames(ii).Names=transpathoptions.stockvareqns.Names.(StockVarsPathNames{ii}); % Really, what I should do is redesign the way the GeneralEqm equations and names work so they are in a structure, and then just leave this in it's structure form as well.
%         StockVariableEqns{ii}=transpathoptions.stockvareqns.lawofmotion.(StockVarsPathNames{ii});
%     end
%     if transpathoptions.parallel==2
%         StockVarsPathOld=gpuArray(StockVarsPathOld);
%     end
end

%%
if transpathoptions.GEnewprice==1
    if transpathoptions.parallel==2
        if transpathoptions.usestockvars==0
            PricePathOld=TransitionPath_Case1_FHorz_PType_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, GeneralEqmEqns, GeneralEqmEqnParamNames, transpathoptions, PTypeStructure);
        else % transpathoptions.usestockvars==1
            % NOT YET IMPLEMENTED
%             [PricePathOld,StockVarsPathOld]=TransitionPath_Case1_FHorz_StockVar_PType_shooting(PricePathOld, PricePathNames, StockVarsPathOld, StockVarsPathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, StockVariable_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, StockVariableEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, StockVariableEqnParamNames, vfoptions, simoptions, transpathoptions);
        end
    else
        fprintf('ERROR: transpathoptions can only be used on GPU (transpathoptions.parallel=2) \n')
        dbstack
        return
%         PricePathOld=TransitionPath_Case1_FHorz_Par1_PType, shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions, transpathoptions);
    end
    % Switch the solution into structure for output.
    for ii=1:length(PricePathNames)
        PricePath.(PricePathNames{ii})=PricePathOld(:,ii);
    end
    if transpathoptions.usestockvars==1
        for ii=1:length(StockVarsPathNames)
            PricePath.(StockVarsPathNames{ii})=StockVarsPathOld(:,ii);
        end
    end
    return
end

l_p=size(PricePathOld,2);

if transpathoptions.verbose==1
    transpathoptions
end

if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

% %% Set up transition path as minimization of a function (default is to use as objective the weighted sum of squares of the general eqm conditions)
% PricePathVec=gather(reshape(PricePathOld,[T*length(PricePathNames),1])); % Has to be vector of fminsearch. Additionally, provides a double check on sizes.
% 
% if transpathoptions.GEnewprice==2 % Function minimization
%     % CURRENTLY ONLY WORKS WITH GPU (PARALLEL==2)
%     if n_d(1)==0
%         GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_FHorz_no_d_subfn(pricepathPricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions,transpathoptions);
%     else
%         GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_FHorz_subfn(pricepath, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions,transpathoptions);
%     end
% end
% 
% % if transpathoptions.GEnewprice2algo==0
% [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathVec);
% % else
% %     [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathOld);
% % end
% 
% % LOOK INTO USING 'SURROGATE OPTIMIZATION'
% 
% if transpathoptions.parallel==2
%     PricePath=gpuArray(reshape(PricePath,[T,length(PricePathNames)])); % Switch back to appropriate shape (out of the vector required to use fminsearch)
% else
%     PricePath=gather(reshape(PricePath,[T,length(PricePathNames)])); % Switch back to appropriate shape (out of the vector required to use fminsearch)    
% end
% 
% for ii=1:length(PricePathNames)
%     PricePath.(PricePathNames{ii})=PricePathOld(:,ii);
% end


end