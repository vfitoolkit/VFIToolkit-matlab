function [VPath,PolicyPath]=ValueFnOnTransPath_Case1_FHorz_PType(PricePath, ParamPath, T, V_final, Policy_final, AgentDist_initial, Parameters, n_d, n_a, n_z, N_j, Names_i, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames,AgeWeightsParamNames, transpathoptions, simoptions, vfoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 

% PricePath is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePath


%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-4);
    transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    transpathoptions.fastOLG=0;
    transpathoptions.exoticpreferences=0;
    transpathoptions.verbose=0;
    transpathoptions.stockvars=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'tolerance')==0
        transpathoptions.tolerance=10^(-4);
    end
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(transpathoptions,'fastOLG')==0
        transpathoptions.fastOLG=0; % I have not yet actually implemented this for EvalFnOnTransPath
    end
    if isfield(transpathoptions,'exoticpreferences')==0
        transpathoptions.exoticpreferences=0;
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


%%
% Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
PricePathStruct=PricePath; % I do this here just to make it easier for the user to read and understand the inputs.
PricePathNames=fieldnames(PricePathStruct);
ParamPathStruct=ParamPath; % I do this here just to make it easier for the user to read and understand the inputs.
ParamPathNames=fieldnames(ParamPathStruct);
if transpathoptions.parallel==2 
    PricePath=zeros(T,length(PricePathNames),'gpuArray');
    for ii=1:length(PricePathNames)
        PricePath(:,ii)=gpuArray(PricePathStruct.(PricePathNames{ii}));
    end
    ParamPath=zeros(T,length(ParamPathNames),'gpuArray');
    for ii=1:length(ParamPathNames)
        ParamPath(:,ii)=gpuArray(ParamPathStruct.(ParamPathNames{ii}));
    end
else
    PricePath=zeros(T,length(PricePathNames));
    for ii=1:length(PricePathNames)
        PricePath(:,ii)=gather(PricePathStruct.(PricePathNames{ii}));
    end
    ParamPath=zeros(T,length(ParamPathNames));
    for ii=1:length(ParamPathNames)
        ParamPath(:,ii)=gather(ParamPathStruct.(ParamPathNames{ii}));
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
    
    
%     if isa(PTypeDistParamNames, 'array')
%         PTypeStructure.(iistr).PTypeWeight=PTypeDistParamNames(ii);
%     else
%         PTypeStructure.(iistr).PTypeWeight=PTypeStructure.(iistr).Parameters.(PTypeDistParamNames{1}); % Don't need '.(Names_i{ii}' as this was already done when putting it into PTypeStrucutre, and here I take it straing from PTypeStructure.(iistr).Parameters rather than from Parameters itself.
%     end
end


%%
if transpathoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   pi_z=gpuArray(pi_z);
   if exist('d_grid','var')
       d_grid=gpuArray(d_grid);
   end
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
   PricePath=gpuArray(PricePath);
else
   % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
   % This may be completely unnecessary.
   pi_z=gather(pi_z);
   if exist('d_grid','var')
       d_grid=gather(d_grid);
   end
   a_grid=gather(a_grid);
   z_grid=gather(z_grid);
   PricePath=gather(PricePath);
end

if transpathoptions.exoticpreferences~=0
    disp('ERROR: Only transpathoptions.exoticpreferences==0 is supported by TransitionPath_Case1')
    dbstack
end


l_p=size(PricePath,2);

if transpathoptions.verbose==1
    transpathoptions
end
if transpathoptions.verbose==1
%     DiscountFactorParamNames
%     ReturnFnParamNames
    ParamPathNames
    PricePathNames
end


if transpathoptions.parallel==2
    
    AgentDist_initial.ptweights=AgentDist_initial.ptweights;
%     for ii=1:PTypeStructure.N_i
%         iistr=PTypeStructure.Names_i{ii};
%         
%         N_a=prod(PTypeStructure.(iistr).n_a);
%         N_z=prod(PTypeStructure.(iistr).n_z);
%         N_j=PTypeStructure.(iistr).N_j;
%         V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z,N_j]);
%         AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a*N_z,N_j]);
%         % V=zeros(size(V_final),'gpuArray'); %preallocate space
%     end
    
    PolicyPath=struct();
    VPath=struct();
    
    % For each agent type, first go back through the value & policy fns.
    % Then forwards through agent dist and agg vars.
    for ii=1:PTypeStructure.N_i
        iistr=PTypeStructure.Names_i{ii};
        
        % Grab everything relevant out of PTypeStructure
        n_d=PTypeStructure.(iistr).n_d; N_d=prod(n_d);
        n_a=PTypeStructure.(iistr).n_a; N_a=prod(n_a);
        n_z=PTypeStructure.(iistr).n_z; N_z=prod(n_z);
        N_j=PTypeStructure.(iistr).N_j;
        d_grid=PTypeStructure.(iistr).d_grid;
        a_grid=PTypeStructure.(iistr).a_grid;
        z_grid=PTypeStructure.(iistr).z_grid;
        pi_z=PTypeStructure.(iistr).pi_z;
        ReturnFn=PTypeStructure.(iistr).ReturnFn;
        Parameters=PTypeStructure.(iistr).Parameters;
        DiscountFactorParamNames=PTypeStructure.(iistr).DiscountFactorParamNames;
        ReturnFnParamNames=PTypeStructure.(iistr).ReturnFnParamNames;
        vfoptions=PTypeStructure.(iistr).vfoptions;
        simoptions=PTypeStructure.(iistr).simoptions;
        
        V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z,N_j]);
        AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a*N_z,N_j]);
        
        if N_d>0
            PolicyPath_ii=zeros(2,N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath_ii(:,:,:,:,T)=Policy_final.(iistr);
%             PolicyIndexesPath.(iistr)=zeros(2,N_a,N_z,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        else
            PolicyPath_ii=zeros(N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath_ii(:,:,:,T)=Policy_final.(iistr);
%             PolicyIndexesPath.(iistr)=zeros(N_a,N_z,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        
        VPath_ii=zeros(N_a,N_z,N_j,T,'gpuArray');
        VPath_ii(:,:,:,T)=V_final.(iistr);

        %First, go from T-1 to 1 calculating the Value function and Optimal
        %policy function at each step. Since we won't need to keep the value
        %functions for anything later we just store the next period one in
        %Vnext, and the current period one to be calculated in V
        Vnext=V_final.(iistr);
        for tt=1:T-1 %so t=T-i
            
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(T-tt,kk);
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-tt,kk);
            end
            
            [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep(Vnext,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The VKron input is next period value fn, the VKron output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')
            
            if N_d>0
                PolicyPath_ii(:,:,:,:,T-tt)=Policy;
%                 PolicyIndexesPath.(iistr)(:,:,:,:,T-tt)=Policy;
            else
                PolicyPath_ii(:,:,:,T-tt)=Policy;
%                 PolicyIndexesPath.(iistr)(:,:,:,T-tt)=Policy;
            end
            VPath_ii(:,:,:,T-tt)=V;
            
            Vnext=V;
        end

        VPath.(iistr)=VPath_ii;
        PolicyPath.(iistr)=PolicyPath_ii;
    end
    
else
    AgentDist_initial.ptweights=AgentDist_initial.ptweights;
%     for ii=1:PTypeStructure.N_i
%         iistr=PTypeStructure.Names_i{ii};
%         
%         N_a=prod(PTypeStructure.(iistr).n_a);
%         N_z=prod(PTypeStructure.(iistr).n_z);
%         N_j=PTypeStructure.(iistr).N_j;
%         V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z,N_j]);
%         AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a*N_z,N_j]);
%         % V=zeros(size(V_final),'gpuArray'); %preallocate space
%     end
    
    PolicyPath_ii=struct();
    
    % For each agent type, first go back through the value & policy fns.
    for ii=1:PTypeStructure.N_i
        iistr=PTypeStructure.Names_i{ii};
        
        % Grab everything relevant out of PTypeStructure
        n_d=PTypeStructure.(iistr).n_d; N_d=prod(n_d);
        n_a=PTypeStructure.(iistr).n_a; N_a=prod(n_a);
        n_z=PTypeStructure.(iistr).n_z; N_z=prod(n_z);
        N_j=PTypeStructure.(iistr).N_j;
        d_grid=PTypeStructure.(iistr).d_grid;
        a_grid=PTypeStructure.(iistr).a_grid;
        z_grid=PTypeStructure.(iistr).z_grid;
        pi_z=PTypeStructure.(iistr).pi_z;
        ReturnFn=PTypeStructure.(iistr).ReturnFn;
        Parameters=PTypeStructure.(iistr).Parameters;
        DiscountFactorParamNames=PTypeStructure.(iistr).DiscountFactorParamNames;
        ReturnFnParamNames=PTypeStructure.(iistr).ReturnFnParamNames;
        vfoptions=PTypeStructure.(iistr).vfoptions;
        simoptions=PTypeStructure.(iistr).simoptions;
        
        V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z,N_j]);
        AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a*N_z,N_j]);
        
        if N_d>0
            PolicyPath_ii=zeros(2,N_a,N_z,N_j,T); %Periods 1 to T
            PolicyPath_ii(:,:,:,:,T)=Policy_final.(iistr);
%             PolicyIndexesPath.(iistr)=zeros(2,N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T
        else
            PolicyPath_ii=zeros(N_a,N_z,N_j,T); %Periods 1 to T
            PolicyPath_ii(:,:,:,T)=Policy_final.(iistr);
%             PolicyIndexesPath.(iistr)=zeros(N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T
        end
        VPath_ii=zeros(N_a,N_z,N_j,T,'gpuArray');
        VPath_ii(:,:,:,T)=V_final.(iistr);
        
        %First, go from T-1 to 1 calculating the Value function and Optimal
        %policy function at each step. Since we won't need to keep the value
        %functions for anything later we just store the next period one in
        %Vnext, and the current period one to be calculated in V
        Vnext=V_final.(iistr);
        for tt=1:T-1 %so t=T-i
            
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(T-tt,kk);
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-tt,kk);
            end
            
            [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep(Vnext,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The VKron input is next period value fn, the VKron output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')
            
            if N_d>0
                PolicyPath_ii(:,:,:,:,T-tt)=Policy;
%                 PolicyIndexesPath.(iistr)(:,:,:,:,T-tt)=Policy;
            else
                PolicyPath_ii(:,:,:,T-tt)=Policy;
%                 PolicyIndexesPath.(iistr)(:,:,:,T-tt)=Policy;
            end
            
            VPath_ii(:,:,:,T-tt)=V;

            Vnext=V;
        end
        
        VPath.(iistr)=VPath_ii;
        PolicyPath.(iistr)=PolicyPath_ii;
        
    end

end

end