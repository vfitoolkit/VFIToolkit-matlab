function [VPath,PolicyPath]=ValueFnOnTransPath_InfHorz_PType(PricePath, ParamPath, T, V_final, Policy_final, Parameters, n_d,n_a,n_z, Names_i, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)

VPath=struct();
PolicyPath=struct();

%%
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
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

%% Loop over permanent types
for ii=1:N_i
    iistr=Names_i{ii};

    transpathoptions_temp=PType_Options(transpathoptions,iistr);
    % Initial values for _tminus1 can differ by ptype: as a structure (by ptype name), or a vector of length N_i
    if exist('transpathoptions','var') && isfield(transpathoptions,'initialvalues')
        transpathoptions_temp.initialvalues=transpathoptions.initialvalues;
        ivnames=fieldnames(transpathoptions.initialvalues);
        for nn=1:length(ivnames)
            temp=transpathoptions.initialvalues.(ivnames{nn});
            if isstruct(temp)
                transpathoptions_temp.initialvalues.(ivnames{nn})=temp.(iistr);
            elseif numel(temp)==N_i && N_i>1
                transpathoptions_temp.initialvalues.(ivnames{nn})=temp(ii);
            end
        end
    end

    vfoptions_temp=PType_Options(vfoptions,iistr);
    vfoptions_temp.parallel=2; % hardcode
    if ~isfield(vfoptions_temp,'verbose')
        vfoptions_temp.verbose=0; % =1 fives feedback
    end
    if ~isfield(vfoptions_temp,'ptypestorecpu')
        vfoptions_temp.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    end

    if vfoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end

    V_final_temp=V_final.(iistr);
    Policy_final_temp=Policy_final.(iistr);


    % Go through everything which might be dependent on permanent type (PType)
    % Notice that the way this is coded the grids (etc.) could be either
    % fixed, or a function (that depends on age, and possibly on permanent
    % type), or they could be a structure. Only in the case where they are
    % a structure is there a need to take just a specific part and send
    % only that to the 'non-PType' version of the command.
    if isa(n_d,'struct')
        n_d_temp=n_d.(iistr);
    else
        n_d_temp=n_d;
    end
    if isa(n_a,'struct')
        n_a_temp=n_a.(iistr);
    else
        n_a_temp=n_a;
    end
    if isa(n_z,'struct')
        n_z_temp=n_z.(iistr);
    else
        n_z_temp=n_z;
    end
    if isa(d_grid,'struct')
        d_grid_temp=d_grid.(iistr);
    else
        d_grid_temp=d_grid;
    end
    if isa(a_grid,'struct')
        a_grid_temp=a_grid.(iistr);
    else
        a_grid_temp=a_grid;
    end
    if isa(z_grid,'struct')
        z_grid_temp=z_grid.(iistr);
    else
        z_grid_temp=z_grid;
    end
    if isa(pi_z,'struct')
        pi_z_temp=pi_z.(iistr);
    else
        pi_z_temp=pi_z;
    end
    if isa(ReturnFn,'struct')
        ReturnFn_temp=ReturnFn.(iistr);
    else
        ReturnFn_temp=ReturnFn;
    end

    % Parameters are allowed to be given as structure, or as vector/matrix
    % (in terms of their dependence on fixed type). So go through each of
    % these in term.
    Parameters_temp=Parameters;
    FullParamNames=fieldnames(Parameters);
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check for permanent type in structure form
            names=fieldnames(Parameters.(FullParamNames{kField}));
            for jj=1:length(names)
                if strcmp(names{jj},Names_i{ii})
                    Parameters_temp.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(names{jj});
                end
            end
        elseif any(size(Parameters.(FullParamNames{kField}))==N_i) % Check for permanent type in vector/matrix form.
            temp=Parameters.(FullParamNames{kField});
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType.
            if ptypedim==1
                Parameters_temp.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                Parameters_temp.(FullParamNames{kField})=temp(:,ii);
            end
        end
    end
    DiscountFactorParamNames_temp=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        names=fieldnames(DiscountFactorParamNames);
        for jj=1:length(names)
            if strcmp(names{jj},Names_i{ii})
                DiscountFactorParamNames_temp=DiscountFactorParamNames.(names{jj});
            end
        end
    end

    % PricePath can include prices that differ by ptype: as a structure (by ptype name), or with N_i rows
    PricePath_temp=PricePath;
    PricePathNames=fieldnames(PricePath);
    for nn=1:length(PricePathNames)
        if isstruct(PricePath_temp.(PricePathNames{nn}))
            PricePath_temp.(PricePathNames{nn})=PricePath.(PricePathNames{nn}).(iistr);
        elseif N_i>1 && size(PricePath_temp.(PricePathNames{nn}),1)==N_i && size(PricePath_temp.(PricePathNames{nn}),2)==T
            temp=PricePath_temp.(PricePathNames{nn});
            PricePath_temp.(PricePathNames{nn})=temp(ii,:);
        elseif N_i>1 && size(PricePath_temp.(PricePathNames{nn}),2)==N_i && size(PricePath_temp.(PricePathNames{nn}),1)==T
            temp=PricePath_temp.(PricePathNames{nn});
            PricePath_temp.(PricePathNames{nn})=temp(:,ii);
        end
    end
    % ParamPath can include parameters that differ by ptype: as a structure (by ptype name; a ptype it has no
    % field for is not on the path for that parameter), or with N_i rows
    ParamPath_temp=ParamPath;
    ParamPathNames=fieldnames(ParamPath);
    for nn=1:length(ParamPathNames)
        if isstruct(ParamPath_temp.(ParamPathNames{nn}))
            if isfield(ParamPath.(ParamPathNames{nn}),iistr)
                ParamPath_temp.(ParamPathNames{nn})=ParamPath.(ParamPathNames{nn}).(iistr);
            else
                ParamPath_temp=rmfield(ParamPath_temp,ParamPathNames{nn});
            end
        elseif N_i>1 && size(ParamPath_temp.(ParamPathNames{nn}),1)==N_i && size(ParamPath_temp.(ParamPathNames{nn}),2)==T
            temp=ParamPath_temp.(ParamPathNames{nn});
            ParamPath_temp.(ParamPathNames{nn})=temp(ii,:);
        elseif N_i>1 && size(ParamPath_temp.(ParamPathNames{nn}),2)==N_i && size(ParamPath_temp.(ParamPathNames{nn}),1)==T
            temp=ParamPath_temp.(ParamPathNames{nn});
            ParamPath_temp.(ParamPathNames{nn})=temp(:,ii);
        end
    end

    [VPath_ii,PolicyPath_ii]=ValueFnOnTransPath_InfHorz(PricePath_temp, ParamPath_temp, T, V_final_temp, Policy_final_temp, Parameters_temp, n_d_temp, n_a_temp, n_z_temp, d_grid_temp, a_grid_temp,z_grid_temp, pi_z_temp, DiscountFactorParamNames_temp, ReturnFn_temp, transpathoptions_temp, vfoptions_temp);
    % Note: T cannot depend on ptype

    if vfoptions_temp.ptypestorecpu==1
        VPath.(iistr)=gather(VPath_ii);
        PolicyPath.(iistr)=gather(PolicyPath_ii);
    else
        VPath.(iistr)=VPath_ii;
        PolicyPath.(iistr)=PolicyPath_ii;
    end

    clear VPath_ii PolicyPath_ii

end


end


