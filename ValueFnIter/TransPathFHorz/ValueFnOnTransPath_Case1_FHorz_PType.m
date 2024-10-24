function [VPath,PolicyPath]=ValueFnOnTransPath_Case1_FHorz_PType(PricePath, ParamPath, T, V_final, Policy_final, Parameters, n_d, n_a, n_z, N_j, Names_i, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)

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

    % First set up transpathoptions
    if exist('transpathoptions','var')
        transpathoptions_temp=PType_Options(transpathoptions,Names_i,ii);
        if ~isfield(transpathoptions_temp,'verbose')
            transpathoptions_temp.verbose=0;
        end
        if ~isfield(transpathoptions_temp,'verboseparams')
            transpathoptions_temp.verboseparams=0;
        end
    else
        transpathoptions_temp.verbose=0;
        transpathoptions_temp.verboseparams=0;
    end

    % First set up vfoptions
    if exist('vfoptions','var')
        vfoptions_temp=PType_Options(vfoptions,Names_i,ii);
        if ~isfield(vfoptions_temp,'verbose')
            vfoptions_temp.verbose=0;
        end
        if ~isfield(vfoptions_temp,'verboseparams')
            vfoptions_temp.verboseparams=0;
        end
        if ~isfield(vfoptions_temp,'ptypestorecpu')
            vfoptions_temp.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
        end
    else
        vfoptions_temp.verbose=0;
        vfoptions_temp.verboseparams=0;
        vfoptions_temp.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    end 
    
    if vfoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end
           
    V_final_temp=V_final.(Names_i{ii});
    Policy_final_temp=Policy_final.(Names_i{ii});
    

    % Go through everything which might be dependent on permanent type (PType)
    % Notice that the way this is coded the grids (etc.) could be either
    % fixed, or a function (that depends on age, and possibly on permanent
    % type), or they could be a structure. Only in the case where they are
    % a structure is there a need to take just a specific part and send
    % only that to the 'non-PType' version of the command.
    if isa(n_d,'struct')
        n_d_temp=n_d.(Names_i{ii});
    else
        n_d_temp=n_d;
    end
    if isa(n_a,'struct')
        n_a_temp=n_a.(Names_i{ii});
    else
        n_a_temp=n_a;
    end
    if isa(n_z,'struct')
        n_z_temp=n_z.(Names_i{ii});
    else
        n_z_temp=n_z;
    end
    if isa(N_j,'struct')
        N_j_temp=N_j.(Names_i{ii});
    else
        N_j_temp=N_j;
    end
    if isa(d_grid,'struct')
        d_grid_temp=d_grid.(Names_i{ii});
    else
        d_grid_temp=d_grid;
    end
    if isa(a_grid,'struct')
        a_grid_temp=a_grid.(Names_i{ii});
    else
        a_grid_temp=a_grid;
    end
    if isa(z_grid,'struct')
        z_grid_temp=z_grid.(Names_i{ii});
    else
        z_grid_temp=z_grid;
    end
    if isa(pi_z,'struct')
        pi_z_temp=pi_z.(Names_i{ii});
    else
        pi_z_temp=pi_z;
    end
    if isa(ReturnFn,'struct')
        ReturnFn_temp=ReturnFn.(Names_i{ii});
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
    
    if vfoptions_temp.verboseparams==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end
    
    % ParamPath can include parameters that differ by ptype
    ParamPath_temp=ParamPath;
    ParamPathNames=fieldnames(ParamPath);
    for nn=1:length(ParamPathNames)
        if isstruct(ParamPath_temp.(ParamPathNames{nn}))
            ParamPath_temp.(ParamPathNames{nn})=ParamPath.(ParamPathNames{nn}).(Names_i{ii});
        end
    end


    [VPath_ii,PolicyPath_ii]=ValueFnOnTransPath_Case1_FHorz(PricePath, ParamPath_temp, T, V_final_temp, Policy_final_temp, Parameters_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, pi_z_temp, d_grid_temp, a_grid_temp,z_grid_temp, DiscountFactorParamNames_temp, ReturnFn_temp, transpathoptions_temp, vfoptions_temp);
    % Note: T cannot depend on ptype, nor can PricePath depend on ptype

    if vfoptions_temp.ptypestorecpu==1
        VPath.(Names_i{ii})=gather(VPath_ii);
        PolicyPath.(Names_i{ii})=gather(PolicyPath_ii);
    else
        VPath.(Names_i{ii})=VPath_ii;
        PolicyPath.(Names_i{ii})=PolicyPath_ii;
    end
        
    clear VPath_ii PolicyPath_ii

end


end


