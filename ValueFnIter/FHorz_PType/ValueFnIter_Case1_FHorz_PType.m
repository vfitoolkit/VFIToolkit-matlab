function [V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z, N_j,Names_i,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)

%
% vfoptions.verbose=1 will give feedback
% vfoptions.verboseparams=1 will give further feedback on the param values of each permanent type
%
V=struct();
Policy=struct();
pt_temp=struct("n_d", n_d, "n_a", n_a, "n_z", n_z, "N_j", N_j, "d_grid", d_grid, "a_grid", a_grid, "z_grid", z_grid, "pi_z", pi_z, "ReturnFn", ReturnFn);

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);
% N_i=prod(n_i);

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

% Go through everything which might be dependent on fixed type (PType)
for n_x=["n_d" "n_a" "n_z" "N_j"]
    if isstruct(pt_temp.(n_x))
        pt_temp.(n_x)=pt_temp.(n_x).(Names_i{1})
    end
end
for x_grid=["d_grid" "a_grid" "z_grid" "pi_z"]
    if isstruct(pt_temp.(x_grid))
        pt_temp.(x_grid)=pt_temp.(x_grid).(Names_i{1})
    end
end
if isstruct(pt_temp.ReturnFn)
    pt_temp.ReturnFn=pt_temp.ReturnFn.(Names_i{1});
end

for ii=1:N_i
        
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
            vfoptions_temp.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
        end
    else
        vfoptions_temp.verbose=0;
        vfoptions_temp.verboseparams=0;
        vfoptions_temp.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
    end 
    
    if vfoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
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

    % ReturnFnParamNames_temp=ReturnFnParamNamesFn(ReturnFn_temp,n_d_temp,n_a_temp,n_z_temp,vfoptions_temp,Parameters_temp);
    
    if vfoptions_temp.verboseparams==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end

    pt_ns = {pt_temp.n_d,pt_temp.n_a,pt_temp.n_z};
    pt_grids = {pt_temp.d_grid, pt_temp.a_grid, pt_temp.z_grid, pt_temp.pi_z};
    if isfinite(pt_temp.N_j)
        [V_ii, Policy_ii]=ValueFnIter_Case1_FHorz(pt_ns{:},pt_temp.N_j,pt_grids{:}, pt_temp.ReturnFn, Parameters_temp, DiscountFactorParamNames_temp, [], vfoptions_temp);
    else % PType actually allows for infinite horizon as well
        [V_ii, Policy_ii]=ValueFnIter_Case1(pt_ns{:},pt_grids{:}, pt_temp.ReturnFn, Parameters_temp, DiscountFactorParamNames_temp, [], vfoptions_temp);
    end

    
    if vfoptions_temp.ptypestorecpu==1
        V.(Names_i{ii})=gather(V_ii); 
        Policy.(Names_i{ii})=gather(Policy_ii);
    else
        V.(Names_i{ii})=V_ii;
        Policy.(Names_i{ii})=Policy_ii;
    end
        
    clear V_ii Policy_ii

end


end