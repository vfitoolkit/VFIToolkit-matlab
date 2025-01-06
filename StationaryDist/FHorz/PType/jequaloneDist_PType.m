function [jequaloneDist,idiminj1dist,Parameters]=jequaloneDist_PType(jequaloneDist,Parameters,simoptions,n_a,n_z,N_i,Names_i,PTypeDistParamNames,outputstruct)
% Deal with jequaloneDist for models with PType
% Parameters is only needed as we might update Parameters.(PTypeDistParamNames{1})

if ~isfield(simoptions,'warnjequaloneptypeasdim')
    simoptions.warnjequaloneptypeasdim=1;
end

% If age one distribution is input as a function, then evaluate it
if isa(jequaloneDist, 'function_handle')
    jequaloneDistFn=jequaloneDist;
    clear jequaloneDist
    % figure out any parameters
    temp=getAnonymousFnInputNames(jequaloneDistFn);
    if length(temp)>4 % first 4 are a_grid,z_grid,n_a,n_z
        jequaloneDistFnParamNames={temp{5:end}}; % the first inputs will always be (a_grid,z_grid,n_a,n_z,...)
    else
        jequaloneDistFnParamNames={};
    end
    jequaloneParamsCell={};
    for pp=1:length(jequaloneDistFnParamNames)
        jequaloneParamsCell{pp}=Parameters.(jequaloneDistFnParamNames{pp});
    end
    % make sure a_grid and z_grid have been put in simoptions
    if ~isfield(simoptions,'a_grid')
        error('When using jequaloneDist as a function you must put a_grid into simoptions.a_grid')
    elseif ~isfield(simoptions,'z_grid')
        error('When using jequaloneDist as a function you must put z_grid into simoptions.z_grid')
    end

    jequaloneDist=jequaloneDistFn(simoptions.a_grid,simoptions.z_grid,n_a,n_z,jequaloneParamsCell{:});
end


if ~isstruct(jequaloneDist)
    % Using matrix, reshape now to save multiple reshapes later
    % (Note that matrix implies same grids for all agents)
    if isfield(simoptions,'n_e')
        if isfield(simoptions,'n_semiz')
            if prod(n_z)==0
                if all(size(jequaloneDist)==[n_a,simoptions.n_semiz,simoptions.n_e])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz),prod(simoptions.n_e)]);
                    idiminj1dist=0;
                elseif all(size(jequaloneDist)==[n_a,simoptions.n_semiz,simoptions.n_e,N_i])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz),prod(simoptions.n_e),N_i]);
                    idiminj1dist=1; % ptype is a dimension of the jequaloneDist
                    if outputstruct==1
                        jequaloneDist_copy=jequaloneDist;
                        clear jequaloneDist
                        jequaloneDist=struct();
                        for ii=1:N_i
                            jequaloneDist.(Names_i{ii})=jequaloneDist_copy(:,:,:,ii);
                        end
                    end
                end
            else
                if all(size(jequaloneDist)==[n_a,simoptions.n_semiz,n_z,simoptions.n_e])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz)*prod(n_z)*prod(simoptions.n_e)]);
                    idiminj1dist=0;
                elseif all(size(jequaloneDist)==[n_a,simoptions.n_semiz,n_z,simoptions.n_e,N_i])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz)*prod(n_z)*prod(simoptions.n_e),N_i]);
                    idiminj1dist=1; % ptype is a dimension of the jequaloneDist
                    if outputstruct==1
                        jequaloneDist_copy=jequaloneDist;
                        clear jequaloneDist
                        jequaloneDist=struct();
                        for ii=1:N_i
                            jequaloneDist.(Names_i{ii})=jequaloneDist_copy(:,:,ii);
                        end
                    end
                end
            end
        else
            if prod(n_z)==0
                if all(size(jequaloneDist)==[n_a,simoptions.n_e])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_e)]);
                    idiminj1dist=0;
                elseif all(size(jequaloneDist)==[n_a,simoptions.n_e,N_i])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_e),N_i]);
                    idiminj1dist=1; % ptype is a dimension of the jequaloneDist
                    if outputstruct==1
                        jequaloneDist_copy=jequaloneDist;
                        clear jequaloneDist
                        jequaloneDist=struct();
                        for ii=1:N_i
                            jequaloneDist.(Names_i{ii})=jequaloneDist_copy(:,:,ii);
                        end
                    end
                end
            else
                if all(size(jequaloneDist)==[n_a,n_z,simoptions.n_e])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(n_z),prod(simoptions.n_e)]);
                    idiminj1dist=0;
                elseif all(size(jequaloneDist)==[n_a,n_z,simoptions.n_e,N_i])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(n_z),prod(simoptions.n_e),N_i]);
                    idiminj1dist=1; % ptype is a dimension of the jequaloneDist
                    if outputstruct==1
                        jequaloneDist_copy=jequaloneDist;
                        clear jequaloneDist
                        jequaloneDist=struct();
                        for ii=1:N_i
                            jequaloneDist.(Names_i{ii})=jequaloneDist_copy(:,:,:,ii);
                        end
                    end
                end
            end
        end
    else % no e variable
        if isfield(simoptions,'n_semiz')
            if prod(n_z)==0
                if all(size(jequaloneDist)==[n_a,simoptions.n_semiz])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz)]);
                    idiminj1dist=0;
                elseif all(size(jequaloneDist)==[n_a,simoptions.n_semiz,N_i])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz),N_i]);
                    idiminj1dist=1; % ptype is a dimension of the jequaloneDist
                    if outputstruct==1
                        jequaloneDist_copy=jequaloneDist;
                        clear jequaloneDist
                        jequaloneDist=struct();
                        for ii=1:N_i
                            jequaloneDist.(Names_i{ii})=jequaloneDist_copy(:,:,ii);
                        end
                    end
                end
            else
                if all(size(jequaloneDist)==[n_a,simoptions.n_semiz,n_z])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz)*prod(n_z)]);
                    idiminj1dist=0;
                elseif all(size(jequaloneDist)==[n_a,simoptions.n_semiz,n_z,N_i])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz)*prod(n_z),N_i]);
                    idiminj1dist=1; % ptype is a dimension of the jequaloneDist
                    if outputstruct==1
                        jequaloneDist_copy=jequaloneDist;
                        clear jequaloneDist
                        jequaloneDist=struct();
                        for ii=1:N_i
                            jequaloneDist.(Names_i{ii})=jequaloneDist_copy(:,:,ii);
                        end
                    end
                end
            end
        else
            if prod(n_z)==0
                if all(size(jequaloneDist)==[n_a,1])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),1]);
                    idiminj1dist=0;
                elseif all(size(jequaloneDist)==[n_a,N_i])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),N_i]);
                    idiminj1dist=1; % ptype is a dimension of the jequaloneDist
                    if outputstruct==1
                        jequaloneDist_copy=jequaloneDist;
                        clear jequaloneDist
                        jequaloneDist=struct();
                        for ii=1:N_i
                            jequaloneDist.(Names_i{ii})=jequaloneDist_copy(:,ii);
                        end
                    end
                end
            else
                if numel(jequaloneDist)==prod([n_a,n_z]) % avoid size()=[n_a,n_z] because this errors if last dimensions are singular as they get dropped from jequaloneDist
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(n_z)]);
                    idiminj1dist=0;
                elseif all(size(jequaloneDist)==[n_a,n_z,N_i])
                    jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(n_z),N_i]);
                    idiminj1dist=1; % ptype is a dimension of the jequaloneDist
                    if outputstruct==1
                        jequaloneDist_copy=jequaloneDist;
                        clear jequaloneDist
                        jequaloneDist=struct();
                        for ii=1:N_i
                            jequaloneDist.(Names_i{ii})=jequaloneDist_copy(:,:,ii);
                        end
                    end
                end
            end
        end
    end
end

% If the initial agent distribution has ptype as a dimension, then use this to overwrite what the ptype masses are
if idiminj1dist==1
    if simoptions.warnjequaloneptypeasdim==1
        warning('jequaloneDist has ptype as a dimension, so using implicit masses for ptypes and ignoring value of Parameter PTypeDistParamNames')
    end
    if outputstruct==1
        ptypemass=zeros(N_i,1);
        for ii=1:N_i
            mass_ii=sum(sum(sum(jequaloneDist.(Names_i{ii}))));
            ptypemass(ii)=mass_ii; % store mass of type ii
            jequaloneDist.(Names_i{ii})=jequaloneDist.(Names_i{ii})/mass_ii; % Normalize to one
        end
    else
        if length(jequaloneDist)==2
            Parameters.(PTypeDistParamNames{1})=sum(jequaloneDist,1)'; % column vector
        elseif length(jequaloneDist)==3 % (a,z,j) in kron form
            Parameters.(PTypeDistParamNames{1})=shiftdim(sum(sum(jequaloneDist,1),2),2); % column vector
        end
    end
end





end