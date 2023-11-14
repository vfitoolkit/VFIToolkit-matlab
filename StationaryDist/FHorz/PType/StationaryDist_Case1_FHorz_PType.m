function StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,Names_i,pi_z,Parameters,simoptions)
% Allows for different permanent (fixed) types of agent. 
% See ValueFnIter_Case1_FHorz_PType for general idea.
%
% simoptions.verbose=1 will give feedback
% simoptions.verboseparams=1 will give further feedback on the param values of each permanent type
%
% jequaloneDist can either be same for all permanent types, or must be passed as a structure.
% AgeWeightParamNames is either same for all permanent types, or must be passed as a structure.
%
%
% How exactly to handle these differences between permanent (fixed) types
% is to some extent left to the user. You can, for example, input
% parameters that differ by permanent type as a vector with different rows f
% for each type, or as a structure with different fields for each type.
%
% Any input that does not depend on the permanent type is just passed in
% exactly the same form as normal.

% Names_i can either be a cell containing the 'names' of the different
% permanent types, or if there are no structures used (just parameters that
% depend on permanent type and inputted as vectors or matrices as appropriate) 
% then Names_i can just be the number of permanent types (but does not have to be, can still be names).
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

if ~isstruct(jequaloneDist)
    % Using matrix, reshape now to save multiple reshapes later
    % (Note that matrix implies same grids for all agents)
    if isfield(simoptions,'n_semiz')
        if prod(n_z)==0
            if all(size(jequaloneDist)==[n_a,simoptions.n_semiz])
                jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz)]);
                idiminj1dist=0;
            elseif all(size(jequaloneDist)==[n_a,simoptions.n_semiz,N_i])
                jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz),N_i]);
                idiminj1dist=1;
            end
        else
            if all(size(jequaloneDist)==[n_a,simoptions.n_semiz,n_z])
                jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz)*prod(n_z)]);
                idiminj1dist=0;
            elseif all(size(jequaloneDist)==[n_a,simoptions.n_semiz,n_z,N_i])
                jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(simoptions.n_semiz)*prod(n_z),N_i]);
                idiminj1dist=1;
            end
        end
    else
        if prod(n_z)==0
            if all(size(jequaloneDist)==[n_a,1])
                jequaloneDist=reshape(jequaloneDist,[prod(n_a),1]);
                idiminj1dist=0;
            elseif all(size(jequaloneDist)==[n_a,N_i])
                jequaloneDist=reshape(jequaloneDist,[prod(n_a),N_i]);
                idiminj1dist=1;
            end
        else
            if all(size(jequaloneDist)==[n_a,n_z])
                jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(n_z)]);
                idiminj1dist=0;
            elseif all(size(jequaloneDist)==[n_a,n_z,N_i])
                jequaloneDist=reshape(jequaloneDist,[prod(n_a),prod(n_z),N_i]);
                idiminj1dist=1;
            end
        end
    end
end

for ii=1:N_i

    % First set up simoptions
    if exist('simoptions','var')
        simoptions_temp=PType_Options(simoptions,Names_i,ii);
        if ~isfield(simoptions_temp,'verbose')
            simoptions_temp.verbose=0;
        end
        if ~isfield(simoptions_temp,'verboseparams')
            simoptions_temp.verboseparams=0;
        end
        if ~isfield(simoptions_temp,'ptypestorecpu')
            simoptions_temp.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
        end
    else
        simoptions_temp.verbose=0;
        simoptions_temp.verboseparams=0;
        simoptions_temp.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
    end 
    
    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end
           
    
    Policy_temp=Policy.(Names_i{ii});
    
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
    if isa(pi_z,'struct')
        pi_z_temp=pi_z.(Names_i{ii});
    else
        pi_z_temp=pi_z;
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
    
    if simoptions_temp.verboseparams==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end
    
    if isa(jequaloneDist,'struct')
        if isfield(jequaloneDist,Names_i{ii})
            jequaloneDist_temp=jequaloneDist.(Names_i{ii});
            % jequaloneDist_temp must be of mass one for the codes to work.
            if sum(jequaloneDist_temp(:))~=1
                error(['The jequaloneDist must be of mass one for each type i (it is not for type ',Names_i{ii}])
            end
        else
            if isfinite(N_j_temp)
                sprintf(['ERROR: You must input jequaloneDist for permanent type ', Names_i{ii}, ' \n'])
                dbstack
            end
        end
    else
        % Note: when jequaloneDist is not a structure all ptypes must have the same grids
        if idiminj1dist==0
            jequaloneDist_temp=jequaloneDist;
        else
            if prod(n_z_temp)==0
                jequaloneDist_temp=jequaloneDist(:,ii)/sum(jequaloneDist(:,ii)); % includes renormalizing so mass of one conditional on ptype
            else
                jequaloneDist_temp=jequaloneDist(:,:,ii)/sum(jequaloneDist(:,:,ii)); % includes renormalizing so mass of one conditional on ptype
            end
        end
        if abs(sum(jequaloneDist_temp(:))-1)>10^(-12)
            error(['The jequaloneDist must be of mass one for each type i (it is not for type ',Names_i{ii}])
        end
    end
    
    AgeWeightParamNames_temp=AgeWeightsParamNames;
    if isa(AgeWeightsParamNames,'struct')
        if isfield(AgeWeightsParamNames,Names_i{ii})
            AgeWeightParamNames_temp=AgeWeightsParamNames.(Names_i{ii});
        else
            if isfinite(N_j_temp)
                sprintf(['ERROR: You must input AgeWeightParamNames for permanent type ', Names_i{ii}, ' \n'])
                dbstack
            end
        end
    end
    
    if isfinite(N_j_temp)
        StationaryDist_ii=StationaryDist_FHorz_Case1(jequaloneDist_temp,AgeWeightParamNames_temp,Policy_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,pi_z_temp,Parameters_temp,simoptions_temp);
    else % PType actually allows for infinite horizon as well
        StationaryDist_ii=StationaryDist_Case1(Policy_temp,n_d_temp,n_a_temp,n_z_temp,pi_z_temp,simoptions_temp,Parameters_temp); % EntryExitParams not yet supported (is on my to-do list)
    end
    
    if simoptions_temp.ptypestorecpu==1
        StationaryDist.(Names_i{ii})=gather(StationaryDist_ii);
    else
        StationaryDist.(Names_i{ii})=StationaryDist_ii;
    end
    
end

StationaryDist.ptweights=reshape(Parameters.(PTypeDistParamNames{:}),[],1); % reshape is to make sure this is a column vector

end
