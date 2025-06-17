function StationaryDist=StationaryDist_FHorz_FieldExp_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,Names_i,pi_z,Parameters, TreatmentAgeRange, TreatmentDuration,simoptions)
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
            simoptions_temp.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
        end
    else
        simoptions_temp.verbose=0;
        simoptions_temp.verboseparams=0;
        simoptions_temp.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    end 
    
    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end
           
    % Because we are looking at the field experiment we need the control-group policy
    Policy_control=Policy.control.(Names_i{ii});
    % And all of the treatment-group policies (one for each initial age)
    Policy_treatment=struct();
    treatmentagenames=fieldnames(Policy);
    for nn=1:length(treatmentagenames)
        if ~strcmp(treatmentagenames{nn},'control')
            Policy_treatment.(treatmentagenames{nn})=Policy.(treatmentagenames{nn}).(Names_i{ii});
        end
    end
    
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
    
    jequaloneDist_temp=jequaloneDist;
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
    
    % We want to get the stationary dist in kron form 
    simoptions_temp.outputkron=1;
    % First, get the whole stationary dist of the control group
    if isfinite(N_j_temp)
        StationaryDist_Control_ii=StationaryDist_FHorz_Case1(jequaloneDist_temp,AgeWeightParamNames_temp,Policy_control,n_d_temp,n_a_temp,n_z_temp,N_j_temp,pi_z_temp,Parameters_temp,simoptions_temp);
    end
    % Now, create the stationary dist of the treatment group
    if isfinite(N_j_temp)
        StationaryDist_Treat_ii=StationaryDist_FHorz_FieldExp_Treatment(StationaryDist_Control_ii,AgeWeightParamNames_temp,Policy_control,n_d_temp,n_a_temp,n_z_temp,N_j_temp,pi_z_temp,Parameters_temp, TreatmentAgeRange, TreatmentDuration,simoptions_temp);
    end

    check_ze=length(size(StationaryDist_Control_ii)); % Used to determine whether using z/e
    
    % Cut the control group to just the treatment age range
    if check_ze==2 % no z, no e
        StationaryDist_Control_ii=StationaryDist_Control_ii(:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif check_ze==3 % one of z or e
        StationaryDist_Control_ii=StationaryDist_Control_ii(:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif check_ze==4 % z and e
        StationaryDist_Control_ii=StationaryDist_Control_ii(:,:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    end
    % As part of this, renormalize mass to 1
    StationaryDist_Control_ii=StationaryDist_Control_ii./sum(StationaryDist_Control_ii(:));
    
        
    % UnKron and put it all into the output structure
    if check_ze==2 % no z, no e
        StationaryDist_Control_ii=reshape(StationaryDist_Control_ii,[n_a,TreatmentAgeRange(2)-TreatmentAgeRange(1)+TreatmentDuration]);
    elseif check_ze==3
        if prod(N_z_temp)==0 % just e
            StationaryDist_Control_ii=reshape(StationaryDist_Control_ii,[n_a,simoptions_temp.n_e,TreatmentAgeRange(2)-TreatmentAgeRange(1)+TreatmentDuration]);
        else % just z
            StationaryDist_Control_ii=reshape(StationaryDist_Control_ii,[n_a,n_z,TreatmentAgeRange(2)-TreatmentAgeRange(1)+TreatmentDuration]);
        end
    elseif check_ze==4 % both z and e
        StationaryDist_Control_ii=reshape(StationaryDist_Control_ii,[n_a,n_z,simoptions_temp.n_e,TreatmentAgeRange(2)-TreatmentAgeRange(1)+TreatmentDuration]);
    end
    if simoptions_temp.ptypestorecpu==1
        StationaryDist.control.(Names_i{ii})=gather(StationaryDist_Control_ii);
    else
        StationaryDist.control.(Names_i{ii})=StationaryDist_Control_ii;
    end
    
    if check_ze==2 % no z, no e
        StationaryDist_Treat_ii=reshape(StationaryDist_Treat_ii,[n_a,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1]);
    elseif check_ze==3
        if prod(N_z_temp)==0 % just e
            StationaryDist_Treat_ii=reshape(StationaryDist_Treat_ii,[n_a,simoptions_temp.n_e,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1]);
        else % just z
            StationaryDist_Treat_ii=reshape(StationaryDist_Treat_ii,[n_a,n_z,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1]);
        end
    elseif check_ze==4 % both z and e
        StationaryDist_Treat_ii=reshape(StationaryDist_Treat_ii,[n_a,n_z,simoptions_temp.n_e,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1]);
    end
    if simoptions_temp.ptypestorecpu==1
        StationaryDist.treatment.(Names_i{ii})=gather(StationaryDist_Treat_ii);
    else
        StationaryDist.treatment.(Names_i{ii})=StationaryDist_Treat_ii;
    end
    
end

StationaryDist.ptweights=reshape(Parameters.(PTypeDistParamNames{:}),[],1); % reshape is to make sure this is a column vector

end
