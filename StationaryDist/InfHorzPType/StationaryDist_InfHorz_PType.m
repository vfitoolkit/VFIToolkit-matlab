function StationaryDist=StationaryDist_InfHorz_PType(PTypeDistParamNames,Policy,n_d,n_a,n_z,Names_i,pi_z,Parameters,simoptions)
% Allows for different permanent (fixed) types of agent.
% See ValueFnIter_InfHorz_PType for general idea.
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

%%
if ~exist('simoptions','var')
    error('You must input simoptions, you can always set simoptions=struct().')
end

%% Check inputs
if ~iscell(PTypeDistParamNames)
    error('PTypeDistParamNames should be a cell, it is not')
end
if abs(sum(Parameters.(PTypeDistParamNames{1}))-1)>10^(-15)
    warning('The permanent type mass weights must sum to one (PTypeDistParamNames points to weights that do not sum to one)')
end



%%
for ii=1:N_i
    iistr=Names_i{ii};
    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,iistr);
    if ~isfield(simoptions_temp,'verbose')
        simoptions_temp.verbose=0;
    end
    if ~isfield(simoptions_temp,'verboseparams')
        simoptions_temp.verboseparams=0;
    end
    if ~isfield(simoptions_temp,'ptypestorecpu')
        simoptions_temp.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
    end

    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end


    Policy_temp=Policy.(iistr);

    %% Go through everything which might be dependent on fixed type (PType)
    if isstruct(n_d)
        n_d_temp=n_d.(iistr);
    else
        n_d_temp=n_d;
    end
    if isstruct(n_a)
        n_a_temp=n_a.(iistr);
    else
        n_a_temp=n_a;
    end

    % Exogenous shocks
    [n_z_temp,~,pi_z_temp,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,[],pi_z,simoptions_temp,3);

    % Parameters
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);

    if simoptions_temp.verboseparams==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end

    StationaryDist_ii=StationaryDist_InfHorz(Policy_temp,n_d_temp,n_a_temp,n_z_temp,pi_z_temp,simoptions_temp,Parameters_temp); % EntryExitParams not yet supported (is on my to-do list)

    if simoptions_temp.ptypestorecpu==1
        StationaryDist.(iistr)=gather(StationaryDist_ii);
    else
        StationaryDist.(iistr)=StationaryDist_ii;
    end

end


if length(Parameters.(PTypeDistParamNames{1}))==N_i
    StationaryDist.ptweights=reshape(Parameters.(PTypeDistParamNames{1}),[],1); % reshape is to make sure this is a column vector
else
    error('Parameter for PTypeDistParamNames does not have the same number of permanent types as N_i/Names_i \n')
end


end
