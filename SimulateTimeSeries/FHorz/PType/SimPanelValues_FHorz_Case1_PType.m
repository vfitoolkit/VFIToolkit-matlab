function SimPanelValues=SimPanelValues_FHorz_Case1_PType(jequaloneDist,PTypeDistParamNames,Policy,FnsToEvaluate,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_grid,pi_z, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist.
%
% simoptions.verbose=1 will give feedback
% simoptions.verboseparams=1 will give further feedback on the param values of each permanent type
%
% Rest of this description describes how those inputs not already used for
% ValueFnIter_PType or StationaryDist_PType should be set up.
%
% jequaloneDist can either be same for all permanent types, or must be passed as a structure.
% AgeWeightParamNames is either same for all permanent types, or must be passed as a structure.
%
% The stationary distribution be a structure and will contain both the
% weights/distribution across the permanent types, as well as a pdf for the
% stationary distribution of each specific permanent type.
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

if isstruct(FnsToEvaluate)
    numFnsToEvaluate=length(fieldnames(FnsToEvaluate));
else
    numFnsToEvaluate=length(FnsToEvaluate);
end
FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i,'gpuArray');

% Obviously simoptions.grouptypesforstats is not relevant here

FnNames=fieldnames(FnsToEvaluate);

%%
if ~exist('simoptions','var')
    simoptions.numbersims=10^5;
    simoptions.simperiods=N_j;
    simoptions.verbose=0;
    simoptions.verboseparams=0;
    simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    simoptions.gridinterplayer=0;
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0;
else
    if ~isfield(simoptions,'numbersims')
        simoptions.numbersims=10^5;
    end
    if ~isfield(simoptions,'simperiods')
        simoptions.simperiods=N_j;
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=100;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=100;
    end
    if ~isfield(simoptions,'ptypestorecpu')
        simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
end
simoptions.outputasstructure=0; % SimPanelValues as matrix

% Need to figure out how many simulations to do for each PType.
% Make them perfectly representative of the PType masses.
PType_numbersims=floor(Parameters.(PTypeDistParamNames{1})*simoptions.numbersims);
% This will not perfectly add up to the right number of sims (floor means it will be a slightly to few)
ExtraSims=simoptions.numbersims-sum(PType_numbersims);
% I just arbitrarily add them to the first PTypes. Your simulation should
% anyway be big enough for this to be irrelevant. (I should probably add
% them randomly, but cant be bothered right now. But otherwise if I don't
% make this random the sample won't satisfy properties of arandom sample)
PType_numbersims(1:ExtraSims)=PType_numbersims(1:ExtraSims)+1;

%% Deal with jequaloneDist
[jequaloneDist,idiminj1dist,Parameters]=jequaloneDist_PType(jequaloneDist,Parameters,simoptions,n_a,n_z,N_i,Names_i,PTypeDistParamNames,0);


%%
SimPanelValues=nan(length(FnsToEvaluate),simoptions.simperiods,simoptions.numbersims);
for ii=1:N_i
    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,Names_i,ii); % Note: already check for existence of simoptions and created it if it was not inputted

    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end

    simoptions_temp.numbersims=PType_numbersims(ii); % How many simulations to do for each PType

    Policy_temp=Policy.(Names_i{ii});

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

    if isstruct(n_d)
        n_d_temp=n_d.(Names_i{ii});
    else
        n_d_temp=n_d;
    end
    if isstruct(n_a)
        n_a_temp=n_a.(Names_i{ii});
    else
        n_a_temp=n_a;
    end
    if isstruct(n_z)
        n_z_temp=n_z.(Names_i{ii});
    else
        n_z_temp=n_z;
    end
    if isstruct(N_j)
        N_j_temp=N_j.(Names_i{ii});
    else
        N_j_temp=N_j;
    end
    if isstruct(d_grid)
        d_grid_temp=d_grid.(Names_i{ii});
    else
        d_grid_temp=d_grid;
    end
    if isstruct(a_grid)
        a_grid_temp=a_grid.(Names_i{ii});
    else
        a_grid_temp=a_grid;
    end


    %% Exogenous shocks
    if isstruct(z_grid)
        z_grid_temp=z_grid.(Names_i{ii});
    else
        nn=size(z_grid,ndims(z_grid));
        if nn==N_i
            otherdims = repmat({':'},1,ndims(z_grid)-1);
            z_grid_temp=z_grid(otherdims{:},ii);
        else
            z_grid_temp=z_grid;
        end
    end
    if isstruct(pi_z)
        pi_z_temp=pi_z.(Names_i{ii});
    else
        nn=size(pi_z,ndims(pi_z));
        if nn==N_i
            otherdims = repmat({':'},1,ndims(pi_z)-1);
            pi_z_temp=pi_z(otherdims{:},ii);
        else
            pi_z_temp=pi_z;
        end
    end

    % e
    if isfield(simoptions_temp,'n_e')
        % If simoptions_temp.e_grid is a structure that was already dealt with by PType_Options() command
        if ~isstruct(simoptions.e_grid)
            % So just need to check if last dimension is of length N_i
            nn=size(simoptions_temp.e_grid,ndims(simoptions_temp.e_grid));
            if nn==N_i
                otherdims = repmat({':'},1,ndims(simoptions_temp.e_grid)-1);
                simoptions_temp.e_grid=simoptions_temp.e_grid(otherdims{:},ii);
            end
        end
        % If simoptions_temp.pi_semiz is a structure that was already dealt with by PType_Options() command
        if ~isstruct(simoptions.pi_e)
            % So just need to check if last dimension is of length N_i
            nn=size(simoptions_temp.pi_e,ndims(simoptions_temp.pi_e));
            if nn==N_i
                otherdims = repmat({':'},1,ndims(simoptions_temp.pi_e)-1);
                simoptions_temp.pi_e=simoptions_temp.pi_e(otherdims{:},ii);
            end
        end
    end

    % semiz
    if isfield(simoptions_temp,'n_semiz')
        % If simoptions_temp.semiz_grid is a structure that was already dealt with by PType_Options() command
        if ~isstruct(simoptions.semiz_grid)
            % So just need to check if last dimension is of length N_i
            nn=size(simoptions_temp.semiz_grid,ndims(simoptions_temp.semiz_grid));
            if nn==N_i
                otherdims = repmat({':'},1,ndims(simoptions_temp.semiz_grid)-1);
                simoptions_temp.semiz_grid=simoptions_temp.semiz_grid(otherdims{:},ii);
            end
        end
        % Might use SemiExoShockFn or pi_semiz, if the later we need to deal with it
        if isfield(simoptions_temp,'pi_semiz')
            % If simoptions_temp.pi_semiz is a structure that was already dealt with by PType_Options() command
            if ~isstruct(simoptions.pi_semiz)
                % So just need to check if last dimension is of length N_i
                nn=size(simoptions_temp.pi_semiz,ndims(simoptions_temp.pi_semiz));
                if nn==N_i
                    otherdims = repmat({':'},1,ndims(simoptions_temp.pi_semiz)-1);
                    simoptions_temp.pi_semiz=simoptions_temp.pi_semiz(otherdims{:},ii);
                end
            end
        end
    end


    %% Parameters
    % Parameters are allowed to be given as structure, or as vector/matrix
    % (in terms of their dependence on permanent type). So go through each of
    % these in term.
    Parameters_temp=Parameters;
    FullParamNames=fieldnames(Parameters);
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isstruct(Parameters.(FullParamNames{kField})) % Check for permanent type in structure form
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
        fprintf('Parameter values for the current permanent type \n')
        Parameters_temp
    end


    %% jequaloneDist
    if isa(jequaloneDist,'struct')
        if isfield(jequaloneDist,Names_i{ii})
            jequaloneDist_temp=jequaloneDist.(Names_i{ii});
            % jequaloneDist_temp must be of mass one for the codes to work.
            if abs(sum(jequaloneDist_temp(:))-1)>10^(-15) % jequaloneDist_temp(:))~=1, but allowing for small numerical errors
                fprintf('Info for following error: sum(jequaloneDist_temp(:))-1=%8.16f (should be zero) \n', sum(jequaloneDist_temp(:))-1)
                error(['The jequaloneDist must be of mass one for each type i (it is not for type ',Names_i{ii}])
            end
        else
            if isfinite(N_j_temp)
                error(['You must input a jequaloneDist for permanent type ', Names_i{ii}, ' \n'])
            end
        end
    else
        % Note: when jequaloneDist is not a structure all ptypes must have the same grids
        if idiminj1dist==0 % ptype is not a dimension
            jequaloneDist_temp=jequaloneDist;
        else % idminj1dist==1
            % ptype is a dimension, so need to get the jequaloneDist for ii and also normalize mass conditional on ptype to be one
            if ndims(jequaloneDist)==5 % has all three of semiz,z,e [other two are a and i]
                jequaloneDist_temp=jequaloneDist(:,:,:,:,ii)/sum(sum(jequaloneDist(:,:,:,:,ii))); % includes renormalizing so mass of one conditional on ptype
            elseif ndims(jequaloneDist)==4 % has two of semiz,z,e
                jequaloneDist_temp=jequaloneDist(:,:,:,ii)/sum(sum(jequaloneDist(:,:,:,ii))); % includes renormalizing so mass of one conditional on ptype
            elseif ndims(jequaloneDist)==3 % has one of semiz,z,e
                jequaloneDist_temp=jequaloneDist(:,:,ii)/sum(sum(jequaloneDist(:,:,ii))); % includes renormalizing so mass of one conditional on ptype
            elseif ndims(jequaloneDist)==2 % has none of semiz,z,e
                jequaloneDist_temp=jequaloneDist(:,ii)/sum(jequaloneDist(:,ii)); % includes renormalizing so mass of one conditional on ptype
            end
        end
        if abs(sum(jequaloneDist_temp(:))-1)>10^(-12)
            error(['The jequaloneDist must be of mass one for each type i (it is not for type ',Names_i{ii}, ' \n'])
        end
    end

    %%
    % Figure out which functions are actually relevant to the present PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluate and FnsToEvaluateFnParamNames are necessarily the same.
    % Allows for FnsToEvaluate as structure.
    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=1;
    end
    l_a_temp=length(n_a_temp);
    l_z_temp=length(n_z_temp);
    [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;


    simoptions_temp.keepoutputasmatrix=1;
    if simoptions_temp.numbersims>0
        SimPanelValues_ii=SimPanelValues_FHorz_Case1(jequaloneDist_temp,Policy_temp,FnsToEvaluate_temp,Parameters_temp,FnsToEvaluateParamNames_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,z_grid_temp,pi_z_temp, simoptions_temp);
        % simoptions.outputasstructure=0; % SimPanelValues as matrix is set above
    else
        SimPanelValues_ii=[];
    end

    if ii==1
        SimPanelValues(WhichFnsForCurrentPType,:,1:sum(PType_numbersims(1:ii)))=SimPanelValues_ii;
        % I decided to get rid of giving the PType as part of the panel as you can always ask for this using FnsToEvaluate anyway if you actually want it.
    else
        SimPanelValues(WhichFnsForCurrentPType,:,(1+sum(PType_numbersims(1:(ii-1)))):sum(PType_numbersims(1:ii)))=SimPanelValues_ii;
        % I decided to get rid of giving the PType as part of the panel as you can always ask for this using FnsToEvaluate anyway if you actually want it.
    end

end

%% Change the output into a structure
SimPanelValues2=SimPanelValues;
clear SimPanelValues
SimPanelValues=struct();
for ff=1:length(FnNames)
    SimPanelValues.(FnNames{ff})=shiftdim(SimPanelValues2(ff,:,:),1);
end


end



