function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1_PType(Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, simoptions)
% Allows for different permanent (fixed) types of agent.
% See ValueFnIter_PType for general idea.
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

% RIGHT NOW THIS ValuesOnGrid ONLY WORKS WHEN ALL AGENTS ARE ON THE SAME GRID
N_a=prod(n_a);
N_z=prod(n_z);
if ~isstruct(FnsToEvaluate)
    ValuesOnDist_Kron=nan(numFnsToEvaluate,N_a,N_z,N_j,'gpuArray');
end
ValuesOnGrid=struct();

% Set default of grouping all the PTypes together when reporting statistics
if ~exist('simoptions','var')
    simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    simoptions.verbose=0;
    simoptions.verboseparams=0;
else
    if ~isfield(simoptions,'ptypestorecpu')
        simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=1;
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=0;
    end
end
% Note: pass to subcommand EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(), so no need to handle alreadygridvals and the like as those can be done there.


%%
for ii=1:N_i % First set up simoptions
    iistr=Names_i{ii};
    simoptions_temp=PType_Options(simoptions,iistr);  % Note: already check for existence of simoptions and created it if it was not inputted

    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end

    if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
        PolicyIndexes_temp=gpuArray(Policy.(iistr));
    else
        PolicyIndexes_temp=Policy.(iistr);
    end
    if isfield(simoptions_temp,'parallel')
        if simoptions.parallel~=2
            PolicyIndexes_temp=gather(PolicyIndexes_temp);
        end
    end


    %% Go through everything which might be dependent on fixed type (PType)

    if isstruct(N_j)
        N_j_temp=N_j.(iistr);
    else
        N_j_temp=N_j;
    end

    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);

    % Exogenous shocks
    [n_z_temp,z_grid_temp,~,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,[],simoptions_temp,3);

    % Parameters
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);

    if simoptions_temp.verboseparams==1
        fprintf('Parameter values for the current permanent type \n')
        Parameters_temp
    end


    % Figure out which functions are actually relevant to the present
    % PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are necessarily the same.
    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=1;
    end
    l_a_temp=length(n_a_temp);
    l_z_temp=length(n_z_temp);
    [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,~]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);

    ValuesOnGrid_ii=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, simoptions_temp);

    n_ze_temp=[];
    if isfield(simoptions_temp,'n_semiz') && prod(simoptions_temp.n_semiz)>0
        n_ze_temp=[n_ze_temp,simoptions_temp.n_semiz];
    end
    if prod(n_z_temp)>0
        n_ze_temp=[n_ze_temp,n_z_temp];
    end
    if isfield(simoptions_temp,'n_e') && prod(simoptions_temp.n_e)>0
        n_ze_temp=[n_ze_temp,simoptions_temp.n_e];
    end

    if isempty(n_ze_temp) % no exogenous states
        if isstruct(FnsToEvaluate)
            FnNames=fieldnames(FnsToEvaluate);
            for kk=1:numFnsToEvaluate
                jj=WhichFnsForCurrentPType(kk);
                if jj>0
                    if simoptions.ptypestorecpu==0
                        ValuesOnGrid.(FnNames{kk}).(Names_i{ii})=reshape(ValuesOnGrid_ii.(FnNames{kk}),[n_a_temp,N_j_temp]);
                    else
                        ValuesOnGrid.(FnNames{kk}).(Names_i{ii})=gather(reshape(ValuesOnGrid_ii.(FnNames{kk}),[n_a_temp,N_j_temp]));
                    end
                end
            end

        else % Note: this only works when all agents use same grid
            for kk=1:numFnsToEvaluate
                jj=WhichFnsForCurrentPType(kk);
                if jj>0
                    ValuesOnDist_Kron(kk,:,:,:)=ValuesOnGrid_ii(jj,:,:,:);
                end
            end
            if simoptions.ptypestorecpu==0
                ValuesOnGrid.(Names_i{ii})=reshape(ValuesOnDist_Kron,[numFnsToEvaluate,n_a_temp,N_j_temp]);
            else
                ValuesOnGrid.(Names_i{ii})=gather(reshape(ValuesOnDist_Kron,[numFnsToEvaluate,n_a_temp,N_j_temp]));
            end
        end
    else
        if isstruct(FnsToEvaluate)
            FnNames=fieldnames(FnsToEvaluate);
            for kk=1:numFnsToEvaluate
                jj=WhichFnsForCurrentPType(kk);
                if jj>0
                    if simoptions.ptypestorecpu==0
                        ValuesOnGrid.(FnNames{kk}).(Names_i{ii})=reshape(ValuesOnGrid_ii.(FnNames{kk}),[n_a_temp,n_ze_temp,N_j_temp]);
                    else
                        ValuesOnGrid.(FnNames{kk}).(Names_i{ii})=gather(reshape(ValuesOnGrid_ii.(FnNames{kk}),[n_a_temp,n_ze_temp,N_j_temp]));
                    end
                end
            end

        else % Note: this only works when all agents use same grid
            for kk=1:numFnsToEvaluate
                jj=WhichFnsForCurrentPType(kk);
                if jj>0
                    ValuesOnDist_Kron(kk,:,:,:)=ValuesOnGrid_ii(jj,:,:,:);
                end
            end
            if simoptions.ptypestorecpu==0
                ValuesOnGrid.(Names_i{ii})=reshape(ValuesOnDist_Kron,[numFnsToEvaluate,n_a_temp,n_ze_temp,N_j_temp]);
            else
                ValuesOnGrid.(Names_i{ii})=gather(reshape(ValuesOnDist_Kron,[numFnsToEvaluate,n_a_temp,n_ze_temp,N_j_temp]));
            end
        end
    end
end

end
