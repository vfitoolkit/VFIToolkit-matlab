function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_InfHorz_PType(Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,Names_i,d_grid, a_grid, z_grid, simoptions)
% The values of the FnsToEvaluate on the grid, with permanent types. Calls EvalFnOnAgentDist_ValuesOnGrid_InfHorz() for each
% permanent type.
%
% Inputs follow ValueFnIter_InfHorz_PType: anything that depends on the permanent type is given as a structure with a field per
% type (Parameters, n_z, z_grid, simoptions fields, FnsToEvaluate fields) or with one value per type where that form is accepted.
% Names_i is the cell of type names, or just the number of types.
%
% Output:
%   ValuesOnGrid.(fnname).(typename)   n_a-by-n_z (n_a if there is no z), the output of EvalFnOnAgentDist_ValuesOnGrid_InfHorz
%                                      for that type (only for the types to which the function is relevant)

if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
    Names_i=cell(1,N_i);
    for ii=1:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

if ~isstruct(FnsToEvaluate)
    error('You can only use PType when FnsToEvaluate is a structure')
end
FnNames=fieldnames(FnsToEvaluate);
numFnsToEvaluate=length(FnNames);

if ~exist('simoptions','var')
    simoptions.verbose=0;
    simoptions.verboseparams=0;
else
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=0;
    end
end

ValuesOnGrid=struct();
for ii=1:N_i
    iistr=Names_i{ii};
    simoptions_temp=PType_Options(simoptions,iistr); % Note: already check for existence of simoptions and created it if it was not inputted
    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end
    PolicyIndexes_temp=gpuArray(Policy.(iistr)); % (in case the solutions are stored on the cpu)

    %% Go through everything which might be dependent on fixed type (PType)
    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);
    % Exogenous shocks
    [n_z_temp,z_grid_temp,~,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,[],simoptions_temp,3);
    % Parameters
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);
    if simoptions_temp.verboseparams==1
        fprintf('Parameter values for the current permanent type \n')
        Parameters_temp
    end

    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=length(n_d_temp);
    end
    l_a_temp=length(n_a_temp);
    if prod(n_z_temp)==0
        l_z_temp=0;
    else
        l_z_temp=length(n_z_temp);
    end

    % Which of the FnsToEvaluate are relevant to this type (kept as a structure)
    [FnsToEvaluate_temp,~,~,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    if sum(FnsAndPTypeIndicator_ii)==0
        continue % none of the FnsToEvaluate are relevant to this type
    end

    ValuesOnGrid_ii=EvalFnOnAgentDist_ValuesOnGrid_InfHorz(PolicyIndexes_temp,FnsToEvaluate_temp,Parameters_temp,[],n_d_temp,n_a_temp,n_z_temp,d_grid_temp,a_grid_temp,z_grid_temp,simoptions_temp);
    for ff=1:numFnsToEvaluate
        if FnsAndPTypeIndicator_ii(ff)==1
            ValuesOnGrid.(FnNames{ff}).(iistr)=ValuesOnGrid_ii.(FnNames{ff});
        end
    end
end

end
