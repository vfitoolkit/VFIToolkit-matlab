function AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist,Policy, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)
% simoptions.whichcombos (a vector of zeros/ones, one per FnsToEvaluate) selects which functions are evaluated; the Mean of a skipped
% function is NaN. Default all ones. Intended for calibration/estimation, which only needs the targeted aggregates.
% simoptions.lowmemory is accepted and ignored (2026-10-06: the lowmemory=1 branch, which looped over age, was removed; it had never
% been exercised and saved only the one values block, the size of the agent distribution, that lowmemory=0 holds per function).

if ~exist('simoptions','var')
    simoptions.lowmemory=0; % accepted and ignored: the whole (a,z,j) block of values for one function is held (same size as the agent distribution), there is nothing left for a low-memory variant to cut
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    % Largely just for internal use
    simoptions.parallel=1+(gpuDeviceCount>0);
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0; % =1 when calling as a subcommand
    simoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
else
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0;
    end
    % Model setup
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    % Largely just for internal use
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0; % =1 when calling as a subcommand
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
    end
end

if simoptions.parallel==1
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_CPU(StationaryDist,Policy, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
    return
end

%%
l_a=length(n_a);
N_a=prod(n_a);

%% Exogenous shock grids
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate
[n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters);

%% Implement new way of handling FnsToEvaluate
% Figure out l_daprime from Policy
l_daprime=size(Policy,1)-2*simoptions.gridinterplayer; % Note: simoptions.gridinterplayer=1 means that PolicyIndexes has an extra 'second layer index' and 'flag'

% Note: l_z includes e and semiz (when appropriate)
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        AggVarNames=simoptions.AggVarNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end

%% simoptions.whichcombos: which FnsToEvaluate to compute (a vector of zeros/ones of length numFnsToEvaluate; skipped Means are NaN)
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(length(FnsToEvaluate),1); % whichcombos here is [nFns, 1]
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ~(isvector(whichcombos) && numel(whichcombos)==length(FnsToEvaluate))
        error(['simoptions.whichcombos must be a vector of length ',num2str(length(FnsToEvaluate)),' (number of FnsToEvaluate)'])
    end
    whichcombos=double(whichcombos(:));
end

%%
a_gridvals=CreateGridvals(n_a,a_grid,1);


%%
if N_z==0
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

    StationaryDist=reshape(StationaryDist,[N_a,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1]); % (N_a,N_j,l_daprime)


    for ff=1:length(FnsToEvaluate)
        if whichcombos(ff)==0 % this function is not wanted
            continue
        end
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(ff).Names)
            ParamCell=cell(0,1);
        else
            % Create a matrix containing all the return function parameters (in order).
            % Each column will be a specific parameter with the values at every age.
            FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

            nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

            ParamCell=cell(nFnToEvaluateParams,1);
            for ii=1:nFnToEvaluateParams
                ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-1)}; % (a,j,l_d+l_a), so we want j to be after N_a
            end
        end

        Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},ParamCell,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]);
        AggVars(ff)=sum(sum(sum(Values.*StationaryDist)));
    end


else % N_z

    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

    StationaryDist=reshape(StationaryDist,[N_a,N_z,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)

    for ff=1:length(FnsToEvaluate)
        if whichcombos(ff)==0 % this function is not wanted
            continue
        end
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(ff).Names)
            ParamCell=cell(0,1);
        else
            % Create a matrix containing all the return function parameters (in order).
            % Each column will be a specific parameter with the values at every age.
            FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

            nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

            ParamCell=cell(nFnToEvaluateParams,1);
            for ii=1:nFnToEvaluateParams
                ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-2)}; % (a,z,j,l_d+l_a), so we want j to be after N_a and N_z
            end
        end

        Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},ParamCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J);
        AggVars(ff)=sum(sum(sum(Values.*StationaryDist)));
    end

end
AggVars(whichcombos==0)=NaN; % the functions that whichcombos skipped



%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    AggVars2=AggVars;
    clear AggVars
    AggVars=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        AggVars.(AggVarNames{ff}).Mean=AggVars2(ff);
    end
end


end
