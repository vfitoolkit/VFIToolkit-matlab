function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_subfn(PolicyValues, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, a_grid, z_grid, simoptions,keepoutputasmatrix)
% subfn version is GPU only, and uses PolicyValues instead of PolicyIndexes
% Still loops over j, I could speed it further by parallel over j

if ~exist('simoptions','var')
    simoptions=struct();
end
if ~exist('keepoutputasmatrix','var')
    keepoutputasmatrix=0;
end

l_a=length(n_a);
N_a=prod(n_a);
N_z=prod(n_z);

a_gridvals=CreateGridvals(n_a,a_grid,1);
z_gridvals=CreateGridvals(n_z,z_grid,1);


%% Exogenous shock grids

% If using e variable, do same for this
if isfield(simoptions,'n_e')

    % Now put e into z as that is easiest way to handle it from now on
    if N_z==0
        z_gridvals=e_gridvals;
        n_z=n_e;
        N_z=prod(n_z);
    else
        z_gridvals=[repmat(z_gridvals,N_e,1),repelem(e_gridvals,N_z,1)];
        n_z=[n_z,n_e];
        N_z=prod(n_z);
    end
end


N_z=prod(n_z);
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end


%% Implement new way of handling FnsToEvaluate
% Figure out l_daprime from Policy
l_daprime=size(PolicyValues,1);


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
if isfield(simoptions,'keepoutputasmatrix')
    if simoptions.keepoutputasmatrix==1
        FnsToEvaluateStruct=0;
    elseif simoptions.keepoutputasmatrix==2
        FnsToEvaluateStruct=2;
    end
end


%% Loop over j
if N_z==0
    ValuesOnGrid=zeros(N_a,length(FnsToEvaluate),'gpuArray');

    for ff=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(ff).Names)
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
        end
        ValuesOnGrid(:,ff)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsVec,PolicyValues,l_daprime,n_a,0,a_gridvals,[]);
    end
else
    ValuesOnGrid=zeros(N_a*N_z,length(FnsToEvaluate),'gpuArray');

    for ff=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(ff).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names));
        end
        ValuesOnGrid(:,ff)=reshape(EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsVec,PolicyValues,l_daprime,n_a,n_z,a_gridvals,z_gridvals),[N_a*N_z,1]);
    end
end


if FnsToEvaluateStruct==1
    ValuesOnGrid2=ValuesOnGrid;
    clear ValuesOnGrid
    ValuesOnGrid=struct();
    if N_z==0
        for ff=1:length(AggVarNames)
            ValuesOnGrid.(AggVarNames{ff})=reshape(ValuesOnGrid2(:,ff),n_a);
            % Change the ordering and size so that ProbDensityFns has same kind of shape as StationaryDist
        end
    else
        for ff=1:length(AggVarNames)
            ValuesOnGrid.(AggVarNames{ff})=reshape(ValuesOnGrid2(:,ff),[n_a,n_z]);
            % Change the ordering and size so that ProbDensityFns has same kind of shape as StationaryDist
        end
    end
elseif FnsToEvaluateStruct==0
    % Change the ordering and size so that ProbDensityFns has same kind of
    % shape as StationaryDist, except first dimension indexes the 'FnsToEvaluate'.
    ValuesOnGrid=permute(ValuesOnGrid,[2,1]);
    if N_z==0
        ValuesOnGrid=reshape(ValuesOnGrid,[length(FnsToEvaluate),n_a]);
    else
        ValuesOnGrid=reshape(ValuesOnGrid,[length(FnsToEvaluate),n_a,n_z]);
    end
elseif FnsToEvaluateStruct==2 % Just a rearranged version of FnsToEvaluateStruct=0 for use internally when length(FnsToEvaluate)==1
    %     ValuesOnGrid=reshape(ValuesOnGrid,[N_a,N_z,N_j]);
    % The output is already in this shape anyway, so no need to actually reshape it at all
end

end
