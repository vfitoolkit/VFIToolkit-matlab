function ProbDensityFns=EvalFnOnAgentDist_ProbDensityFn_InfHorz(StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid,simoptions,EntryExitParamNames)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
%
% simoptions and EntryExitParamNames are optional inputs, only needed when using endogenous entry

%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    % Model solution
    simoptions.gridinterplayer=0;
    % Model setup
    simoptions.experienceasset=0;
    simoptions.experienceassetz=0;
    simoptions.experienceassete=0;
    simoptions.experienceassetze=0;
    simoptions.inheritanceasset=0;
    simoptions.n_e=0;
    simoptions.n_semiz=0;
    % Internal options
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    % Model solution
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    % Model setup
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    if ~isfield(simoptions,'experienceassetz')
        simoptions.experienceassetz=0;
    end
    if ~isfield(simoptions,'experienceassete')
        simoptions.experienceassete=0;
    end
    if ~isfield(simoptions,'experienceassetze')
        simoptions.experienceassetze=0;
    end
    if ~isfield(simoptions,'inheritanceasset')
        simoptions.inheritanceasset=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    % Internal options
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
end


l_a=length(n_a);

n_a=gpuArray(n_a);

N_a=prod(n_a);
a_gridvals=CreateGridvals(n_a,gpuArray(a_grid),1);
% Switch to z_gridvals (folding e and semiz into z if appropriate)
[n_z,z_gridvals,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_InfHorz(n_z,z_grid,simoptions,Parameters);

%%
if isstruct(StationaryDist)
    % Even though Mass is unimportant, still need to deal with 'exit' in PolicyIndexes.
    ProbDensityFns=EvalFnOnAgentDist_ProbDensityFn_InfHorz_Mass(StationaryDist.pdf,StationaryDist.mass, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames,EntryExitParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, 2,simoptions);
    return
end

Policy=gpuArray(Policy);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
PolicyValues=PolicyInd2Val_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid,simoptions);
l_daprime=size(PolicyValues,1);

%% Implement new way of handling FnsToEvaluate
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

%%
StationaryDist=gpuArray(StationaryDist);

if N_z==0
    StationaryDistVec=reshape(StationaryDist,[N_a,1]);

    ProbDensityFns=zeros(N_a,length(FnsToEvaluate),'gpuArray');

    permuteindexes=[1+(1:1:l_a),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,l_d+l_a]

    for ff=1:length(FnsToEvaluate)
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
        Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,[]);
        Values=reshape(Values,[N_a,1]);
        ProbDensityFns(:,ff)=Values.*StationaryDistVec;
    end
else % N_z>0
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

    ProbDensityFns=zeros(N_a*N_z,length(FnsToEvaluate),'gpuArray');

    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a]

    for ff=1:length(FnsToEvaluate)
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
        Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
        Values=reshape(Values,[N_a*N_z,1]);
        ProbDensityFns(:,ff)=Values.*StationaryDistVec;
    end
end

% Normalize to 1 (to make it a pdf)
for ff=1:length(FnsToEvaluate)
    ProbDensityFns(:,ff)=ProbDensityFns(:,ff)/sum(ProbDensityFns(:,ff));
end

% When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be
% 0) we get 'NaN'. Just eliminate those.
ProbDensityFns(isnan(ProbDensityFns))=0;

%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    ProbDensityFns2=ProbDensityFns'; % Note the transpose
    clear ProbDensityFns
    ProbDensityFns=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    if N_z==0
        for ff=1:length(AggVarNames)
            ProbDensityFns.(AggVarNames{ff})=reshape(ProbDensityFns2(ff,:),[n_a,1]);
        end
    else
        for ff=1:length(AggVarNames)
            ProbDensityFns.(AggVarNames{ff})=reshape(ProbDensityFns2(ff,:),[n_a,n_z]);
        end
    end
else
    % Change the ordering and size so that ProbDensityFns has same kind of
    % shape as StationaryDist, except first dimension indexes the 'FnsToEvaluate'.
    ProbDensityFns=ProbDensityFns';
    if N_z==0
        ProbDensityFns=reshape(ProbDensityFns,[length(FnsToEvaluate),n_a]);
    else
        ProbDensityFns=reshape(ProbDensityFns,[length(FnsToEvaluate),n_a,n_z]);
    end
end

end
