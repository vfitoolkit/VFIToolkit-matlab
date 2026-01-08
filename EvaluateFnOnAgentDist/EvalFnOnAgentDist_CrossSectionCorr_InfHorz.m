function CrossSectionCorr=EvalFnOnAgentDist_CrossSectionCorr_InfHorz(StationaryDist, Policy, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid,simoptions)
% Evaluates the cross-sectional correlation between every possible pair from FnsToEvaluate
% eg. if you give a FnsToEvaluate with three functions you will get nine cross-sectional correlations; with two function you get four.
%
% Since they are calculated anyway as intermediate steps,
% Also reports the Mean and Standard Deviation of every function
% And the Covariance of every pair of functions.


%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    % Model solution
    simoptions.gridinterplayer=0;
    % Model setup
    simoptions.experienceasset=0;
    simoptions.inheritanceasset=0;
    simoptions.n_e=0;
    simoptions.n_semiz=0;
    % Internal options
    simoptions.alreadygridvals=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    % Model solution
    if ~isfield(simoptions, 'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    % Model setup
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
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
    if ~isfield(simoptions, 'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
end

if isfield(simoptions,'conditionalrestrictions')
    warning('Have not yet implemented simoptions.conditionalrestrictions for CorrTransProbs_InfHorz, ask on forum if you need this')
end

%%
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

l_daprime=size(Policy,1);
if simoptions.gridinterplayer==1
    l_daprime=l_daprime-1;
end
a_gridvals=CreateGridvals(n_a,a_grid,1);
z_gridvals=CreateGridvals(n_z,z_grid,1);

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
StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

CrossSectionCorr=struct();

PolicyValues=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid,a_grid,simoptions);
permuteindexes=[1+(1:1:(l_a+l_z)),1];
PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]


% Report output by name, but also create the covariance matrix and the correlation matrix
CrossSectionCorr.CovarianceMatrix=zeros(length(FnsToEvaluate),length(FnsToEvaluate));
CrossSectionCorr.CorrelationMatrix=zeros(length(FnsToEvaluate),length(FnsToEvaluate));

%% Calculate all the cross-sectional correlations, note that this creates the 'upper triangular' part
for ff1=1:length(FnsToEvaluate)
    FnToEvaluateParamsCell1=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff1).Names);
    Values1=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff1}, FnToEvaluateParamsCell1,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
    Values1=reshape(Values1,[N_a*N_z,1]);

    Mean1=sum(Values1.*StationaryDistVec);
    StdDev1=sqrt(sum(StationaryDistVec.*((Values1-Mean1.*ones(N_a*N_z,1)).^2)));

    CrossSectionCorr.(AggVarNames{ff1}).Mean=Mean1;
    CrossSectionCorr.(AggVarNames{ff1}).StdDeviation=StdDev1;

    for ff2=ff1:length(FnsToEvaluate)
        if ff1==ff2
            CrossSectionCorr.(AggVarNames{ff1}).(AggVarNames{ff2})=1;

            % and matrix version
            CrossSectionCorr.CovarianceMatrix(ff1,ff2)=StdDev1^2;
            CrossSectionCorr.CorrelationMatrix(ff1,ff2)=1;
        else
            FnToEvaluateParamsCell2=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff2).Names);
            Values2=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff2}, FnToEvaluateParamsCell2,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
            Values2=reshape(Values2,[N_a*N_z,1]);

            Mean2=sum(Values2.*StationaryDistVec);
            StdDev2=sqrt(sum(StationaryDistVec.*((Values2-Mean2.*ones(N_a*N_z,1)).^2)));

            CoVar=sum((Values1-Mean1*ones(N_a*N_z,1,'gpuArray')).*(Values2-Mean2*ones(N_a*N_z,1,'gpuArray')).*StationaryDistVec);
            Corr=CoVar/(StdDev1*StdDev2);

            % Store them
            CrossSectionCorr.(AggVarNames{ff1}).CovarianceWith.(AggVarNames{ff2})=CoVar;
            CrossSectionCorr.(AggVarNames{ff1}).CorrelationWith.(AggVarNames{ff2})=Corr;

            % and matrix version
            CrossSectionCorr.CovarianceMatrix(ff1,ff2)=CoVar;
            CrossSectionCorr.CorrelationMatrix(ff1,ff2)=Corr;
        end
    end
end


%% Just to make them easier to find, fill in the 'lower triangular' part
for ff1=1:length(FnsToEvaluate)
    for ff2=1:ff1-1
        CrossSectionCorr.(AggVarNames{ff1}).CovarianceWith.(AggVarNames{ff2})=CrossSectionCorr.(AggVarNames{ff2}).CovarianceWith.(AggVarNames{ff1});
        CrossSectionCorr.(AggVarNames{ff1}).CorrelationWith.(AggVarNames{ff2})=CrossSectionCorr.(AggVarNames{ff2}).CorrelationWith.(AggVarNames{ff1});

        % and matrix version
        CrossSectionCorr.CovarianceMatrix(ff1,ff2)=CrossSectionCorr.CovarianceMatrix(ff2,ff1);
        CrossSectionCorr.CorrelationMatrix(ff1,ff2)=CrossSectionCorr.CorrelationMatrix(ff2,ff1);
    end
end

CrossSectionCorr.Notes='The CovarianceMatrix and Correlation matrix are essentially duplicating the individual correlations and covariances, but depending on what you want to do the matrix or the individual named pairs might be easier to use so both are created.';

end
