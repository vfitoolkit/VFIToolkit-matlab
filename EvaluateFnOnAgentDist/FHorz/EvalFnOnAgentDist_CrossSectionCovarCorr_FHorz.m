function CrossSectionCorr=EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz(StationaryDist, Policy, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions)
% Evaluates the cross-sectional correlation between every possible pair from FnsToEvaluate
% eg. if you give a FnsToEvaluate with three functions you will get nine cross-sectional correlations; with two function you get four.
%
% Since they are calculated anyway as intermediate steps,
% Also reports the Mean and Standard Deviation of every function
% And the Covariance of every pair of functions.


%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    % Internal options
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
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
    % Internal options
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
end

if isfield(simoptions,'conditionalrestrictions')
    warning('Have not yet implemented simoptions.conditionalrestrictions for CrossSectionCovarCorr_FHorz so ignoring them, ask on forum if you need this')
end

%%
l_a=length(n_a);
N_a=prod(n_a);

%% Exogenous shock grids
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate
[n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters);

%%
a_gridvals=CreateGridvals(n_a,a_grid,1);

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

%% Setup PolicyValues and reshape StationaryDist
if N_z==0
    StationaryDistVec=reshape(StationaryDist,[N_a*N_j,1]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1]); % (N_a,N_j,l_daprime)
else
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)
end
N_total=length(StationaryDistVec);

CrossSectionCorr=struct();

% Report output by name, but also create the covariance matrix and the correlation matrix
CrossSectionCorr.CovarianceMatrix=zeros(length(FnsToEvaluate),length(FnsToEvaluate));
CrossSectionCorr.CorrelationMatrix=zeros(length(FnsToEvaluate),length(FnsToEvaluate));

%% Calculate all the cross-sectional correlations, note that this creates the 'upper triangular' part
for ff1=1:length(FnsToEvaluate)
    % Includes check for cases in which no parameters are actually required
    if isempty(FnsToEvaluateParamNames(ff1).Names)
        ParamCell1=cell(0,1);
    else
        FnToEvaluateParamsAgeMatrix1=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff1).Names,N_j);
        nFnToEvaluateParams1=size(FnToEvaluateParamsAgeMatrix1,2);
        ParamCell1=cell(nFnToEvaluateParams1,1);
        if N_z==0
            for ii=1:nFnToEvaluateParams1
                ParamCell1(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix1(:,ii),-1)}; % (a,j,l_d+l_a), so we want j to be after N_a
            end
        else
            for ii=1:nFnToEvaluateParams1
                ParamCell1(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix1(:,ii),-2)}; % (a,z,j,l_d+l_a), so we want j to be after N_a and N_z
            end
        end
    end
    Values1=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff1},ParamCell1,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J);
    Values1=reshape(Values1,[N_total,1]);

    Mean1=sum(Values1.*StationaryDistVec);
    StdDev1=sqrt(sum(StationaryDistVec.*((Values1-Mean1.*ones(N_total,1)).^2)));

    CrossSectionCorr.(AggVarNames{ff1}).Mean=Mean1;
    CrossSectionCorr.(AggVarNames{ff1}).StdDeviation=StdDev1;

    for ff2=ff1:length(FnsToEvaluate)
        if ff1==ff2
            CrossSectionCorr.(AggVarNames{ff1}).(AggVarNames{ff2})=1;

            % and matrix version
            CrossSectionCorr.CovarianceMatrix(ff1,ff2)=StdDev1^2;
            CrossSectionCorr.CorrelationMatrix(ff1,ff2)=1;
        else
            if isempty(FnsToEvaluateParamNames(ff2).Names)
                ParamCell2=cell(0,1);
            else
                FnToEvaluateParamsAgeMatrix2=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff2).Names,N_j);
                nFnToEvaluateParams2=size(FnToEvaluateParamsAgeMatrix2,2);
                ParamCell2=cell(nFnToEvaluateParams2,1);
                if N_z==0
                    for ii=1:nFnToEvaluateParams2
                        ParamCell2(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix2(:,ii),-1)};
                    end
                else
                    for ii=1:nFnToEvaluateParams2
                        ParamCell2(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix2(:,ii),-2)};
                    end
                end
            end
            Values2=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff2},ParamCell2,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J);
            Values2=reshape(Values2,[N_total,1]);

            Mean2=sum(Values2.*StationaryDistVec);
            StdDev2=sqrt(sum(StationaryDistVec.*((Values2-Mean2.*ones(N_total,1)).^2)));

            CoVar=sum((Values1-Mean1*ones(N_total,1,'gpuArray')).*(Values2-Mean2*ones(N_total,1,'gpuArray')).*StationaryDistVec);
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
