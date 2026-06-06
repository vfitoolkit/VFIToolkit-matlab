function AgeConditionalCrossSectionCorr=EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz(StationaryDist, Policy, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions)
% Evaluates the age-conditional cross-sectional correlation between every possible pair from FnsToEvaluate
% eg. if you give a FnsToEvaluate with three functions you will get nine cross-sectional correlations (per age group); with two function you get four.
%
% Since they are calculated anyway as intermediate steps,
% Also reports the Mean and Standard Deviation of every function (per age group)
% And the Covariance of every pair of functions (per age group)
%
% simoptions.agegroupings can be used to do conditional on 'age bins' rather than age
% e.g., simoptions.agegroupings=1:10:N_j will divide into 10 year age bins.


%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.agegroupings=1:1:N_j; % by default does each period separately
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    % Internal options
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'agegroupings')
        simoptions.agegroupings=1:1:N_j; % by default does each period separately
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
    % Internal options
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
end

if isfield(simoptions,'conditionalrestrictions')
    warning('Have not yet implemented simoptions.conditionalrestrictions for AgeConditionalStats_CrossSectionCovarCorr_FHorz so ignoring them, ask on forum if you need this')
end

if gpuDeviceCount==0
    error('AgeConditionalStats_CrossSectionCovarCorr_FHorz requires a GPU')
end

%%
l_a=length(n_a);
N_a=prod(n_a);

ngroups=length(simoptions.agegroupings);

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
numFnsToEvaluate=length(FnsToEvaluate);

%% Setup PolicyValues (trailing j axis so we can slice per age) and reshape StationaryDist
if N_z==0
    StationaryDist=reshape(StationaryDist,[N_a,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermuteJ=permute(PolicyValues,[2,1,3]); % (N_a,l_daprime,N_j)
else
    StationaryDist=reshape(StationaryDist,[N_a*N_z,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermuteJ=permute(PolicyValues,[2,3,1,4]); % (N_a,N_z,l_daprime,N_j)
end

%% Preallocate output
AgeConditionalCrossSectionCorr=struct();
for ff=1:numFnsToEvaluate
    AgeConditionalCrossSectionCorr.(AggVarNames{ff}).Mean=nan(1,ngroups,'gpuArray');
    AgeConditionalCrossSectionCorr.(AggVarNames{ff}).StdDeviation=nan(1,ngroups,'gpuArray');
end
for ff1=1:numFnsToEvaluate
    for ff2=1:numFnsToEvaluate
        if ff1==ff2
            AgeConditionalCrossSectionCorr.(AggVarNames{ff1}).(AggVarNames{ff2})=ones(1,ngroups,'gpuArray');
        else
            AgeConditionalCrossSectionCorr.(AggVarNames{ff1}).CovarianceWith.(AggVarNames{ff2})=nan(1,ngroups,'gpuArray');
            AgeConditionalCrossSectionCorr.(AggVarNames{ff1}).CorrelationWith.(AggVarNames{ff2})=nan(1,ngroups,'gpuArray');
        end
    end
end
AgeConditionalCrossSectionCorr.CovarianceMatrix=nan(numFnsToEvaluate,numFnsToEvaluate,ngroups,'gpuArray');
AgeConditionalCrossSectionCorr.CorrelationMatrix=nan(numFnsToEvaluate,numFnsToEvaluate,ngroups,'gpuArray');

%% Loop over age groupings and compute the cross-sectional covar/corr for each
for kk=1:ngroups
    j1=simoptions.agegroupings(kk);
    if kk<ngroups
        jend=simoptions.agegroupings(kk+1)-1;
    else
        jend=N_j;
    end
    jspan=jend-j1+1;

    % Build the normalized within-group distribution
    if N_z==0
        StationaryDistVec_kk=reshape(StationaryDist(:,j1:jend),[N_a*jspan,1]);
    else
        StationaryDistVec_kk=reshape(StationaryDist(:,j1:jend),[N_a*N_z*jspan,1]);
    end
    massvec_kk=sum(StationaryDistVec_kk);
    if massvec_kk>0
        StationaryDistVec_kk=StationaryDistVec_kk./massvec_kk;
    else
        % No mass in this age grouping; everything stays NaN from preallocation
        continue
    end
    N_total_kk=length(StationaryDistVec_kk);

    %% Step 1: evaluate every function on the within-group grid
    ValuesCell=cell(numFnsToEvaluate,1);
    if N_z==0
        for ff=1:numFnsToEvaluate
            Values=nan(N_a,jspan,'gpuArray');
            for jj=j1:jend
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                Values(:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            end
            ValuesCell{ff}=reshape(Values,[N_total_kk,1]);
        end
    else
        for ff=1:numFnsToEvaluate
            Values=nan(N_a,N_z,jspan,'gpuArray');
            for jj=j1:jend
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                Values(:,:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
            end
            ValuesCell{ff}=reshape(Values,[N_total_kk,1]);
        end
    end

    %% Step 2: per-function Mean and StdDev under the within-group distribution
    Means_kk=zeros(numFnsToEvaluate,1,'gpuArray');
    StdDevs_kk=zeros(numFnsToEvaluate,1,'gpuArray');
    for ff=1:numFnsToEvaluate
        Means_kk(ff)=sum(ValuesCell{ff}.*StationaryDistVec_kk);
        StdDevs_kk(ff)=sqrt(sum(StationaryDistVec_kk.*((ValuesCell{ff}-Means_kk(ff).*ones(N_total_kk,1,'gpuArray')).^2)));

        AgeConditionalCrossSectionCorr.(AggVarNames{ff}).Mean(kk)=Means_kk(ff);
        AgeConditionalCrossSectionCorr.(AggVarNames{ff}).StdDeviation(kk)=StdDevs_kk(ff);
    end

    %% Step 3: upper-triangular covariance/correlation
    for ff1=1:numFnsToEvaluate
        for ff2=ff1:numFnsToEvaluate
            if ff1==ff2
                AgeConditionalCrossSectionCorr.CovarianceMatrix(ff1,ff2,kk)=StdDevs_kk(ff1)^2;
                AgeConditionalCrossSectionCorr.CorrelationMatrix(ff1,ff2,kk)=1;
            else
                CoVar=sum((ValuesCell{ff1}-Means_kk(ff1)*ones(N_total_kk,1,'gpuArray')).*(ValuesCell{ff2}-Means_kk(ff2)*ones(N_total_kk,1,'gpuArray')).*StationaryDistVec_kk);
                Corr=CoVar/(StdDevs_kk(ff1)*StdDevs_kk(ff2));

                AgeConditionalCrossSectionCorr.(AggVarNames{ff1}).CovarianceWith.(AggVarNames{ff2})(kk)=CoVar;
                AgeConditionalCrossSectionCorr.(AggVarNames{ff1}).CorrelationWith.(AggVarNames{ff2})(kk)=Corr;

                AgeConditionalCrossSectionCorr.CovarianceMatrix(ff1,ff2,kk)=CoVar;
                AgeConditionalCrossSectionCorr.CorrelationMatrix(ff1,ff2,kk)=Corr;
            end
        end
    end

    %% Step 4: mirror to the lower-triangular part
    for ff1=1:numFnsToEvaluate
        for ff2=1:ff1-1
            AgeConditionalCrossSectionCorr.(AggVarNames{ff1}).CovarianceWith.(AggVarNames{ff2})(kk)=AgeConditionalCrossSectionCorr.(AggVarNames{ff2}).CovarianceWith.(AggVarNames{ff1})(kk);
            AgeConditionalCrossSectionCorr.(AggVarNames{ff1}).CorrelationWith.(AggVarNames{ff2})(kk)=AgeConditionalCrossSectionCorr.(AggVarNames{ff2}).CorrelationWith.(AggVarNames{ff1})(kk);

            AgeConditionalCrossSectionCorr.CovarianceMatrix(ff1,ff2,kk)=AgeConditionalCrossSectionCorr.CovarianceMatrix(ff2,ff1,kk);
            AgeConditionalCrossSectionCorr.CorrelationMatrix(ff1,ff2,kk)=AgeConditionalCrossSectionCorr.CorrelationMatrix(ff2,ff1,kk);
        end
    end
end

AgeConditionalCrossSectionCorr.Notes='The CovarianceMatrix and CorrelationMatrix are nFn x nFn x ngroups; the third index aligns with simoptions.agegroupings. They essentially duplicate the individual named correlations and covariances, but depending on what you want to do the matrix or the individual named pairs might be easier to use so both are created.';

end
