function CrossSectionCorr=EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz(StationaryDist, Policy, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid,simoptions)
% Evaluates the cross-sectional correlation between every possible pair from FnsToEvaluate
% eg. if you give a FnsToEvaluate with three functions you will get nine cross-sectional correlations; with two function you get four.
%
% Since they are calculated anyway as intermediate steps,
% Also reports the Mean and Standard Deviation of every function
% And the Covariance of every pair of functions.
%
% simoptions.conditionalrestrictions: evaluate the same statistics conditional on each restriction being one (not zero). The
% restricted results are in CrossSectionCorr.(restrictionname), with the same fields as the unrestricted ones, plus
% CrossSectionCorr.(restrictionname).RestrictedSampleMass. A restriction with zero mass gives a warning, and NaN.


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

%%
l_a=length(n_a);

N_a=prod(n_a);

a_gridvals=CreateGridvals(n_a,a_grid,1);
% Switch to z_gridvals (folding e and semiz into z if appropriate)
[n_z,z_gridvals,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_InfHorz(n_z,z_grid,simoptions,Parameters);

%%
StationaryDistVec=reshape(StationaryDist,[N_a*max(N_z,1),1]);


PolicyValues=PolicyInd2Val_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid,simoptions);
% Note: must collapse n_a (and n_z) into N_a (and N_z) before the permute, as a_gridvals is
% the joint grid over N_a. [Otherwise, with two endogenous states, the assets stay split and
% do not match a_gridvals; only coincides when there is a single endogenous state.]
if N_z==0
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a]),[2,1]); %[N_a,l_d+l_a]
else
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]
end
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


N_total=N_a*max(N_z,1);
numFnsToEvaluate=length(FnsToEvaluate);

%% If there are any conditional restrictions, set up for these
% Code works by evaluating the restriction and imposing this on the distribution (and renormalizing it).
useCondlRest=0;
nwhichpages=1; % 'pages': the unrestricted stats, then one per restriction
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);
    nwhichpages=1+length(CondlRestnFnNames);

    restrictedsamplemass=nan(length(CondlRestnFnNames),1);
    RestrictionMask=cell(length(CondlRestnFnNames),1); % each restriction kept as a logical mask over the grid; the restricted weights are formed from it where used

    for rr=1:length(CondlRestnFnNames)
        % The current conditional restriction function
        CondlRestnFn=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});
        % Get parameter names for Conditional Restriction functions
        temp=getAnonymousFnInputNames(CondlRestnFn);
        if length(temp)>(l_daprime+l_a+l_z)
            CondlRestnFnParamNames={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            CondlRestnFnParamNames={};
        end
        CondlRestnFnParamsCell=CreateCellFromParams(Parameters,CondlRestnFnParamNames);

        RestrictionValues=logical(EvalFnOnAgentDist_Grid(CondlRestnFn, CondlRestnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals));
        RestrictionMask{rr}=reshape(RestrictionValues,[N_total,1]);
        restrictedsamplemass(rr)=sum(StationaryDistVec.*RestrictionMask{rr}); % mass that satisfies the restriction

        if restrictedsamplemass(rr)==0
            warning('One of the conditional restrictions evaluates to a zero mass')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
        end
    end
end
pagewanted=true(nwhichpages,1); % a restriction of zero mass has nothing to compute (its entries are NaN)
if useCondlRest==1
    pagewanted(2:end)=(restrictedsamplemass>0);
end

% Each page is filled as its own output structure; page 1 (unrestricted) becomes CrossSectionCorr and pages 2:end CrossSectionCorr.(restrictionname)
PageOut=cell(nwhichpages,1);
for pp=1:nwhichpages
    PageOut{pp}=struct();
    % Report output by name, but also create the covariance matrix and the correlation matrix
    if pp==1
        PageOut{pp}.CovarianceMatrix=zeros(numFnsToEvaluate,numFnsToEvaluate);
        PageOut{pp}.CorrelationMatrix=zeros(numFnsToEvaluate,numFnsToEvaluate);
    else
        PageOut{pp}.CovarianceMatrix=nan(numFnsToEvaluate,numFnsToEvaluate);
        PageOut{pp}.CorrelationMatrix=nan(numFnsToEvaluate,numFnsToEvaluate);
    end
end

%% Calculate all the cross-sectional correlations, note that this creates the 'upper triangular' part
for ff1=1:numFnsToEvaluate
    FnToEvaluateParamsCell1=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff1).Names);
    Values1=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff1}, FnToEvaluateParamsCell1,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
    Values1=reshape(Values1,[N_total,1]);

    Mean1=nan(nwhichpages,1);
    StdDev1=nan(nwhichpages,1);
    for pp=1:nwhichpages
        if ~pagewanted(pp)
            continue
        end
        if pp==1
            PageDist=StationaryDistVec;
        else
            PageDist=StationaryDistVec.*RestrictionMask{pp-1}/restrictedsamplemass(pp-1); % the restricted distribution, normalised to mass one
        end
        Mean1(pp)=sum(Values1.*PageDist);
        StdDev1(pp)=sqrt(sum(PageDist.*((Values1-Mean1(pp).*ones(N_total,1)).^2)));

        PageOut{pp}.(AggVarNames{ff1}).Mean=Mean1(pp);
        PageOut{pp}.(AggVarNames{ff1}).StdDeviation=StdDev1(pp);
    end

    for ff2=ff1:numFnsToEvaluate
        if ff1==ff2
            for pp=1:nwhichpages
                if pagewanted(pp)
                    PageOut{pp}.(AggVarNames{ff1}).(AggVarNames{ff2})=1;

                    % and matrix version
                    PageOut{pp}.CovarianceMatrix(ff1,ff2)=StdDev1(pp)^2;
                    PageOut{pp}.CorrelationMatrix(ff1,ff2)=1;
                end
            end
        else
            FnToEvaluateParamsCell2=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff2).Names);
            Values2=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff2}, FnToEvaluateParamsCell2,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
            Values2=reshape(Values2,[N_total,1]);
            for pp=1:nwhichpages
                if ~pagewanted(pp)
                    continue
                end
                if pp==1
                    PageDist=StationaryDistVec;
                else
                    PageDist=StationaryDistVec.*RestrictionMask{pp-1}/restrictedsamplemass(pp-1); % the restricted distribution, normalised to mass one
                end
                Mean2=sum(Values2.*PageDist);
                StdDev2=sqrt(sum(PageDist.*((Values2-Mean2.*ones(N_total,1)).^2)));

                CoVar=sum((Values1-Mean1(pp)*ones(N_total,1,'gpuArray')).*(Values2-Mean2*ones(N_total,1,'gpuArray')).*PageDist);
                Corr=CoVar/(StdDev1(pp)*StdDev2);

                % Store them
                PageOut{pp}.(AggVarNames{ff1}).CovarianceWith.(AggVarNames{ff2})=CoVar;
                PageOut{pp}.(AggVarNames{ff1}).CorrelationWith.(AggVarNames{ff2})=Corr;

                % and matrix version
                PageOut{pp}.CovarianceMatrix(ff1,ff2)=CoVar;
                PageOut{pp}.CorrelationMatrix(ff1,ff2)=Corr;
            end
        end
    end
end


%% Just to make them easier to find, fill in the 'lower triangular' part
for pp=1:nwhichpages
    if ~pagewanted(pp)
        continue
    end
    for ff1=1:numFnsToEvaluate
        for ff2=1:ff1-1
            PageOut{pp}.(AggVarNames{ff1}).CovarianceWith.(AggVarNames{ff2})=PageOut{pp}.(AggVarNames{ff2}).CovarianceWith.(AggVarNames{ff1});
            PageOut{pp}.(AggVarNames{ff1}).CorrelationWith.(AggVarNames{ff2})=PageOut{pp}.(AggVarNames{ff2}).CorrelationWith.(AggVarNames{ff1});

            % and matrix version
            PageOut{pp}.CovarianceMatrix(ff1,ff2)=PageOut{pp}.CovarianceMatrix(ff2,ff1);
            PageOut{pp}.CorrelationMatrix(ff1,ff2)=PageOut{pp}.CorrelationMatrix(ff2,ff1);
        end
    end
end

%% Assemble the output
CrossSectionCorr=PageOut{1};
if useCondlRest==1
    for rr=1:length(CondlRestnFnNames)
        CrossSectionCorr.(CondlRestnFnNames{rr})=PageOut{1+rr};
        CrossSectionCorr.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr);
    end
end

CrossSectionCorr.Notes='The CovarianceMatrix and Correlation matrix are essentially duplicating the individual correlations and covariances, but depending on what you want to do the matrix or the individual named pairs might be easier to use so both are created.';

end
