function CrossSectionCorr=EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz(StationaryDist, Policy, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid,simoptions)
% Evaluates the cross-sectional correlation between every possible pair from FnsToEvaluate
% eg. if you give a FnsToEvaluate with three functions you will get nine cross-sectional correlations; with two function you get four.
%
% Since they are calculated anyway as intermediate steps,
% Also reports the Mean and Standard Deviation of every function
% And the Covariance of every pair of functions.
%
% simoptions.whichcombos ([numFnsToEvaluate, numFnsToEvaluate] of zeros/ones: the diagonal selects a function's Mean/StdDeviation, the
% off-diagonal a pair's covariance/correlation; only the upper triangle is read) selects what is computed; see below.
%
% simoptions.conditionalrestrictions: evaluate the same statistics conditional on each restriction being one (not zero). The
% restricted results are in CrossSectionCorr.(restrictionname), with the same fields as the unrestricted ones, plus
% CrossSectionCorr.(restrictionname).RestrictedSampleMass. With restrictions, simoptions.whichcombos can also be given as
% [numFnsToEvaluate, numFnsToEvaluate, 1+number of conditional restrictions] (page 1 unrestricted, pages 2:end the restrictions).


%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
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
    if ~isfield(simoptions,'warnzerorestrictedmass')
        simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
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

%% Conditional restrictions: the number of 'pages' (1 unrestricted, plus one per restriction)
useCondlRest=0;
nwhichpages=1;
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);
    nwhichpages=1+length(CondlRestnFnNames);
end

%% simoptions.whichcombos: which functions and pairs to compute
% [numFnsToEvaluate, numFnsToEvaluate] of zeros/ones. The diagonal (ff,ff) selects the Mean and StdDeviation of function ff (and its
% variance on the CovarianceMatrix diagonal); the off-diagonal (ff1,ff2) selects the covariance and correlation of the pair. Only the
% upper triangle (ff1<=ff2) is read, so a symmetric matrix or just its upper triangle can be given. A function with nothing selected
% (its diagonal and all its pairs zero) is not evaluated; a selected pair has both its functions evaluated (their means and std devs
% are needed for the pair, and are then also reported only if the diagonal asks). Skipped entries are NaN. Default all ones.
% With conditional restrictions it is [numFnsToEvaluate, numFnsToEvaluate, 1+number of conditional restrictions]: page 1 is the
% unrestricted stats, pages 2:end the restrictions in the fieldnames order of simoptions.conditionalrestrictions. A
% [numFnsToEvaluate, numFnsToEvaluate] input with restrictions is applied to every page.
numFnsToEvaluate=length(FnsToEvaluate);
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,numFnsToEvaluate,nwhichpages); % whichcombos here is [nFns, nFns, 1+nRestr]
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ismatrix(whichcombos) && nwhichpages>1 && isequal(size(whichcombos),[numFnsToEvaluate,numFnsToEvaluate])
        whichcombos=repmat(whichcombos,[1,1,nwhichpages]); % one matrix with restrictions: apply to every page
    end
    if ~isequal(size(whichcombos,1:3),[numFnsToEvaluate,numFnsToEvaluate,nwhichpages]) || ndims(whichcombos)>3
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(numFnsToEvaluate),',',num2str(nwhichpages),'] (number of FnsToEvaluate, twice, 1+number of conditional restrictions; the third dimension is dropped when there are no conditional restrictions)'])
    end
    whichcombos=double(whichcombos);
    for pp=1:nwhichpages
        whichcombos(:,:,pp)=max(triu(whichcombos(:,:,pp)),triu(whichcombos(:,:,pp))'); % symmetric, from the upper triangle
    end
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

%% If there are any conditional restrictions, set up for these
% Code works by evaluating the restriction and imposing this on the distribution (and renormalizing it).
if useCondlRest==1
    restrictedsamplemass=nan(length(CondlRestnFnNames),1);
    RestrictionMask=cell(length(CondlRestnFnNames),1); % each restriction kept as a logical mask over the grid (1 byte per point); the restricted weights are formed from it where used

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

        if N_z==0
            CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters,CondlRestnFnParamNames,N_j,2); % j in 2nd dimension: (a,j,l_d+l_a), so we want j to be after N_a
            RestrictionValues=logical(EvalFnOnAgentDist_Grid_J(CondlRestnFn,CellOverAgeOfParamValues,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]));
        else
            CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters,CondlRestnFnParamNames,N_j,3); % j in 3rd dimension: (a,z,j,l_d+l_a), so we want j to be after N_a and N_z
            RestrictionValues=logical(EvalFnOnAgentDist_Grid_J(CondlRestnFn,CellOverAgeOfParamValues,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J));
        end

        % Keep the restricted mass, and the mask
        RestrictionMask{rr}=reshape(RestrictionValues,[N_total,1]);
        restrictedsamplemass(rr)=sum(StationaryDistVec.*RestrictionMask{rr}); % mass that satisfies the restriction

        if restrictedsamplemass(rr)==0
            if simoptions.warnzerorestrictedmass==2
                warning('One of the conditional restrictions evaluates to a zero mass')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
            end
            whichcombos(:,:,1+rr)=0; % nothing can be computed on this page, its entries stay NaN
        end
    end
end
fnwantedpage=reshape(any(whichcombos,2),[numFnsToEvaluate,nwhichpages]); % the functions that get evaluated (their own stats or any pair), per page
fnwanted=any(fnwantedpage,2); % the functions that get evaluated on any page

% Each page is filled as its own output structure; page 1 (unrestricted) becomes CrossSectionCorr and pages 2:end CrossSectionCorr.(restrictionname)
PageOut=cell(nwhichpages,1);
for pp=1:nwhichpages
    PageOut{pp}=struct();
    % Report output by name, but also create the covariance matrix and the correlation matrix
    PageOut{pp}.CovarianceMatrix=nan(length(FnsToEvaluate),length(FnsToEvaluate));
    PageOut{pp}.CorrelationMatrix=nan(length(FnsToEvaluate),length(FnsToEvaluate));
end

%% Calculate all the cross-sectional correlations, note that this creates the 'upper triangular' part
for ff1=1:length(FnsToEvaluate)
    if ~fnwanted(ff1) % nothing involving this function is wanted
        continue
    end
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

    Mean1=nan(nwhichpages,1);
    StdDev1=nan(nwhichpages,1);
    for pp=1:nwhichpages
        if ~fnwantedpage(ff1,pp) % nothing involving this function is wanted on this page
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

    for ff2=ff1:length(FnsToEvaluate)
        if ff1==ff2 % the own stats of an evaluated function are byproducts of the pairs and are reported regardless of the diagonal
            for pp=1:nwhichpages
                if fnwantedpage(ff1,pp)
                    PageOut{pp}.(AggVarNames{ff1}).(AggVarNames{ff2})=1;

                    % and matrix version
                    PageOut{pp}.CovarianceMatrix(ff1,ff2)=StdDev1(pp)^2;
                    PageOut{pp}.CorrelationMatrix(ff1,ff2)=1;
                end
            end
        else
            if any(whichcombos(ff1,ff2,:)) % the pair is wanted on some page
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
            end
            for pp=1:nwhichpages
                if whichcombos(ff1,ff2,pp)==1 % the pair is wanted on this page
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
                elseif fnwantedpage(ff1,pp) && fnwantedpage(ff2,pp) % the pair is not wanted but both functions are evaluated: NaN (the matrices stay NaN); with a partner that is not evaluated there is no field, as in the mirror below
                    PageOut{pp}.(AggVarNames{ff1}).CovarianceWith.(AggVarNames{ff2})=NaN;
                    PageOut{pp}.(AggVarNames{ff1}).CorrelationWith.(AggVarNames{ff2})=NaN;
                end
            end
        end
    end
end


%% Just to make them easier to find, fill in the 'lower triangular' part
for pp=1:nwhichpages
    for ff1=1:length(FnsToEvaluate)
        for ff2=1:ff1-1
            if ~(fnwantedpage(ff1,pp) && fnwantedpage(ff2,pp)) % a pair with a function that was not evaluated has no fields to mirror
                continue
            end
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
