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
%
% simoptions.ntimeperiods (default 1; a vector, e.g. [1,2]) and simoptions.ntimeperiodstype ('average', the default, or 'sum'): the
% same statistics of the M-period sums/averages W_j=x_j+...+x_{j+M-1} of the functions, pooled over the window start ages j whose
% window fits (j<=N_j-M+1), each start age weighted by its mass (with the age weights, as the single-period statistics pool the ages):
% the Mean and StdDeviation of each function's window and the CovarianceWith/CorrelationWith of every pair of functions over the same
% window (only windows of the same length are paired), under [fnname,'_',M,'periodAverage'] (or 'periodSum'), with the matrices in
% CovarianceMatrix_2periodAverage and CorrelationMatrix_2periodAverage. They are exact (the moments of the window sums are pushed
% through the transitions). Any M>1 needs the z transition probabilities as simoptions.pi_z (as LifeCycleProfiles_FHorz_Case1). A
% conditional restriction must hold at every age of the window; the restricted window outputs have
% (restrictionname).RestrictedSampleMass_2periodAverage, the mass at the start ages that satisfies it throughout. With ntimeperiods of
% length L, simoptions.whichcombos may be [L*nFns, L*nFns, 1+nRestr] (block diagonal, blocks in the order of ntimeperiods; the
% off-diagonal blocks, pairs of different window lengths, must be zero); an nFns x nFns form applies to every M.
% The M>1 windows handle an experience asset (simoptions.experienceasset, with simoptions.aprimeFn): a2'=aprimeFn(d2,a2) is put on the a2 grid by linear interpolation, as the agent distribution does (d_grid and a_grid are the full grids); and simoptions.experienceassetu (a2'=aprimeFn(d2,a2,u), with simoptions.n_u, u_grid, pi_u: u is iid between periods).


%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    simoptions.ntimeperiods=1; % the statistics of the sums/averages over M consecutive periods, for each M in ntimeperiods (1 is the single period)
    simoptions.ntimeperiodstype='average'; % 'average' or 'sum' over the M periods
    simoptions.experienceasset=0; % an experience asset (the last simoptions.experienceasset endogenous states, with simoptions.aprimeFn)
    simoptions.l_dexperienceasset=1; % the number of decision variables (the last ones, before any for semiz) that the experience asset depends on
    simoptions.experienceassetu=0; % an experience asset with an iid shock u (the last simoptions.experienceassetu endogenous states, with simoptions.aprimeFn, n_u, u_grid, pi_u)
    simoptions.l_dexperienceassetu=1; % the number of decision variables (the last ones, before any for semiz) that the experienceassetu depends on
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
    if ~isfield(simoptions,'ntimeperiods')
        simoptions.ntimeperiods=1; % the statistics of the sums/averages over M consecutive periods, for each M in ntimeperiods (1 is the single period)
    end
    if ~isfield(simoptions,'ntimeperiodstype')
        simoptions.ntimeperiodstype='average'; % 'average' or 'sum' over the M periods
    end
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0; % an experience asset (the last simoptions.experienceasset endogenous states, with simoptions.aprimeFn)
    end
    if ~isfield(simoptions,'l_dexperienceasset')
        simoptions.l_dexperienceasset=1; % the number of decision variables (the last ones, before any for semiz) that the experience asset depends on
    end
    if ~isfield(simoptions,'experienceassetu')
        simoptions.experienceassetu=0; % an experience asset with an iid shock u (the last simoptions.experienceassetu endogenous states, with simoptions.aprimeFn, n_u, u_grid, pi_u)
    end
    if ~isfield(simoptions,'l_dexperienceassetu')
        simoptions.l_dexperienceassetu=1; % the number of decision variables (the last ones, before any for semiz) that the experienceassetu depends on
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

%% Multi-period sums/averages: simoptions.ntimeperiods (the window lengths M, 1 is the single period) and simoptions.ntimeperiodstype
ntimeperiods=gather(simoptions.ntimeperiods(:)'); % the order is kept: it is the order of the blocks of whichcombos
if isempty(ntimeperiods) || ~isnumeric(ntimeperiods) || any(ntimeperiods<1) || any(ntimeperiods~=round(ntimeperiods))
    error('CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods must be a vector of positive integers (the number of periods summed/averaged over, 1 is the single period)')
end
if length(unique(ntimeperiods))<length(ntimeperiods)
    error('CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods must not contain repeated values')
end
if any(ntimeperiods>N_j)
    error('CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods cannot exceed N_j (no window of that many periods fits in the life cycle)')
end
if ~(ischar(simoptions.ntimeperiodstype) || isstring(simoptions.ntimeperiodstype)) || ~any(strcmp(simoptions.ntimeperiodstype,{'average','sum'}))
    error('CrossSectionCovarCorr_FHorz: simoptions.ntimeperiodstype must be ''average'' or ''sum''')
end
nblocks=length(ntimeperiods);
has1=any(ntimeperiods==1);
windowM=ntimeperiods(ntimeperiods>1);
nW=length(windowM);
if strcmp(simoptions.ntimeperiodstype,'average')
    windowstr='periodAverage';
else
    windowstr='periodSum';
end
if simoptions.experienceasset>=1 && simoptions.experienceassetu>=1
    error('CrossSectionCovarCorr_FHorz: simoptions.experienceasset and simoptions.experienceassetu cannot both be used')
end
n_zW=n_z; % the windows need the transition probabilities of (semiz,z,e) separately: keep the shapes and simoptions as given
simoptionsW=simoptions;

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
% With simoptions.ntimeperiods of length L>1 (or without 1 in it) whichcombos is block diagonal, [L*nFns, L*nFns, 1+nRestr], one
% nFns x nFns block per window length in the order of ntimeperiods (an nFns x nFns form, with or without pages, applies to every
% block); the off-diagonal blocks pair windows of different lengths and must be zero. Internally whichcombos is the M=1 block (zeros
% when 1 is not in ntimeperiods) and whichcombosW(:,:,:,lw) the block of windowM(lw).
if ~(nblocks==1 && has1)
    nRwc=nblocks*numFnsToEvaluate;
    if ~isfield(simoptions,'whichcombos')
        wcIn=ones(numFnsToEvaluate,numFnsToEvaluate,nwhichpages);
    else
        wcIn=simoptions.whichcombos;
        if ~(isnumeric(wcIn) || islogical(wcIn)) || any(wcIn(:)~=0 & wcIn(:)~=1)
            error('simoptions.whichcombos must contain only zeros and ones')
        end
        wcIn=double(wcIn);
    end
    if size(wcIn,1)==numFnsToEvaluate && size(wcIn,2)==numFnsToEvaluate && nRwc~=numFnsToEvaluate % one block for every window length
        temp=zeros(nRwc,nRwc,size(wcIn,3));
        for bb=1:nblocks
            temp((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,(bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:)=wcIn;
        end
        wcIn=temp;
    end
    if ismatrix(wcIn) && nwhichpages>1 && isequal(size(wcIn),[nRwc,nRwc])
        wcIn=repmat(wcIn,[1,1,nwhichpages]); % one matrix with restrictions: apply to every page
    end
    if ~isequal(size(wcIn,1:3),[nRwc,nRwc,nwhichpages]) || ndims(wcIn)>3
        error(['simoptions.whichcombos must be of size [',num2str(nRwc),',',num2str(nRwc),',',num2str(nwhichpages),'] (length of simoptions.ntimeperiods times the number of FnsToEvaluate, twice, 1+number of conditional restrictions; block diagonal), or have ',num2str(numFnsToEvaluate),' rows and columns to apply to every window length'])
    end
    for pp=1:nwhichpages
        wcIn(:,:,pp)=max(triu(wcIn(:,:,pp)),triu(wcIn(:,:,pp))'); % symmetric, from the upper triangle
    end
    blockmask=kron(eye(nblocks),ones(numFnsToEvaluate));
    if any(wcIn.*(1-blockmask)~=0,'all')
        error('simoptions.whichcombos selects a pair of windows of different lengths (an off-diagonal block); only windows of the same length are paired')
    end
    if has1
        bb=find(ntimeperiods==1);
        whichcombos=wcIn((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,(bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:);
    else
        whichcombos=zeros(numFnsToEvaluate,numFnsToEvaluate,nwhichpages);
    end
    whichcombosW=zeros(numFnsToEvaluate,numFnsToEvaluate,nwhichpages,nW);
    for lw=1:nW
        bb=find(ntimeperiods==windowM(lw));
        whichcombosW(:,:,:,lw)=wcIn((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,(bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:);
    end
elseif ~isfield(simoptions,'whichcombos')
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


%% Multi-period sums/averages (simoptions.ntimeperiods with M>1): the cross-section statistics of the windows, by start age
% For a start age j and a page (the unrestricted population, or those satisfying a restriction at every age of the window), the
% moments of the window sums S_f=x~_f,j+...+x~_f,j+M-1 (x~ the value minus a centering constant) are carried as measures over the
% states: m0, m0*S_f for each function, and m0*S_f*S_g for each pair (f with itself too). At the start age they are the page's mass at
% j times 1, x~_f and x~_f*x~_g; each later age they are pushed through the age's transition (the policy step, then z, then e, as
% the agent distribution iteration does; these measures are signed but the pushes are linear), masked by the restriction at the new
% age, and the new age's values are added (m0*S_f*S_g += x~_f*(m0*S_g) + x~_g*(m0*S_f) + x~_f*x~_g*m0, then m0*S_f += x~_f*m0). At
% the end of a window the means, variances and covariances are those of the measures, exactly. The centering constant of a function
% at an age is its mean on the page at that age (the variances and covariances do not depend on it; the means add it back). All the
% windows in flight are pushed together, one age's policy step at a time.
if nW>0
    % The pooled statistics are pooled over the start ages: the per-start-age machinery of the age-conditional command (with the
    % agent distribution, the restrictions and the policy values by age) computes the window moments at each start age that fits
    StationaryDistA=reshape(StationaryDistVec,[],N_j);
    RestrictionMaskA=cell(nwhichpages-1,1);
    for rr=1:nwhichpages-1
        RestrictionMaskA{rr}=reshape(RestrictionMask{rr},[],N_j);
    end
    if N_z==0
        PolicyValuesPermuteJ=permute(PolicyValues,[2,1,3]); % (N_a,l_daprime,N_j)
    else
        PolicyValuesPermuteJ=permute(PolicyValues,[2,3,1,4]); % (N_a,N_z,l_daprime,N_j)
    end
    whichcombosWa=zeros(numFnsToEvaluate,numFnsToEvaluate,N_j,nwhichpages,nW); % [nFns,nFns,start age,pages,nW]: every start age whose window fits
    for lw=1:nW
        whichcombosWa(:,:,1:N_j-windowM(lw)+1,:,lw)=repmat(reshape(whichcombosW(:,:,:,lw),[numFnsToEvaluate,numFnsToEvaluate,1,nwhichpages]),[1,1,N_j-windowM(lw)+1,1]);
    end
    WMassAll=nan(N_j,nwhichpages,nW);
    if (isfield(simoptions,'experienceassetz') && simoptions.experienceassetz>=1) || (isfield(simoptions,'experienceassete') && simoptions.experienceassete>=1) || (isfield(simoptions,'experienceassetze') && simoptions.experienceassetze>=1) || (isfield(simoptions,'experienceassetsemiz') && simoptions.experienceassetsemiz>=1) || (isfield(simoptions,'riskyasset') && simoptions.riskyasset>=1) || (isfield(simoptions,'inheritanceasset') && simoptions.inheritanceasset==1) || (isfield(simoptions,'residualasset') && simoptions.residualasset>=1)
        error('CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods with M>1 is not yet implemented with experienceassetz, experienceassete, experienceassetze, experienceassetsemiz, riskyasset, inheritanceasset or residualasset, ask on forum if you need this')
    end
    % The shocks, separately: (semiz,z,e) [whichever are present], the states are ordered (a,semiz,z,e)
    N_zW=prod(n_zW);
    N_e=prod(simoptionsW.n_e);
    N_semiz=prod(simoptionsW.n_semiz);
    N_semizr=max(N_semiz,1);
    N_zr=max(N_zW,1);
    N_er=max(N_e,1);
    N_states=N_a*N_semizr*N_zr*N_er;
    if N_zW>0 || N_e>0
        if N_zW>0 && ~isfield(simoptionsW,'pi_z') && ~isfield(simoptionsW,'ExogShockFn')
            error('CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods with M>1 needs the transition probabilities of z, give them as simoptions.pi_z (the same as the pi_z input of StationaryDist_FHorz_Case1)')
        end
        if isfield(simoptionsW,'pi_z')
            pi_zW=simoptionsW.pi_z;
        else
            pi_zW=[];
        end
        [~,pi_z_J,simoptionsW]=ExogShockSetup_FHorz(n_zW,z_grid,pi_zW,N_j,Parameters,simoptionsW,2,0);
        if N_zW>0
            pi_z_J_gpu=gpuArray(pi_z_J);
        end
        if N_e>0
            pi_e_J_gpu=gpuArray(simoptionsW.pi_e_J);
        end
    end
    if N_semiz>0
        if ~isfield(simoptionsW,'l_dsemiz')
            simoptionsW.l_dsemiz=1; % by default, just one decision variable is used for the semi-exo state
        end
        simoptionsW=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,simoptionsW,2);
    end
    % The policy step: the indexes and weights of a' (and of the semi-exogenous transition) that each age's sparse matrix is built from
    if isempty(n_d) || prod(n_d)==0
        l_d=0;
    else
        l_d=length(n_d);
    end
    expassetbranch=(simoptionsW.experienceasset>=1 || simoptionsW.experienceassetu>=1); % the experience-asset branch of the policy step below
    PolicyW=gpuArray(reshape(Policy,[size(Policy,1),N_a,N_semizr*N_zr*N_er,N_j]));
    if expassetbranch==1
        % Experience asset (as StationaryDist_FHorz_ExpAsset, _ExpAsset_noz and _ExpAssetSemiExo): the last simoptions.experienceasset
        % endogenous states are a2, with a2'=aprimeFn(d2,a2), which is in general between two grid points of each a2 dimension. So a
        % state goes to the N_probs=2^l_a2 corners of the a2 grid around a2' (with the a1' of the Policy), with the linear-interpolation
        % weights; with gridinterplayer=1 each corner is also split between the lower and upper a1' grid point (N_probs doubles).
        % Policy_aprime (the index into a') and PolicyProbs (the weights) are [N_states,N_probs,N_j].
        if ~isfield(simoptionsW,'aprimeFn')
            error('CrossSectionCovarCorr_FHorz: to use an experience asset you must define simoptions.aprimeFn')
        end
        if simoptionsW.experienceasset+simoptionsW.experienceassetu>2 || l_a<simoptionsW.experienceasset+simoptionsW.experienceassetu
            error('CrossSectionCovarCorr_FHorz: simoptions.experienceasset (or experienceassetu) must be 1 or 2 (the number of experience assets), and at most length(n_a)')
        end
        if simoptionsW.experienceassetu>=1 && ~(isfield(simoptionsW,'n_u') && isfield(simoptionsW,'u_grid') && isfield(simoptionsW,'pi_u'))
            error('CrossSectionCovarCorr_FHorz: to use experienceassetu you must define simoptions.n_u, simoptions.u_grid and simoptions.pi_u')
        end
        if simoptionsW.experienceasset>=1
            l_a2=simoptionsW.experienceasset;
            l_dexp=simoptionsW.l_dexperienceasset;
            l_u=0;
            N_uu=1; % a2'=aprimeFn(d2,a2): one a2' per state
        else
            % experienceassetu (as StationaryDist_FHorz_ExpAssetu, _ExpAssetu_noz and _ExpAssetuSemiExo): a2'=aprimeFn(d2,a2,u), with u
            % iid between periods (the Policy does not depend on u), so a state has an a2' for each u, with probability pi_u
            l_a2=simoptionsW.experienceassetu;
            l_dexp=simoptionsW.l_dexperienceassetu;
            n_u=simoptionsW.n_u;
            l_u=length(n_u);
            N_uu=prod(n_u);
            u_grid=gpuArray(simoptionsW.u_grid(:));
            pi_u=gpuArray(simoptionsW.pi_u(:));
        end
        if l_a==l_a2
            n_a1=0; % no standard endogenous state
            l_a1=0;
            N_a1=1; % so that the a' index below is just the a2' corner
        else
            n_a1=n_a(1:end-l_a2);
            l_a1=length(n_a1);
            N_a1=prod(n_a1);
        end
        n_a2=n_a(end-l_a2+1:end);
        a2_grid=gpuArray(a_grid(sum(n_a1)+1:end)); % on the gpu (as d_grid below), so that the arrayfun of aprimeFn runs on the gpu and expands [N_a,1] a2 values against [N_a,N_semizze] d values
        if N_semiz==0
            whichisdforexpasset=l_d-l_dexp+1:l_d; % the last l_dexp decision variables
        else
            whichisdforexpasset=l_d-l_dexp-simoptionsW.l_dsemiz+1:l_d-simoptionsW.l_dsemiz; % the l_dexp decision variables before the semiz ones
        end
        temp=getAnonymousFnInputNames(simoptionsW.aprimeFn);
        if length(temp)>(l_dexp+l_a2+l_u+(l_a2>=2)) % the (l_a2>=2) term is the 'whicha' selector slot, which aprimeFn only takes when there are two experience assets
            aprimeFnParamNames={temp{l_dexp+l_a2+l_u+(l_a2>=2)+1:end}}; % the first inputs are (d2,a2), then u for experienceassetu, plus the 'whicha' selector when l_a2>=2
        else
            aprimeFnParamNames={};
        end
        N_probs=N_uu*2^l_a2; % the a2 corners, each for every u (corner cc, u uu is column (cc-1)*N_uu+uu)
        Policy_aprime=zeros(N_states,N_probs,N_j,'gpuArray');
        PolicyProbs=zeros(N_states,N_probs,N_j,'gpuArray');
        for jj=1:N_j
            aprimeFnParamsVec=CreateVectorFromParams(Parameters,aprimeFnParamNames,jj);
            if simoptionsW.experienceasset>=1
                [aprimeIndexes,aprimeProbs]=CreateaprimePolicyExperienceAsset(gpuArray(PolicyW(:,:,:,jj)),simoptionsW.aprimeFn,whichisdforexpasset,n_d,n_a1,n_a2,N_semizr*N_zr*N_er,gpuArray(d_grid),a2_grid,aprimeFnParamsVec);
                piu=1;
            else
                [aprimeIndexes,aprimeProbs]=CreateaprimePolicyExperienceAssetu(gpuArray(PolicyW(:,:,:,jj)),simoptionsW.aprimeFn,whichisdforexpasset,n_d,n_a1,n_a2,N_semizr*N_zr*N_er,n_u,gpuArray(d_grid),a2_grid,u_grid,aprimeFnParamsVec);
                piu=pi_u'; % [1,N_u]
            end
            % l_a2==1: [N_a,N_semizr*N_zr*N_er(,N_u)], the lower a2' grid point and its probability; l_a2==2: [N_a,2,N_semizr*N_zr*N_er(,N_u)], the same for each a2 dimension
            % The a1' index of each state, zero-based (the a2' corner adds the rest)
            a1primeoffset=zeros(N_states,1,'gpuArray');
            for ii=1:l_a1
                a1primeoffset=a1primeoffset+prod(n_a1(1:ii-1))*(reshape(PolicyW(l_d+ii,:,:,jj),[N_states,1])-1);
            end
            if l_a2==1
                loIdx=reshape(aprimeIndexes,[N_states,N_uu]);
                probLo=reshape(aprimeProbs,[N_states,N_uu]);
                Policy_aprime(:,1:N_uu,jj)=a1primeoffset+1+N_a1*(loIdx-1);
                Policy_aprime(:,N_uu+1:2*N_uu,jj)=a1primeoffset+1+N_a1*loIdx;
                PolicyProbs(:,1:N_uu,jj)=probLo.*piu;
                PolicyProbs(:,N_uu+1:2*N_uu,jj)=(1-probLo).*piu;
            else % l_a2==2: the four corners of the a2 grid around (a2_1',a2_2')
                loIdx_1=reshape(aprimeIndexes(:,1,:,:),[N_states,N_uu]);
                loIdx_2=reshape(aprimeIndexes(:,2,:,:),[N_states,N_uu]);
                prob_1=reshape(aprimeProbs(:,1,:,:),[N_states,N_uu]);
                prob_2=reshape(aprimeProbs(:,2,:,:),[N_states,N_uu]);
                bits=[0 0; 1 0; 0 1; 1 1];
                for cc=1:4
                    a2Kron=(loIdx_1+bits(cc,1))+n_a2(1)*(loIdx_2+bits(cc,2)-1);
                    Policy_aprime(:,(cc-1)*N_uu+1:cc*N_uu,jj)=a1primeoffset+1+N_a1*(a2Kron-1);
                    if bits(cc,1)==0
                        p1=prob_1;
                    else
                        p1=1-prob_1;
                    end
                    if bits(cc,2)==0
                        p2=prob_2;
                    else
                        p2=1-prob_2;
                    end
                    PolicyProbs(:,(cc-1)*N_uu+1:cc*N_uu,jj)=p1.*p2.*piu;
                end
            end
        end
        if simoptions.gridinterplayer==1
            % each corner at the lower a1' grid point and at the one above it (a1 is the fastest index of a'), with the second-layer weights
            L2index=PolicyW(end-1,:,:,:); % L2 index (end-1 because end is L2flag)
            L2flag=PolicyW(end,:,:,:);
            L2index(L2flag==1)=1;                        % force all weight to lower grid point
            L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
            probupper=reshape((L2index-1)/(simoptions.ngridinterp+1),[N_states,1,N_j]); % probability of the upper grid point
            Policy_aprime=cat(2,Policy_aprime,Policy_aprime+1);
            PolicyProbs=cat(2,PolicyProbs.*(1-probupper),PolicyProbs.*probupper);
            N_probs=2*N_probs;
        end
    elseif l_a==1
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:),1);
    elseif l_a==2
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:)+n_a(1)*(PolicyW(l_d+2,:,:,:)-1),1);
    elseif l_a==3
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:)+n_a(1)*(PolicyW(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyW(l_d+3,:,:,:)-1),1);
    elseif l_a==4
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:)+n_a(1)*(PolicyW(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyW(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyW(l_d+4,:,:,:)-1),1);
    else
        error('CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods with M>1 cannot handle length(n_a)>4, contact me if you need this')
    end
    if simoptions.gridinterplayer==1 && expassetbranch==0
        L2index=PolicyW(end-1,:,:,:); % L2 index (end-1 because end is L2flag)
        L2flag=PolicyW(end,:,:,:);
        L2index(L2flag==1)=1;                        % force all weight to lower grid point
        L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
        probupper=shiftdim((L2index-1)/(simoptions.ngridinterp+1),1); % [N_a,N_semizr*N_zr*N_er,N_j]: probability of the upper grid point
    end
    if N_semiz==0
        if expassetbranch==1
            % Experience asset: add the z index of each state to each of its N_probs a' points to get the index into (a',z)
            zindex=repmat(repelem(N_a*gpuArray(0:1:N_zr-1)',N_a,1),N_er,1); % [N_states,1]
            Policy_aprimez=Policy_aprime+zindex; % [N_states,N_probs,N_j], index into (a',z)
            IIind=gpuArray(repmat((1:1:N_states)',N_probs,1)); % the state of each entry of reshape(Policy_aprimez(:,:,jj),[],1)
        else
            zindex=N_a*repmat(gpuArray(0:1:N_zr-1),1,N_er);
            Policy_aprimez=Policy_aprime+zindex; % [N_a,N_zr*N_er,N_j], index into (a',z)
            if simoptions.gridinterplayer==0
                IIind=gpuArray((1:1:N_states)');
            elseif simoptions.gridinterplayer==1
                IIind=gpuArray(repmat((1:1:N_states)',2,1));
            end
        end
    else
        l_d1=l_d-simoptionsW.l_dsemiz; % the last l_dsemiz decision variables are the ones that influence semiz
        N_dsemiz=prod(n_d(l_d1+1:l_d));
        if simoptionsW.l_dsemiz==1
            Policy_dsemiexo=PolicyW(l_d1+1,:,:,:);
        elseif simoptionsW.l_dsemiz==2
            Policy_dsemiexo=PolicyW(l_d1+1,:,:,:)+n_d(l_d1+1)*(PolicyW(l_d1+2,:,:,:)-1);
        elseif simoptionsW.l_dsemiz==3
            Policy_dsemiexo=PolicyW(l_d1+1,:,:,:)+n_d(l_d1+1)*(PolicyW(l_d1+2,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*(PolicyW(l_d1+3,:,:,:)-1);
        elseif simoptionsW.l_dsemiz==4
            Policy_dsemiexo=PolicyW(l_d1+1,:,:,:)+n_d(l_d1+1)*(PolicyW(l_d1+2,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*(PolicyW(l_d1+3,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*n_d(l_d1+3)*(PolicyW(l_d1+4,:,:,:)-1);
        end
        Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_states,1,N_j]));
        N_semizshort=max(max(max(sum((simoptionsW.pi_semiz_J>0),2))));
        [pi_semiz_J_short,idx]=sort(simoptionsW.pi_semiz_J,2); % puts the zeros on the left
        pi_semiz_J_short=gather(pi_semiz_J_short(:,end-N_semizshort+1:end,:,:)); % [N_semiz,N_semizshort,N_dsemiz,N_j-1]
        idxshort=gather(idx(:,end-N_semizshort+1:end,:,:)); % the semiz' each kept entry belongs to
        semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_zr*N_er,1)+N_semiz*(0:1:N_semizshort-1); % [N_states,N_semizshort]
        zprimeoffset=repmat(repelem(N_a*N_semiz*(0:1:N_zr-1)',N_a*N_semiz,1),N_er,1); % [N_states,1]: the z of each state, as an offset into (a',semiz',z)
        if expassetbranch==0
            Policy_aprime_cpu=gather(reshape(Policy_aprime,[N_states,1,N_j]));
        end
        if expassetbranch==1
            % Experience asset: each of the N_probs a' points of a state, each with every semiz' (as StationaryDist_FHorz_Iteration_SemiExo_nProbs_raw)
            Policy_aprime_cpu=gather(Policy_aprime); % [N_states,N_probs,N_j]
            PolicyProbs_cpu=gather(PolicyProbs); % [N_states,N_probs,N_j]
            II2=repelem((1:1:N_states)',1,N_probs*N_semizshort);
        elseif simoptions.gridinterplayer==0
            II2=repelem((1:1:N_states)',1,N_semizshort);
        elseif simoptions.gridinterplayer==1
            Policy_aprime2_cpu=cat(2,Policy_aprime_cpu,Policy_aprime_cpu+1); % [N_states,2,N_j]: the lower grid point and the one above it
            probupper_cpu=gather(reshape(probupper,[N_states,1,N_j]));
            PolicyProbs_cpu=cat(2,1-probupper_cpu,probupper_cpu); % [N_states,2,N_j]
            II2=repelem((1:1:N_states)',1,2*N_semizshort);
        end
    end

    % What is wanted: at (start age, page) the functions with some selection (their window Mean/StdDeviation) for each window length
    wantFn=reshape(any(whichcombosWa,2),[numFnsToEvaluate,N_j,nwhichpages,nW]); % [nFns,N_j,pages,nW]
    Mstop=zeros(N_j,nwhichpages); % the longest window wanted at each (start age, page)
    for lw=1:nW
        Mstop=max(Mstop,windowM(lw)*reshape(any(wantFn(:,:,:,lw),1),[N_j,nwhichpages]));
    end
    fnW=any(reshape(wantFn,[numFnsToEvaluate,N_j*nwhichpages*nW]),2); % the functions evaluated for the windows
    % The values of these functions at every age, and their mean on each page at each age (the centering constants)
    ValuesW=cell(numFnsToEvaluate,1);
    cW=zeros(numFnsToEvaluate,N_j,nwhichpages);
    for ff=1:numFnsToEvaluate
        if ~fnW(ff)
            continue
        end
        ValuesW{ff}=zeros(N_states,N_j,'gpuArray');
        for jj=1:N_j
            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
            if N_z==0
                ValuesW{ff}(:,jj)=reshape(EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,0,a_gridvals,[]),[N_states,1]);
            else
                ValuesW{ff}(:,jj)=reshape(EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj)),[N_states,1]);
            end
            for pp=1:nwhichpages
                if pp==1
                    mp=StationaryDistA(:,jj);
                else
                    mp=StationaryDistA(:,jj).*RestrictionMaskA{pp-1}(:,jj);
                end
                if sum(mp)>0
                    cW(ff,jj,pp)=gather(sum(mp.*ValuesW{ff}(:,jj))/sum(mp));
                end
            end
        end
    end
    % The outputs of the windows: [nFns,N_j,pages,nW] for the means and std devs, [nFns,nFns,N_j,pages,nW] for the covariances
    WMean=nan(numFnsToEvaluate,N_j,nwhichpages,nW);
    WStd=nan(numFnsToEvaluate,N_j,nwhichpages,nW);
    WCov=nan(numFnsToEvaluate,numFnsToEvaluate,N_j,nwhichpages,nW);
    WCorr=nan(numFnsToEvaluate,numFnsToEvaluate,N_j,nwhichpages,nW);
    WRSMass=nan(N_j,nwhichpages,nW); % (restricted pages) the start mass satisfying the restriction at every age of the window
    % The groups in flight: start age, page, functions, and the columns of inflight (m0, then m0*S_f, then m0*S_f*S_g for f<=g)
    inflight=zeros(N_states,0,'gpuArray');
    G_j0=zeros(1,0); G_pp=zeros(1,0); G_col=zeros(1,0); G_ncol=zeros(1,0); G_fns=cell(1,0);
    for jj=1:N_j
        % (i) Start the groups of start age jj
        for pp=1:nwhichpages
            if Mstop(jj,pp)==0
                continue
            end
            if pp==1
                m0=StationaryDistA(:,jj);
            else
                m0=StationaryDistA(:,jj).*RestrictionMaskA{pp-1}(:,jj); % the restriction at the start age (and below, at every later age of the window)
            end
            if sum(m0)==0
                if pp>1 && sum(StationaryDistA(:,jj))>0
                    for lw=1:nW
                        if any(wantFn(:,jj,pp,lw))
                            WRSMass(jj,pp,lw)=0; % nobody at the start age satisfies the restriction (the age itself has mass)
                        end
                    end
                end
                continue
            end
            fns=find(any(reshape(wantFn(:,jj,pp,:),[numFnsToEvaluate,nW]),2))';
            nfg=length(fns);
            cols=zeros(N_states,1+nfg+nfg*(nfg+1)/2,'gpuArray');
            cols(:,1)=m0;
            xt=zeros(N_states,nfg,'gpuArray');
            for kf=1:nfg
                xt(:,kf)=ValuesW{fns(kf)}(:,jj)-cW(fns(kf),jj,pp);
                cols(:,1+kf)=m0.*xt(:,kf);
            end
            cc=1+nfg;
            for kf=1:nfg
                for kg=kf:nfg
                    cc=cc+1;
                    cols(:,cc)=m0.*xt(:,kf).*xt(:,kg);
                end
            end
            G_j0=[G_j0,jj]; G_pp=[G_pp,pp]; G_col=[G_col,size(inflight,2)+1]; G_ncol=[G_ncol,size(cols,2)]; G_fns=[G_fns,{fns}]; %#ok<AGROW>
            inflight=[inflight,cols]; %#ok<AGROW>
        end

        % (ii) Read the windows that end at age jj
        for gg=1:length(G_j0)
            j0=G_j0(gg); pp=G_pp(gg);
            lw=find(windowM==jj-j0+1);
            if isempty(lw)
                continue
            end
            M=windowM(lw);
            fns=G_fns{gg}; nfg=length(fns);
            c0=G_col(gg);
            mass=gather(sum(inflight(:,c0)));
            if pp>1
                WRSMass(j0,pp,lw)=mass;
            end
            WMassAll(j0,pp,lw)=mass; % (every page) the start mass of the window population, the weight of start age j0 in the pooling
            if mass<=0
                continue
            end
            E1=gather(sum(inflight(:,c0+1:c0+nfg),1))'/mass; % nfg x 1
            E2=zeros(nfg,nfg);
            cc=c0+nfg;
            for kf=1:nfg
                for kg=kf:nfg
                    cc=cc+1;
                    E2(kf,kg)=gather(sum(inflight(:,cc)))/mass;
                    E2(kg,kf)=E2(kf,kg);
                end
            end
            CV=E2-E1*E1';
            for kf=1:nfg
                if CV(kf,kf)<=64*eps*abs(E2(kf,kf))
                    CV(kf,kf)=0; % below the rounding level of the difference it was computed as (the window sum is constant in the population)
                end
            end
            if strcmp(simoptions.ntimeperiodstype,'average')
                sc=1/M;
            else
                sc=1;
            end
            for kf=1:nfg
                ff=fns(kf);
                if ~wantFn(ff,j0,pp,lw)
                    continue
                end
                WMean(ff,j0,pp,lw)=sc*(sum(cW(ff,j0:jj,pp))+E1(kf));
                WStd(ff,j0,pp,lw)=sc*sqrt(max(CV(kf,kf),0));
                WCov(ff,ff,j0,pp,lw)=sc^2*CV(kf,kf);
                WCorr(ff,ff,j0,pp,lw)=1;
                for kg=1:nfg
                    gf=fns(kg);
                    if kg~=kf && whichcombosWa(ff,gf,j0,pp,lw)==1
                        WCov(ff,gf,j0,pp,lw)=sc^2*CV(kf,kg);
                        denom=sqrt(max(CV(kf,kf),0))*sqrt(max(CV(kg,kg),0));
                        if denom>1e-15
                            WCorr(ff,gf,j0,pp,lw)=CV(kf,kg)/denom;
                        end
                    end
                end
            end
        end

        % (iii) Drop the groups whose longest window has been read
        keepg=true(1,length(G_j0));
        keepcols=true(1,size(inflight,2));
        for gg=1:length(G_j0)
            if jj-G_j0(gg)+1>=Mstop(G_j0(gg),G_pp(gg))
                keepg(gg)=false;
                keepcols(G_col(gg):G_col(gg)+G_ncol(gg)-1)=false;
            end
        end
        newcol=cumsum(keepcols);
        G_col=newcol(G_col(keepg));
        G_j0=G_j0(keepg); G_pp=G_pp(keepg); G_ncol=G_ncol(keepg); G_fns=G_fns(keepg);
        inflight=inflight(:,keepcols);
        if isempty(G_j0)
            continue
        end

        % (iv) Push everything in flight from age jj to jj+1, then mask by the restriction at jj+1 and add the age-jj+1 values
        if N_semiz==0
            if expassetbranch==1
                Gammatranspose_jj=sparse(reshape(Policy_aprimez(:,:,jj),[],1),IIind,reshape(PolicyProbs(:,:,jj),[],1),N_a*N_zr,N_states); % sparse gpuArray; sparse() accumulates repeated indexes (corners that coincide)
            elseif simoptions.gridinterplayer==0
                Gammatranspose_jj=sparse(reshape(Policy_aprimez(:,:,jj),[],1),IIind,ones(N_states,1,'gpuArray'),N_a*N_zr,N_states); % sparse gpuArray
            elseif simoptions.gridinterplayer==1
                Policy_aprimez_jj=reshape(Policy_aprimez(:,:,jj),[],1); % the lower grid point of each state (the upper one is the next index)
                probupper_jj=reshape(probupper(:,:,jj),[],1);
                Gammatranspose_jj=sparse([Policy_aprimez_jj;Policy_aprimez_jj+1],IIind,[1-probupper_jj;probupper_jj],N_a*N_zr,N_states); % sparse gpuArray
            end
        else
            semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % linear index into pi_semiz_J_short and idxshort
            if expassetbranch==1
                Policy_aprimesemizz_jj=repelem(Policy_aprime_cpu(:,:,jj),1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short_jj)-1),1,N_probs)+zprimeoffset; % [N_states,N_probs*N_semizshort]: the a' points, each with every semiz'
                PolicyProbs_jj=repelem(PolicyProbs_cpu(:,:,jj),1,N_semizshort).*repmat(pi_semiz_J_short(semizindex_short_jj),1,N_probs);
                Gammatranspose_jj=sparse(gpuArray(Policy_aprimesemizz_jj),gpuArray(II2),gpuArray(PolicyProbs_jj),N_a*N_semiz*N_zr,N_states); % sparse gpuArray; sparse() accumulates repeated indexes (corners that coincide)
            elseif simoptions.gridinterplayer==0
                Policy_aprimesemizz_jj=repelem(Policy_aprime_cpu(:,1,jj),1,N_semizshort)+N_a*(idxshort(semizindex_short_jj)-1)+zprimeoffset;
                Gammatranspose_jj=sparse(gpuArray(Policy_aprimesemizz_jj),gpuArray(II2),gpuArray(pi_semiz_J_short(semizindex_short_jj)),N_a*N_semiz*N_zr,N_states); % sparse gpuArray
            elseif simoptions.gridinterplayer==1
                Policy_aprimesemizz_jj=repelem(Policy_aprime2_cpu(:,:,jj),1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short_jj)-1),1,2)+zprimeoffset;
                PolicyProbs_jj=repelem(PolicyProbs_cpu(:,:,jj),1,N_semizshort).*repmat(pi_semiz_J_short(semizindex_short_jj),1,2);
                Gammatranspose_jj=sparse(gpuArray(Policy_aprimesemizz_jj),gpuArray(II2),gpuArray(PolicyProbs_jj),N_a*N_semiz*N_zr,N_states); % sparse gpuArray; sparse() accumulates repeated indexes
            end
        end
        ncols=size(inflight,2);
        temp=Gammatranspose_jj*inflight; % policy step: now over (a',semiz',z)
        if N_zW>0
            temp=reshape(pagemtimes(reshape(temp,[N_a*N_semizr,N_zr,ncols]),pi_z_J_gpu(:,:,jj)),[N_a*N_semizr*N_zr,ncols]); % z step
        end
        if N_e>0
            temp=reshape(reshape(temp,[N_a*N_semizr*N_zr,1,ncols]).*reshape(pi_e_J_gpu(:,jj+1),[1,N_er]),[N_states,ncols]); % e step
        end
        inflight=temp;
        for gg=1:length(G_j0)
            pp=G_pp(gg); fns=G_fns{gg}; nfg=length(fns); c0=G_col(gg);
            if pp>1 % a restriction must hold at every age of the window
                inflight(:,c0:c0+G_ncol(gg)-1)=inflight(:,c0:c0+G_ncol(gg)-1).*RestrictionMaskA{pp-1}(:,jj+1);
            end
            xt=zeros(N_states,nfg,'gpuArray');
            for kf=1:nfg
                xt(:,kf)=ValuesW{fns(kf)}(:,jj+1)-cW(fns(kf),jj+1,pp);
            end
            m0=inflight(:,c0);
            mS=inflight(:,c0+1:c0+nfg); % (before this age is added)
            cc=c0+nfg;
            for kf=1:nfg
                for kg=kf:nfg
                    cc=cc+1;
                    inflight(:,cc)=inflight(:,cc)+xt(:,kf).*mS(:,kg)+xt(:,kg).*mS(:,kf)+xt(:,kf).*xt(:,kg).*m0;
                end
            end
            inflight(:,c0+1:c0+nfg)=mS+xt.*m0;
        end
    end

    % Pool over the start ages, each weighted by the mass of its window population (WMassAll): the pooled mean is the weighted mean of
    % the start-age means, the pooled covariance the weighted mean of the start-age covariances plus that of the products of the
    % deviations of the start-age means from the pooled ones (exact). Fields for a function with some selection on the page.
    for lw=1:nW
        M=windowM(lw);
        for pp=1:nwhichpages
            fnp=find(any(whichcombosW(:,:,pp,lw),2))';
            if isempty(fnp)
                continue
            end
            wn=cell(1,numFnsToEvaluate);
            for ff=1:numFnsToEvaluate
                wn{ff}=[AggVarNames{ff},'_',num2str(M),windowstr];
            end
            wj=reshape(WMassAll(:,pp,lw),[N_j,1]);
            wj(isnan(wj))=0;
            PMean=nan(numFnsToEvaluate,1);
            PCov=nan(numFnsToEvaluate,numFnsToEvaluate);
            PCorr=nan(numFnsToEvaluate,numFnsToEvaluate);
            for ff=fnp
                use=(wj>0) & ~isnan(reshape(WMean(ff,:,pp,lw),[N_j,1]));
                if sum(wj(use))>0
                    p_j=wj(use)/sum(wj(use));
                    mu_j=reshape(WMean(ff,use,pp,lw),[],1);
                    PMean(ff)=sum(p_j.*mu_j);
                end
            end
            for ff=fnp
                for gf=fnp
                    if gf<ff || isnan(PMean(ff)) || isnan(PMean(gf))
                        continue
                    end
                    if gf~=ff && whichcombosW(ff,gf,pp,lw)==0
                        continue
                    end
                    cv_j=reshape(WCov(ff,gf,:,pp,lw),[N_j,1]);
                    use=(wj>0) & ~isnan(cv_j);
                    if sum(wj(use))>0
                        p_j=wj(use)/sum(wj(use));
                        muf=reshape(WMean(ff,use,pp,lw),[],1);
                        mug=reshape(WMean(gf,use,pp,lw),[],1);
                        PCov(ff,gf)=sum(p_j.*(cv_j(use)+(muf-PMean(ff)).*(mug-PMean(gf))));
                        PCov(gf,ff)=PCov(ff,gf);
                    end
                end
            end
            for ff=fnp
                if ~isnan(PCov(ff,ff))
                    PCorr(ff,ff)=1;
                end
                for gf=fnp
                    if gf~=ff && ~isnan(PCov(ff,gf))
                        denom=sqrt(max(PCov(ff,ff),0))*sqrt(max(PCov(gf,gf),0));
                        if denom>1e-15
                            PCorr(ff,gf)=PCov(ff,gf)/denom;
                        end
                    end
                end
            end
            for ff=fnp
                PageOut{pp}.(wn{ff}).Mean=PMean(ff);
                PageOut{pp}.(wn{ff}).StdDeviation=sqrt(max(PCov(ff,ff),0));
                if isnan(PCov(ff,ff))
                    PageOut{pp}.(wn{ff}).StdDeviation=NaN;
                end
                PageOut{pp}.(wn{ff}).(wn{ff})=PCorr(ff,ff);
                for gf=fnp
                    if gf~=ff
                        PageOut{pp}.(wn{ff}).CovarianceWith.(wn{gf})=PCov(ff,gf);
                        PageOut{pp}.(wn{ff}).CorrelationWith.(wn{gf})=PCorr(ff,gf);
                    end
                end
            end
            PageOut{pp}.(['CovarianceMatrix_',num2str(M),windowstr])=PCov;
            PageOut{pp}.(['CorrelationMatrix_',num2str(M),windowstr])=PCorr;
            if pp>1
                PageOut{pp}.(['RestrictedSampleMass_',num2str(M),windowstr])=sum(wj); % the mass at the start ages that satisfies the restriction at every age of the window
            end
        end
    end
end
if ~has1
    % the single-period statistics are not wanted: no fields for them
    for pp=1:nwhichpages
        for ff=1:numFnsToEvaluate
            if isfield(PageOut{pp},AggVarNames{ff})
                PageOut{pp}=rmfield(PageOut{pp},AggVarNames{ff});
            end
        end
        PageOut{pp}=rmfield(PageOut{pp},{'CovarianceMatrix','CorrelationMatrix'});
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
