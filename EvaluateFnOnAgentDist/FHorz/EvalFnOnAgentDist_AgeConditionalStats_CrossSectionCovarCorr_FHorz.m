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
%
% simoptions.whichcombos ([numFnsToEvaluate, numFnsToEvaluate, number of age groups] of zeros/ones: in each age group the diagonal selects
% a function's Mean/StdDeviation, the off-diagonal a pair; only the upper triangle is read) selects what is computed; see below.
%
% simoptions.conditionalrestrictions: evaluate the same statistics conditional on each restriction being one (not zero). The
% restricted results are in AgeConditionalCrossSectionCorr.(restrictionname), with the same fields as the unrestricted ones, plus
% AgeConditionalCrossSectionCorr.(restrictionname).RestrictedSampleMass (1-by-N_j, the restricted mass at each age). Within an age group
% the restricted distribution is normalised to mass one over the group. With restrictions, simoptions.whichcombos can also be given as
% [numFnsToEvaluate, numFnsToEvaluate, number of age groups, 1+number of conditional restrictions] (page 1 unrestricted, pages 2:end
% the restrictions).
%
% simoptions.ntimeperiods (default 1; a vector, e.g. [1,2]) and simoptions.ntimeperiodstype ('average', the default, or 'sum'): the
% same statistics of the M-period sums/averages W_j=x_j+...+x_{j+M-1} of the functions, by window start age j: the Mean and
% StdDeviation of each function's window, and the CovarianceWith/CorrelationWith of every pair of functions over the same window (only
% windows of the same length are paired). The M>1 outputs are under [fnname,'_',M,'periodAverage'] (or 'periodSum'), e.g.
% AgeConditionalCrossSectionCorr.earnings_2periodAverage.CorrelationWith.medex_2periodAverage, with the matrices in
% CovarianceMatrix_2periodAverage and CorrelationMatrix_2periodAverage (nFn x nFn x N_j); the last M-1 start ages are NaN (the
% window runs past N_j). They are exact (the moments of the window sums are pushed through the transitions). Any M>1 needs
% simoptions.agegroupings=1:1:N_j and the z transition probabilities as simoptions.pi_z (as LifeCycleProfiles_FHorz_Case1). A
% conditional restriction must hold at every age of the window; the restricted window outputs have
% (restrictionname).RestrictedSampleMass_2periodAverage (1 x N_j), the mass at the start age that satisfies it throughout. With
% ntimeperiods of length L, simoptions.whichcombos may be [L*nFns, L*nFns, N_j, 1+nRestr] (block diagonal, blocks in the order of
% ntimeperiods; the off-diagonal blocks, pairs of different window lengths, must be zero); an nFns x nFns form applies to every M.


%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.agegroupings=1:1:N_j; % by default does each period separately
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    simoptions.ntimeperiods=1; % the statistics of the sums/averages over M consecutive periods, for each M in ntimeperiods (1 is the single period)
    simoptions.ntimeperiodstype='average'; % 'average' or 'sum' over the M periods
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
    if ~isfield(simoptions,'warnzerorestrictedmass')
        simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    end
    if ~isfield(simoptions,'ntimeperiods')
        simoptions.ntimeperiods=1; % the statistics of the sums/averages over M consecutive periods, for each M in ntimeperiods (1 is the single period)
    end
    if ~isfield(simoptions,'ntimeperiodstype')
        simoptions.ntimeperiodstype='average'; % 'average' or 'sum' over the M periods
    end
    % Internal options
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
end

if gpuDeviceCount==0
    error('AgeConditionalStats_CrossSectionCovarCorr_FHorz requires a GPU')
end

%%
l_a=length(n_a);
N_a=prod(n_a);

ngroups=length(simoptions.agegroupings);

%% Multi-period sums/averages: simoptions.ntimeperiods (the window lengths M, 1 is the single period) and simoptions.ntimeperiodstype
ntimeperiods=gather(simoptions.ntimeperiods(:)'); % the order is kept: it is the order of the blocks of whichcombos
if isempty(ntimeperiods) || ~isnumeric(ntimeperiods) || any(ntimeperiods<1) || any(ntimeperiods~=round(ntimeperiods))
    error('AgeConditionalStats_CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods must be a vector of positive integers (the number of periods summed/averaged over, 1 is the single period)')
end
if length(unique(ntimeperiods))<length(ntimeperiods)
    error('AgeConditionalStats_CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods must not contain repeated values')
end
if any(ntimeperiods>N_j)
    error('AgeConditionalStats_CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods cannot exceed N_j (no window of that many periods fits in the life cycle)')
end
if ~(ischar(simoptions.ntimeperiodstype) || isstring(simoptions.ntimeperiodstype)) || ~any(strcmp(simoptions.ntimeperiodstype,{'average','sum'}))
    error('AgeConditionalStats_CrossSectionCovarCorr_FHorz: simoptions.ntimeperiodstype must be ''average'' or ''sum''')
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
if nW>0 && ~isequal(reshape(gather(simoptions.agegroupings),1,[]),1:1:N_j)
    error('AgeConditionalStats_CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods with M>1 (multi-period sums/averages) can only be used with simoptions.agegroupings=1:1:N_j (the default); the statistics of the windows are by start age j')
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
numFnsToEvaluate=length(FnsToEvaluate);

%% Conditional restrictions: the number of 'pages' (1 unrestricted, plus one per restriction)
useCondlRest=0;
nwhichpages=1;
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);
    nwhichpages=1+length(CondlRestnFnNames);
end

%% simoptions.whichcombos: which functions, pairs and age groups to compute
% [numFnsToEvaluate, numFnsToEvaluate, ngroups] of zeros/ones. In each age group the diagonal (ff,ff) selects the Mean and StdDeviation
% of function ff (and its variance on the CovarianceMatrix diagonal) and the off-diagonal (ff1,ff2) selects the covariance and
% correlation of the pair; only the upper triangle (ff1<=ff2) is read. A [numFnsToEvaluate, numFnsToEvaluate] input is applied to every
% age group. In an age group a function with nothing selected is not evaluated; a selected pair has both its functions evaluated, and the
% own stats (Mean, StdDeviation, variance, self-correlation) of every evaluated function are reported, as they are byproducts of the pairs
% (so the diagonal matters only for a function with no selected pair in that age group). Skipped pairs are NaN. Default all ones.
% With conditional restrictions it is [numFnsToEvaluate, numFnsToEvaluate, ngroups, 1+number of conditional restrictions]: page 1 is
% the unrestricted stats, pages 2:end the restrictions in the fieldnames order of simoptions.conditionalrestrictions. A
% [numFnsToEvaluate, numFnsToEvaluate] or [numFnsToEvaluate, numFnsToEvaluate, ngroups] input with restrictions is applied to every page.
% With simoptions.ntimeperiods of length L>1 (or without 1 in it) whichcombos is block diagonal, [L*nFns, L*nFns, ngroups, 1+nRestr],
% one nFns x nFns block per window length in the order of ntimeperiods (an nFns x nFns form, with or without the age and page
% dimensions, applies to every block); the off-diagonal blocks pair windows of different lengths and must be zero. Internally
% whichcombos is the M=1 block (zeros when 1 is not in ntimeperiods) and whichcombosW(:,:,:,:,lw) the block of windowM(lw).
if ~(nblocks==1 && has1)
    nRwc=nblocks*numFnsToEvaluate;
    if ~isfield(simoptions,'whichcombos')
        wcIn=ones(numFnsToEvaluate,numFnsToEvaluate,ngroups,nwhichpages);
    else
        wcIn=simoptions.whichcombos;
        if ~(isnumeric(wcIn) || islogical(wcIn)) || any(wcIn(:)~=0 & wcIn(:)~=1)
            error('simoptions.whichcombos must contain only zeros and ones')
        end
        wcIn=double(wcIn);
    end
    if size(wcIn,1)==numFnsToEvaluate && size(wcIn,2)==numFnsToEvaluate && nRwc~=numFnsToEvaluate % one block for every window length
        temp=zeros(nRwc,nRwc,size(wcIn,3),size(wcIn,4));
        for bb=1:nblocks
            temp((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,(bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:)=wcIn;
        end
        wcIn=temp;
    end
    if ismatrix(wcIn) && isequal(size(wcIn),[nRwc,nRwc])
        wcIn=repmat(wcIn,[1,1,ngroups]); % one matrix: apply to every age group
    end
    if ndims(wcIn)<=3 && nwhichpages>1 && isequal(size(wcIn,1:3),[nRwc,nRwc,ngroups])
        wcIn=repmat(wcIn,[1,1,1,nwhichpages]); % one set of age groups with restrictions: apply to every page
    end
    if ~isequal(size(wcIn,1:4),[nRwc,nRwc,ngroups,nwhichpages]) || ndims(wcIn)>4
        error(['simoptions.whichcombos must be of size [',num2str(nRwc),',',num2str(nRwc),',',num2str(ngroups),',',num2str(nwhichpages),'] (length of simoptions.ntimeperiods times the number of FnsToEvaluate, twice, number of age groups, 1+number of conditional restrictions; block diagonal), or have ',num2str(numFnsToEvaluate),' rows and columns to apply to every window length'])
    end
    for pp=1:nwhichpages
        for kk=1:ngroups
            wcIn(:,:,kk,pp)=max(triu(wcIn(:,:,kk,pp)),triu(wcIn(:,:,kk,pp))'); % symmetric, from the upper triangle
        end
    end
    blockmask=kron(eye(nblocks),ones(numFnsToEvaluate));
    if any(wcIn.*(1-blockmask)~=0,'all')
        error('simoptions.whichcombos selects a pair of windows of different lengths (an off-diagonal block); only windows of the same length are paired')
    end
    if has1
        bb=find(ntimeperiods==1);
        whichcombos=wcIn((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,(bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:);
    else
        whichcombos=zeros(numFnsToEvaluate,numFnsToEvaluate,ngroups,nwhichpages);
    end
    whichcombosW=zeros(numFnsToEvaluate,numFnsToEvaluate,ngroups,nwhichpages,nW);
    for lw=1:nW
        bb=find(ntimeperiods==windowM(lw));
        whichcombosW(:,:,:,:,lw)=wcIn((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,(bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:);
        whichcombosW(:,:,N_j-windowM(lw)+2:N_j,:,lw)=0; % no window of M periods starts after age N_j-M+1
    end
elseif ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,numFnsToEvaluate,ngroups,nwhichpages); % whichcombos here is [nFns, nFns, nAgeGroups, 1+nRestr]
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ismatrix(whichcombos) && isequal(size(whichcombos),[numFnsToEvaluate,numFnsToEvaluate])
        whichcombos=repmat(whichcombos,[1,1,ngroups]); % one matrix: apply to every age group
    end
    if ndims(whichcombos)<=3 && nwhichpages>1 && isequal(size(whichcombos,1:3),[numFnsToEvaluate,numFnsToEvaluate,ngroups])
        whichcombos=repmat(whichcombos,[1,1,1,nwhichpages]); % one set of age groups with restrictions: apply to every page
    end
    if ~isequal(size(whichcombos,1:4),[numFnsToEvaluate,numFnsToEvaluate,ngroups,nwhichpages]) || ndims(whichcombos)>4
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(numFnsToEvaluate),',',num2str(ngroups),',',num2str(nwhichpages),'] (number of FnsToEvaluate, twice, number of age groups, 1+number of conditional restrictions; the fourth dimension is dropped when there are no conditional restrictions)'])
    end
    whichcombos=double(whichcombos);
    for pp=1:nwhichpages
        for kk=1:ngroups
            whichcombos(:,:,kk,pp)=max(triu(whichcombos(:,:,kk,pp)),triu(whichcombos(:,:,kk,pp))'); % symmetric, from the upper triangle
        end
    end
end

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

%% If there are any conditional restrictions, set up for these
% Code works by evaluating the restriction and imposing this on the distribution (and renormalizing it within each age group).
if useCondlRest==1
    restrictedsamplemass=nan(length(CondlRestnFnNames),N_j);
    RestrictionMask=cell(length(CondlRestnFnNames),1); % each restriction kept as a logical mask over the (a,z) x j grid (1 byte per point); the restricted weights are formed from it where used

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
            RestrictionValues=false(N_a,N_j,'gpuArray');
        else
            RestrictionValues=false(N_a,N_z,N_j,'gpuArray');
        end
        for jj=1:N_j
            % Get parameter values for Conditional Restriction functions
            CondlRestnFnParamsCell=CreateCellFromParams(Parameters,CondlRestnFnParamNames,jj);

            % Compute the restrictions
            if N_z==0
                RestrictionValues(:,jj)=logical(EvalFnOnAgentDist_Grid(CondlRestnFn,CondlRestnFnParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,0,a_gridvals,[]));
            else
                RestrictionValues(:,:,jj)=logical(EvalFnOnAgentDist_Grid(CondlRestnFn,CondlRestnFnParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj)));
            end
        end
        if N_z>0
            RestrictionValues=reshape(RestrictionValues,[N_a*N_z,N_j]);
        end

        % Keep the restricted mass at each age, and the mask
        RestrictionMask{rr}=RestrictionValues;
        restrictedsamplemass(rr,:)=sum(StationaryDist.*RestrictionMask{rr},1); % mass at each age that satisfies the restriction (includes the age weights)

        if all(restrictedsamplemass(rr,:)==0) && simoptions.warnzerorestrictedmass==2
            warning('One of the conditional restrictions evaluates to a zero mass (at all j)')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
        end
    end
end

%% Preallocate output
% Each page is filled as its own output structure; page 1 (unrestricted) becomes AgeConditionalCrossSectionCorr and pages 2:end AgeConditionalCrossSectionCorr.(restrictionname)
PageOut=cell(nwhichpages,1);
for pp=1:nwhichpages
    PageOut{pp}=struct();
    for ff=1:numFnsToEvaluate
        PageOut{pp}.(AggVarNames{ff}).Mean=nan(1,ngroups,'gpuArray');
        PageOut{pp}.(AggVarNames{ff}).StdDeviation=nan(1,ngroups,'gpuArray');
    end
    for ff1=1:numFnsToEvaluate
        for ff2=1:numFnsToEvaluate
            if ff1==ff2
                PageOut{pp}.(AggVarNames{ff1}).(AggVarNames{ff2})=ones(1,ngroups,'gpuArray');
            else
                PageOut{pp}.(AggVarNames{ff1}).CovarianceWith.(AggVarNames{ff2})=nan(1,ngroups,'gpuArray');
                PageOut{pp}.(AggVarNames{ff1}).CorrelationWith.(AggVarNames{ff2})=nan(1,ngroups,'gpuArray');
            end
        end
    end
    PageOut{pp}.CovarianceMatrix=nan(numFnsToEvaluate,numFnsToEvaluate,ngroups,'gpuArray');
    PageOut{pp}.CorrelationMatrix=nan(numFnsToEvaluate,numFnsToEvaluate,ngroups,'gpuArray');
end

%% Loop over age groupings and compute the cross-sectional covar/corr for each
for kk=1:ngroups
    j1=simoptions.agegroupings(kk);
    if kk<ngroups
        jend=simoptions.agegroupings(kk+1)-1;
    else
        jend=N_j;
    end
    jspan=jend-j1+1;

    % The within-group distribution (normalised separately for each page below)
    if N_z==0
        StationaryDistVec_kk=reshape(StationaryDist(:,j1:jend),[N_a*jspan,1]);
    else
        StationaryDistVec_kk=reshape(StationaryDist(:,j1:jend),[N_a*N_z*jspan,1]);
    end
    N_total_kk=length(StationaryDistVec_kk);

    % The mass of each page in this age group; a page with no mass in this age group is skipped, and everything stays NaN from preallocation (the self-correlations are set to NaN below)
    massvec_kk=zeros(nwhichpages,1);
    massvec_kk(1)=gather(sum(StationaryDistVec_kk));
    for rr=1:nwhichpages-1
        massvec_kk(1+rr)=gather(sum(restrictedsamplemass(rr,j1:jend)));
    end
    pagewanted_kk=(massvec_kk>0);
    fnwanted_kk=reshape(any(whichcombos(:,:,kk,pagewanted_kk),2),[numFnsToEvaluate,sum(pagewanted_kk)]);
    fnwanted_kk=any(fnwanted_kk,2); % the functions evaluated in this age group (wanted on some page with mass)

    %% Step 1: evaluate every function on the within-group grid
    ValuesCell=cell(numFnsToEvaluate,1);
    if N_z==0
        for ff=1:numFnsToEvaluate
            if ~fnwanted_kk(ff) % nothing involving this function is wanted in this age group
                continue
            end
            Values=nan(N_a,jspan,'gpuArray');
            for jj=j1:jend
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                Values(:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            end
            ValuesCell{ff}=reshape(Values,[N_total_kk,1]);
        end
    else
        for ff=1:numFnsToEvaluate
            if ~fnwanted_kk(ff) % nothing involving this function is wanted in this age group
                continue
            end
            Values=nan(N_a,N_z,jspan,'gpuArray');
            for jj=j1:jend
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                Values(:,:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
            end
            ValuesCell{ff}=reshape(Values,[N_total_kk,1]);
        end
    end

    for pp=1:nwhichpages
        if ~pagewanted_kk(pp) % no mass in this age group: NaN everywhere, including the self-correlations (preallocated as ones)
            for ff=1:numFnsToEvaluate
                PageOut{pp}.(AggVarNames{ff}).(AggVarNames{ff})(kk)=NaN;
            end
            continue
        end
        % The normalized within-group distribution of this page
        if pp==1
            PageDist_kk=StationaryDistVec_kk./massvec_kk(1);
        else
            PageDist_kk=StationaryDistVec_kk.*reshape(RestrictionMask{pp-1}(:,j1:jend),[N_total_kk,1])./massvec_kk(pp); % the restricted mass at each point of the age group, normalised to mass one over the group (so each age of the group is weighted by its restricted mass)
        end

        %% Step 2: per-function Mean and StdDev under the within-group distribution
        Means_kk=zeros(numFnsToEvaluate,1,'gpuArray');
        StdDevs_kk=zeros(numFnsToEvaluate,1,'gpuArray');
        for ff=1:numFnsToEvaluate
            if ~any(whichcombos(ff,:,kk,pp)) % nothing involving this function is wanted in this age group: its own stats stay NaN, including the self-correlation
                PageOut{pp}.(AggVarNames{ff}).(AggVarNames{ff})(kk)=NaN;
                continue
            end
            Means_kk(ff)=sum(ValuesCell{ff}.*PageDist_kk);
            StdDevs_kk(ff)=sqrt(sum(PageDist_kk.*((ValuesCell{ff}-Means_kk(ff).*ones(N_total_kk,1,'gpuArray')).^2)));

            PageOut{pp}.(AggVarNames{ff}).Mean(kk)=Means_kk(ff);
            PageOut{pp}.(AggVarNames{ff}).StdDeviation(kk)=StdDevs_kk(ff);
        end

        %% Step 3: upper-triangular covariance/correlation
        for ff1=1:numFnsToEvaluate
            for ff2=ff1:numFnsToEvaluate
                if ff1==ff2 % the own stats of a function evaluated in this age group are byproducts of the pairs and are reported regardless of the diagonal
                    if any(whichcombos(ff1,:,kk,pp)) % (a function not evaluated in this age group keeps NaN on the diagonal)
                        PageOut{pp}.CovarianceMatrix(ff1,ff2,kk)=StdDevs_kk(ff1)^2;
                        PageOut{pp}.CorrelationMatrix(ff1,ff2,kk)=1;
                    end
                elseif whichcombos(ff1,ff2,kk,pp)==1 % the pair is wanted in this age group (else it stays NaN)
                    CoVar=sum((ValuesCell{ff1}-Means_kk(ff1)*ones(N_total_kk,1,'gpuArray')).*(ValuesCell{ff2}-Means_kk(ff2)*ones(N_total_kk,1,'gpuArray')).*PageDist_kk);
                    Corr=CoVar/(StdDevs_kk(ff1)*StdDevs_kk(ff2));

                    PageOut{pp}.(AggVarNames{ff1}).CovarianceWith.(AggVarNames{ff2})(kk)=CoVar;
                    PageOut{pp}.(AggVarNames{ff1}).CorrelationWith.(AggVarNames{ff2})(kk)=Corr;

                    PageOut{pp}.CovarianceMatrix(ff1,ff2,kk)=CoVar;
                    PageOut{pp}.CorrelationMatrix(ff1,ff2,kk)=Corr;
                end
            end
        end

        %% Step 4: mirror to the lower-triangular part
        for ff1=1:numFnsToEvaluate
            for ff2=1:ff1-1
                PageOut{pp}.(AggVarNames{ff1}).CovarianceWith.(AggVarNames{ff2})(kk)=PageOut{pp}.(AggVarNames{ff2}).CovarianceWith.(AggVarNames{ff1})(kk);
                PageOut{pp}.(AggVarNames{ff1}).CorrelationWith.(AggVarNames{ff2})(kk)=PageOut{pp}.(AggVarNames{ff2}).CorrelationWith.(AggVarNames{ff1})(kk);

                PageOut{pp}.CovarianceMatrix(ff1,ff2,kk)=PageOut{pp}.CovarianceMatrix(ff2,ff1,kk);
                PageOut{pp}.CorrelationMatrix(ff1,ff2,kk)=PageOut{pp}.CorrelationMatrix(ff2,ff1,kk);
            end
        end
    end
end

%% Assemble the output

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
    if (isfield(simoptions,'experienceasset') && simoptions.experienceasset>=1) || (isfield(simoptions,'inheritanceasset') && simoptions.inheritanceasset==1) || (isfield(simoptions,'residualasset') && simoptions.residualasset==1)
        error('AgeConditionalStats_CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods with M>1 is not yet implemented with experience, inheritance or residual assets, ask on forum if you need this')
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
            error('AgeConditionalStats_CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods with M>1 needs the transition probabilities of z, give them as simoptions.pi_z (the same as the pi_z input of StationaryDist_FHorz_Case1)')
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
    PolicyW=gpuArray(reshape(Policy,[size(Policy,1),N_a,N_semizr*N_zr*N_er,N_j]));
    if l_a==1
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:),1);
    elseif l_a==2
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:)+n_a(1)*(PolicyW(l_d+2,:,:,:)-1),1);
    elseif l_a==3
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:)+n_a(1)*(PolicyW(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyW(l_d+3,:,:,:)-1),1);
    elseif l_a==4
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:)+n_a(1)*(PolicyW(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyW(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyW(l_d+4,:,:,:)-1),1);
    else
        error('AgeConditionalStats_CrossSectionCovarCorr_FHorz: simoptions.ntimeperiods with M>1 cannot handle length(n_a)>4, contact me if you need this')
    end
    if simoptions.gridinterplayer==1
        L2index=PolicyW(end-1,:,:,:); % L2 index (end-1 because end is L2flag)
        L2flag=PolicyW(end,:,:,:);
        L2index(L2flag==1)=1;                        % force all weight to lower grid point
        L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
        probupper=shiftdim((L2index-1)/(simoptions.ngridinterp+1),1); % [N_a,N_semizr*N_zr*N_er,N_j]: probability of the upper grid point
    end
    if N_semiz==0
        zindex=N_a*repmat(gpuArray(0:1:N_zr-1),1,N_er);
        Policy_aprimez=Policy_aprime+zindex; % [N_a,N_zr*N_er,N_j], index into (a',z)
        if simoptions.gridinterplayer==0
            IIind=gpuArray((1:1:N_states)');
        elseif simoptions.gridinterplayer==1
            IIind=gpuArray(repmat((1:1:N_states)',2,1));
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
        Policy_aprime_cpu=gather(reshape(Policy_aprime,[N_states,1,N_j]));
        if simoptions.gridinterplayer==0
            II2=repelem((1:1:N_states)',1,N_semizshort);
        elseif simoptions.gridinterplayer==1
            Policy_aprime2_cpu=cat(2,Policy_aprime_cpu,Policy_aprime_cpu+1); % [N_states,2,N_j]: the lower grid point and the one above it
            probupper_cpu=gather(reshape(probupper,[N_states,1,N_j]));
            PolicyProbs_cpu=cat(2,1-probupper_cpu,probupper_cpu); % [N_states,2,N_j]
            II2=repelem((1:1:N_states)',1,2*N_semizshort);
        end
    end

    % What is wanted: at (start age, page) the functions with some selection (their window Mean/StdDeviation) for each window length
    wantFn=reshape(any(whichcombosW,2),[numFnsToEvaluate,N_j,nwhichpages,nW]); % [nFns,N_j,pages,nW]
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
                    mp=StationaryDist(:,jj);
                else
                    mp=StationaryDist(:,jj).*RestrictionMask{pp-1}(:,jj);
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
                m0=StationaryDist(:,jj);
            else
                m0=StationaryDist(:,jj).*RestrictionMask{pp-1}(:,jj); % the restriction at the start age (and below, at every later age of the window)
            end
            if sum(m0)==0
                if pp>1 && sum(StationaryDist(:,jj))>0
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
                    if kg~=kf && whichcombosW(ff,gf,j0,pp,lw)==1
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
            if simoptions.gridinterplayer==0
                Gammatranspose_jj=sparse(reshape(Policy_aprimez(:,:,jj),[],1),IIind,ones(N_states,1,'gpuArray'),N_a*N_zr,N_states); % sparse gpuArray
            elseif simoptions.gridinterplayer==1
                Policy_aprimez_jj=reshape(Policy_aprimez(:,:,jj),[],1); % the lower grid point of each state (the upper one is the next index)
                probupper_jj=reshape(probupper(:,:,jj),[],1);
                Gammatranspose_jj=sparse([Policy_aprimez_jj;Policy_aprimez_jj+1],IIind,[1-probupper_jj;probupper_jj],N_a*N_zr,N_states); % sparse gpuArray
            end
        else
            semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % linear index into pi_semiz_J_short and idxshort
            if simoptions.gridinterplayer==0
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
                inflight(:,c0:c0+G_ncol(gg)-1)=inflight(:,c0:c0+G_ncol(gg)-1).*RestrictionMask{pp-1}(:,jj+1);
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

    % Assemble the window outputs into the pages: fields for a function with some selection on the page (for that window length)
    for lw=1:nW
        M=windowM(lw);
        for pp=1:nwhichpages
            fnp=find(any(reshape(wantFn(:,:,pp,lw),[numFnsToEvaluate,N_j]),2))';
            if isempty(fnp)
                continue
            end
            wn=cell(1,numFnsToEvaluate);
            for ff=1:numFnsToEvaluate
                wn{ff}=[AggVarNames{ff},'_',num2str(M),windowstr];
            end
            for ff=fnp
                PageOut{pp}.(wn{ff}).Mean=WMean(ff,:,pp,lw);
                PageOut{pp}.(wn{ff}).StdDeviation=WStd(ff,:,pp,lw);
                selfc=nan(1,N_j);
                selfc(~isnan(WMean(ff,:,pp,lw)))=1; % the self-correlation where the window stats were computed
                PageOut{pp}.(wn{ff}).(wn{ff})=selfc;
                for gf=fnp
                    if gf~=ff
                        cv=reshape(WCov(min(ff,gf),max(ff,gf),:,pp,lw),[1,N_j]); % (computed from the upper triangle)
                        cr=reshape(WCorr(min(ff,gf),max(ff,gf),:,pp,lw),[1,N_j]);
                        PageOut{pp}.(wn{ff}).CovarianceWith.(wn{gf})=cv;
                        PageOut{pp}.(wn{ff}).CorrelationWith.(wn{gf})=cr;
                    end
                end
            end
            CVm=WCov(:,:,:,pp,lw);
            CRm=WCorr(:,:,:,pp,lw);
            for ff=1:numFnsToEvaluate
                for gf=1:ff-1
                    CVm(ff,gf,:)=CVm(gf,ff,:);
                    CRm(ff,gf,:)=CRm(gf,ff,:);
                end
            end
            PageOut{pp}.(['CovarianceMatrix_',num2str(M),windowstr])=CVm;
            PageOut{pp}.(['CorrelationMatrix_',num2str(M),windowstr])=CRm;
            if pp>1
                PageOut{pp}.(['RestrictedSampleMass_',num2str(M),windowstr])=reshape(WRSMass(:,pp,lw),[1,N_j]);
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

AgeConditionalCrossSectionCorr=PageOut{1};
if useCondlRest==1
    for rr=1:length(CondlRestnFnNames)
        AgeConditionalCrossSectionCorr.(CondlRestnFnNames{rr})=PageOut{1+rr};
        AgeConditionalCrossSectionCorr.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr,:);
    end
end

AgeConditionalCrossSectionCorr.Notes='The CovarianceMatrix and CorrelationMatrix are nFn x nFn x ngroups; the third index aligns with simoptions.agegroupings. They essentially duplicate the individual named correlations and covariances, but depending on what you want to do the matrix or the individual named pairs might be easier to use so both are created.';

end
