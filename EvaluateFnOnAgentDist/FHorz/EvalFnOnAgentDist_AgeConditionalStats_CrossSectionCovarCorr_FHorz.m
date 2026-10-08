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


%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.agegroupings=1:1:N_j; % by default does each period separately
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
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,numFnsToEvaluate,ngroups,nwhichpages);
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
AgeConditionalCrossSectionCorr=PageOut{1};
if useCondlRest==1
    for rr=1:length(CondlRestnFnNames)
        AgeConditionalCrossSectionCorr.(CondlRestnFnNames{rr})=PageOut{1+rr};
        AgeConditionalCrossSectionCorr.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr,:);
    end
end

AgeConditionalCrossSectionCorr.Notes='The CovarianceMatrix and CorrelationMatrix are nFn x nFn x ngroups; the third index aligns with simoptions.agegroupings. They essentially duplicate the individual named correlations and covariances, but depending on what you want to do the matrix or the individual named pairs might be easier to use so both are created.';

end
