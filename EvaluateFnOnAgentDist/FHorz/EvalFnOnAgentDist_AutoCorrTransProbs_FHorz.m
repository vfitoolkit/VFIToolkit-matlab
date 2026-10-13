function CorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_FHorz(StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, pi_z, simoptions)
% Returns stats on (auto) correlation and transition probabilities for FHorz models.
% Auto-correlation/-covariance are reported per-age (j -> j+1), so length N_j-1.
% Mean and StdDeviation are reported per-age, so length N_j.
%
% Use simoptions.transprobs={'name1','name2',...} (cell of FnsToEval names) to
% request transition probabilities for those functions (none by default).
%
% Use simoptions.timehorizons=[2,5] (vector of horizons k>=2) to also get the
% auto-covariance/-correlation between age j and age j+k (the horizon 1 is
% always computed). Horizon k is reported in fields with the suffix _kK, e.g.
% AutoCovariance_k2 is 1 x (N_j-2) with index jj the pair (age jj, age jj+2).
%
% Use simoptions.conditionalrestrictions (structure of functions, same form as
% FnsToEvaluate, returning 0/1) to also get everything conditional on a
% restriction: means/std devs at age j are over those satisfying the
% restriction at age j, and the auto-covariance between ages j and j+k is over
% those satisfying the restriction at BOTH ages j and j+k (the 'pairs'; e.g.
% alive at both ages). Reported under CorrTransProbs.(restrictionname).(fnname).
%
% Outputs (per FnsToEvaluate field):
%   .Mean             1 x N_j
%   .StdDeviation     1 x N_j
%   .AutoCovariance   1 x (N_j-1)   Cov(x_j, x_{j+1}), with x_j centered on Mean(j) and x_{j+1} on Mean(j+1)
%   .AutoCorrelation  1 x (N_j-1)   AutoCovariance/(StdDeviation(j)*StdDeviation(j+1))
%   .AutoCovariance_kK, .AutoCorrelation_kK   1 x (N_j-K), for each K in simoptions.timehorizons
%   .TransitionProbs  cell {N_j-1} of n_fvals_j x n_fvals_{j+1} matrices (default),
%                     or n_fvals x n_fvals x (N_j-1) array when simoptions.transprobquantiles is set
%   .TransitionValues_j, .TransitionValues_jplus1  cells {N_j-1} of the unique function
%                     values labelling the rows/columns of TransitionProbs{jj}
%                     (not provided when simoptions.transprobquantiles is set)
%   .TransitionMass_j cell {N_j-1}, within-age mass of each origin bin (row) of
%                     TransitionProbs{jj}; multiply by the age weight to get population
%                     mass (not provided when simoptions.transprobquantiles is set)
% Outputs per conditional restriction, CorrTransProbs.(restrictionname):
%   .RestrictedSampleMass   1 x N_j, population mass satisfying the restriction at each age (includes the age weights)
%   .(fnname).Mean, .StdDeviation   1 x N_j, over those satisfying the restriction at that age
%   .(fnname).AutoCovariance, .AutoCorrelation   1 x (N_j-1), over the pairs (satisfy the restriction at both j and j+1),
%                     centered on the pair means (so this is the covariance/correlation of the pair population)
%   .(fnname).PairMass   1 x (N_j-1), population mass of the pairs (an age-j agent counts if it satisfies the
%                     restriction at j and will satisfy it at j+1; includes the age-j weight). Zero when nobody at
%                     age j satisfies the restriction; NaN only when age j has no mass at all.
%   .(fnname).PairMean_j, .PairMean_jplusk, .PairStdDeviation_j, .PairStdDeviation_jplusk   1 x (N_j-1),
%                     the means/std devs of x_j and of x_{j+1} in the pair population
%   and the same with suffix _kK for each K in simoptions.timehorizons (all 1 x (N_j-K))
%   TransitionProbs are not computed under conditional restrictions.
%
% Use simoptions.ntimeperiods (default 1) to get the same statistics of the sum or average of a function over M consecutive
% periods: the statistic of a person at age j is W_j=x_j+...+x_{j+M-1} (simoptions.ntimeperiodstype='sum'), or that divided
% by M (simoptions.ntimeperiodstype='average', the default). A vector, e.g. simoptions.ntimeperiods=[1,2,5], gives each of
% them (M=1 is the single-period statistic). AutoCovariance_kK/AutoCorrelation_kK is then between the window starting at
% age j and the window starting at age j+K, where K is in model periods (the same horizons as for M=1, simoptions.timehorizons,
% and horizon 1 is always computed; when K<M the two windows share M-K periods, and a warning says so). Conditional
% restrictions are evaluated at the two window start ages, j and j+K (as for M=1, where these are the two ages of the pair).
% The M=1 outputs are under CorrTransProbs.(fnname) as always; the M>1 outputs are under CorrTransProbs.([fnname,'_',M,'periodAverage'])
% (or 'periodSum'), e.g. CorrTransProbs.earnings_2periodAverage, with the same fields: Mean and StdDeviation are of W_j
% (1 x N_j), the auto-covariances 1 x (N_j-K), and the last M-1 entries of each are NaN (the window runs past age N_j).
% TransitionProbs are for M=1 only (simoptions.transprobs needs 1 in simoptions.ntimeperiods).
% With ntimeperiods of length L, simoptions.whichcombos may have L*numFnsToEvaluate rows, in blocks of numFnsToEvaluate
% in the order of simoptions.ntimeperiods (row (ll-1)*numFnsToEvaluate+ff is function ff with M=ntimeperiods(ll)); a
% whichcombos with numFnsToEvaluate rows applies to every M.
%
% Note: an age with zero mass (e.g. before the entry age of a cohort started using
% simoptions.jequaloneDistAge) gets NaN in every output that involves it.
%
% Not yet implemented (will error):
%   simoptions.agegroupings non-default  -- error
%
% simoptions.whichcombos ([numFnsToEvaluate, N_j, 1+number of conditional restrictions] of zeros/ones) selects which (fn, start age,
% restriction) combinations are computed; see the whichcombos section below.

%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.transprobs=zeros(length(fieldnames(FnsToEvaluate)),1);
    simoptions.timehorizons=[]; % multi-period horizons (horizon 1 is always calculated)
    simoptions.transprobquantiles=[];
    simoptions.agegroupings=1:1:N_j; % age bins -- not yet implemented (default = each age separately)
    simoptions.ntimeperiods=1; % the statistics of the sum/average over M consecutive periods, for each M in ntimeperiods (1 is the single period)
    simoptions.ntimeperiodstype='average'; % 'average' or 'sum' over the M periods
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    % Other endogenous states
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    simoptions.experienceasset=0;
    simoptions.inheritanceasset=0;
    % Internal options
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0;
else
    if ~isfield(simoptions,'transprobs')
        simoptions.transprobs=zeros(length(fieldnames(FnsToEvaluate)),1);
    end
    if ~isfield(simoptions,'timehorizons')
        simoptions.timehorizons=[];
    end
    if ~isfield(simoptions,'transprobquantiles')
        simoptions.transprobquantiles=[];
    end
    if ~isfield(simoptions,'ntimeperiods')
        simoptions.ntimeperiods=1; % the statistics of the sum/average over M consecutive periods, for each M in ntimeperiods (1 is the single period)
    end
    if ~isfield(simoptions,'ntimeperiodstype')
        simoptions.ntimeperiodstype='average'; % 'average' or 'sum' over the M periods
    end
    % Model setup
    if ~isfield(simoptions,'agegroupings')
        simoptions.agegroupings=1:1:N_j;
    end
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
    % Other endogenous states
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    if ~isfield(simoptions,'inheritanceasset')
        simoptions.inheritanceasset=0;
    end
    % Internal options
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
end

if ~isequal(simoptions.agegroupings,1:1:N_j)
    error('AutoCorrTransProbs_FHorz: simoptions.agegroupings (age bins) not yet implemented; will implement later')
end

%% Time horizons: horizon 1 is always done, the others come from simoptions.timehorizons
if ~isempty(simoptions.timehorizons)
    if any(simoptions.timehorizons<1) || any(simoptions.timehorizons~=round(simoptions.timehorizons))
        error('AutoCorrTransProbs_FHorz: simoptions.timehorizons must be positive integers')
    end
    if any(simoptions.timehorizons>N_j-1)
        error('AutoCorrTransProbs_FHorz: simoptions.timehorizons cannot exceed N_j-1 (there is no pair of ages that far apart)')
    end
end
horizons=unique([1,gather(simoptions.timehorizons(:)')]); % sorted, starts with 1
nhorizons=length(horizons);
Kmax=horizons(end);
horizonstr=cell(1,nhorizons); % suffix of the output field names
horizonstr{1}=''; % horizon 1 has no suffix (AutoCovariance, AutoCorrelation, ...)
for hh=2:nhorizons
    horizonstr{hh}=['_k',num2str(horizons(hh))];
end

%% Multi-period sums/averages: simoptions.ntimeperiods (the window lengths M, 1 is the single period) and simoptions.ntimeperiodstype
ntimeperiods=gather(simoptions.ntimeperiods(:)'); % the order is kept: it is the order of the row blocks of whichcombos
if isempty(ntimeperiods) || ~isnumeric(ntimeperiods) || any(ntimeperiods<1) || any(ntimeperiods~=round(ntimeperiods))
    error('AutoCorrTransProbs_FHorz: simoptions.ntimeperiods must be a vector of positive integers (the number of periods summed/averaged over, 1 is the single period)')
end
if length(unique(ntimeperiods))<length(ntimeperiods)
    error('AutoCorrTransProbs_FHorz: simoptions.ntimeperiods must not contain repeated values')
end
if any(ntimeperiods>N_j)
    error('AutoCorrTransProbs_FHorz: simoptions.ntimeperiods cannot exceed N_j (no window of that many periods fits in the life cycle)')
end
if ~(ischar(simoptions.ntimeperiodstype) || isstring(simoptions.ntimeperiodstype)) || ~any(strcmp(simoptions.ntimeperiodstype,{'average','sum'}))
    error('AutoCorrTransProbs_FHorz: simoptions.ntimeperiodstype must be ''average'' or ''sum''')
end
nblocks=length(ntimeperiods); % the number of row blocks of whichcombos
has1=any(ntimeperiods==1); % the single-period statistics are wanted
windowM=ntimeperiods(ntimeperiods>1); % the window lengths M>1, in the order given
nW=length(windowM);
if nW>0
    Mmax=max(windowM);
else
    Mmax=0;
end
if strcmp(simoptions.ntimeperiodstype,'average')
    windowstr='periodAverage';
else
    windowstr='periodSum';
end
% Overlapping windows: the window starting at j and the one starting at j+K share M-K periods when K<M (horizon 1 is always computed)
if nW>0
    overlapstr='';
    for lw=1:nW
        Koverlap=horizons(horizons<windowM(lw));
        if ~isempty(Koverlap)
            overlapstr=[overlapstr,' M=',num2str(windowM(lw)),' at horizons K=',num2str(Koverlap),';']; %#ok<AGROW>
        end
    end
    if ~isempty(overlapstr)
        warning(['AutoCorrTransProbs_FHorz: some of the windows are overlapping (the windows starting at ages j and j+K share M-K periods when the horizon K is less than the number of periods M; horizon 1 is always computed):',overlapstr])
    end
end

%%
N_a=prod(n_a);

if isempty(n_d) || prod(n_d)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% Exogenous shocks
N_z=prod(n_z);
N_e=prod(simoptions.n_e);
N_semiz=prod(simoptions.n_semiz);

% For z and e
[z_gridvals_J, pi_z_J, simoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,simoptions,3,0);
% For semiz
simoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,simoptions,3);

if N_e==0
    if N_z==0
        if N_semiz==0 % none
            n_semizze=0;
            semizze_gridvals_J=[];
        else % semiz
            n_semizze=simoptions.n_semiz;
            semizze_gridvals_J=simoptions.semiz_gridvals_J;
        end
    else % z
        if N_semiz==0
            n_semizze=n_z;
            semizze_gridvals_J=z_gridvals_J;
        else % semiz,z
            n_semizze=[simoptions.n_semiz,n_z];
            semizze_gridvals_J=[repmat(simoptions.semiz_gridvals_J,prod(n_z),1),repelem(z_gridvals_J,prod(simoptions.n_semiz),1)];
        end
    end
else
    if N_z==0
        if N_semiz==0 % e
            n_semizze=simoptions.n_e;
            semizze_gridvals_J=simoptions.e_gridvals_J;
        else % semiz,e
            n_semizze=[simoptions.n_semiz,simoptions.n_e];
            semizze_gridvals_J=[repmat(simoptions.semiz_gridvals_J,prod(simoptions.n_e),1),repelem(simoptions.e_gridvals_J,prod(simoptions.n_semiz),1)];
        end
    else
        if N_semiz==0 % z,e
            n_semizze=[n_z,simoptions.n_e];
            semizze_gridvals_J=[repmat(z_gridvals_J,prod(simoptions.n_e),1),repelem(simoptions.e_gridvals_J,prod(n_z),1)];
        else % semiz,z,e
            n_semizze=[simoptions.n_semiz,n_z,simoptions.n_e];
            semizze_gridvals_J=[repmat(simoptions.semiz_gridvals_J,prod(n_z),1),repelem(z_gridvals_J,prod(simoptions.n_semiz),1)]; % semiz & z
            semizze_gridvals_J=[repmat(semizze_gridvals_J,prod(simoptions.n_e),1),repelem(simoptions.e_gridvals_J,prod([simoptions.n_semiz,n_z]),1)]; % now add e
        end
    end
end

N_semizze=prod(n_semizze);
if N_semizze==0
    l_semizze=0;
else
    l_semizze=length(n_semizze);
end

%
if N_semiz>0
    if ~isfield(simoptions,'l_dsemiz')
        simoptions.l_dsemiz=1; % by default, just one decision variable is used for the semi-exo state
    end
end


%% Build PolicyValues with trailing-j axis
Policy=gpuArray(Policy);
if N_semizze==0
    Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);
else
    Policy=reshape(Policy,[size(Policy,1),N_a,N_semizze,N_j]);
end

PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
if N_semizze==0
    PolicyValuesPermuteJ=permute(PolicyValues,[2,1,3]); % (N_a,l_daprime,N_j)
else
    PolicyValuesPermuteJ=permute(PolicyValues,[2,3,1,4]); % (N_a,N_semizze,l_daprime,N_j)
end


%% Implement new way of handling FnsToEvaluate
l_daprime=size(PolicyValues,1);

if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_semizze)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_semizze+1:end}};
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

%% Convert simoptions.transprobs from names to 0-1 mask
if iscell(simoptions.transprobs)
    temp=simoptions.transprobs;
    simoptions.transprobs=zeros(length(FnsToEvalNames),1);
    for ff=1:length(FnsToEvalNames)
        if any(strcmp(temp,FnsToEvalNames{ff}))
            simoptions.transprobs(ff)=1;
        end
    end
end
if ~has1 && any(simoptions.transprobs(:)>0)
    error('AutoCorrTransProbs_FHorz: simoptions.transprobs are single-period transition probabilities, so they need 1 in simoptions.ntimeperiods')
end

%% Output
CorrTransProbs=struct();

% For the rest, just pretend N_semizze=1 during reshapes
if N_semizze==0
    N_semizze_reshape=1;
else
    N_semizze_reshape=N_semizze;
end

%% Reshape StationaryDist
StationaryDist=gpuArray(reshape(StationaryDist,[N_a*N_semizze_reshape,N_j]));

%% Conditional restrictions: evaluate each restriction on the grid at every age (0/1)
% RestrictionValues(:,jj,rr) is 1 on the age-jj states that satisfy restriction rr.
useCondlRest=0;
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);
    RestrictionValues=false(N_a*N_semizze_reshape,N_j,length(CondlRestnFnNames),'gpuArray'); % logical masks (1 byte per point); every use multiplies them into a double measure
    for rr=1:length(CondlRestnFnNames)
        CondlRestnFn=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});
        temp=getAnonymousFnInputNames(CondlRestnFn);
        if length(temp)>(l_daprime+l_a+l_semizze)
            CondlRestnFnParamNames={temp{l_daprime+l_a+l_semizze+1:end}};
        else
            CondlRestnFnParamNames={};
        end
        if N_semizze==0
            for jj=1:N_j
                CondlRestnFnParamsCell=CreateCellFromParams(Parameters,CondlRestnFnParamNames,jj);
                slice=EvalFnOnAgentDist_Grid(CondlRestnFn,CondlRestnFnParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,n_semizze,a_gridvals,[]);
                RestrictionValues(:,jj,rr)=(slice~=0);
            end
        else
            for jj=1:N_j
                CondlRestnFnParamsCell=CreateCellFromParams(Parameters,CondlRestnFnParamNames,jj);
                slice=EvalFnOnAgentDist_Grid(CondlRestnFn,CondlRestnFnParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_semizze,a_gridvals,semizze_gridvals_J(:,:,jj));
                RestrictionValues(:,jj,rr)=reshape(slice~=0,[N_a*N_semizze_reshape,1]);
            end
        end
        CorrTransProbs.(CondlRestnFnNames{rr}).RestrictedSampleMass=sum(StationaryDist.*RestrictionValues(:,:,rr),1);
        if all(CorrTransProbs.(CondlRestnFnNames{rr}).RestrictedSampleMass==0) && simoptions.warnzerorestrictedmass==2
            warning('One of the conditional restrictions evaluates to a zero mass (at all j)')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
        end
    end
end

%% simoptions.whichcombos: which (fn, start age, restriction) combinations to compute
% [numFnsToEvaluate, N_j, 1+number of conditional restrictions] of zeros/ones: page 1 is the unrestricted outputs, pages 2:end the
% restrictions in the fieldnames order of simoptions.conditionalrestrictions; the second dimension is the start age j. A one at
% (ff,j,page) asks for the outputs that START at age j: the auto-covariances/-correlations (every horizon), the
% pair outputs starting at age j (restricted pages), and the transition from age j to j+1 (TransitionProbs, unrestricted page). The
% age-j Mean and StdDeviation are computed at every age regardless (the horizon outputs starting at j need the means at j+k) and so are
% reported at every age: whichcombos controls what is computed, and what is computed as a byproduct is reported.
% Skipped entries stay NaN (an empty cell for TransitionProbs); a (fn, page) with nothing selected has no output fields at all; a
% function with nothing selected on any page is not evaluated. RestrictedSampleMass is always filled. Default all ones. A
% [numFnsToEvaluate, 1+number of restrictions] or [numFnsToEvaluate, N_j] input is expanded over the missing dimension.
if useCondlRest==1
    nwhichpages=1+length(CondlRestnFnNames);
else
    nwhichpages=1;
end
numFnsToEvaluate=length(FnsToEvalNames);
% With simoptions.ntimeperiods of length L>1 the rows are L blocks of numFnsToEvaluate, one block per M in the order of
% simoptions.ntimeperiods; an input with numFnsToEvaluate rows applies to every M. Internally whichcombos is the M=1 block
% (zeros when 1 is not in ntimeperiods) and whichcombosW(:,:,:,lw) is the block of the window length windowM(lw).
nrowswc=nblocks*numFnsToEvaluate;
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,N_j,nwhichpages); % whichcombos here is [nFns, N_j, 1+nRestr]
    if nW>0
        whichcombosW=ones(numFnsToEvaluate,N_j,nwhichpages,nW);
    end
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    nrowsin=size(whichcombos,1); % numFnsToEvaluate, or nblocks*numFnsToEvaluate
    if ismatrix(whichcombos) && isequal(size(whichcombos),[nrowsin,nwhichpages]) && nwhichpages~=N_j && (nrowsin==numFnsToEvaluate || nrowsin==nrowswc)
        whichcombos=repmat(reshape(whichcombos,[nrowsin,1,nwhichpages]),[1,N_j,1]); % (fn, page): apply to every start age
    elseif ismatrix(whichcombos) && isequal(size(whichcombos),[nrowsin,N_j]) && (nrowsin==numFnsToEvaluate || nrowsin==nrowswc)
        whichcombos=repmat(whichcombos,[1,1,nwhichpages]); % (fn, start age): apply to every page
    end
    if nblocks>1 && nrowsin==numFnsToEvaluate
        whichcombos=repmat(whichcombos,[nblocks,1,1]); % one selection for every M
    end
    if ~isequal(size(whichcombos,1:3),[nrowswc,N_j,nwhichpages])
        if nblocks==1
            error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(N_j),',',num2str(nwhichpages),'] (number of FnsToEvaluate, N_j, 1+number of conditional restrictions)'])
        else
            error(['simoptions.whichcombos must be of size [',num2str(nrowswc),',',num2str(N_j),',',num2str(nwhichpages),'] (length of simoptions.ntimeperiods times the number of FnsToEvaluate, N_j, 1+number of conditional restrictions), or have ',num2str(numFnsToEvaluate),' rows to apply to every M'])
        end
    end
    whichcombos=double(whichcombos);
    if nblocks>1 || ~has1
        whichcombosRows=whichcombos;
        if has1
            bb=find(ntimeperiods==1);
            whichcombos=whichcombosRows((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:);
        else
            whichcombos=zeros(numFnsToEvaluate,N_j,nwhichpages);
        end
        whichcombosW=zeros(numFnsToEvaluate,N_j,nwhichpages,nW);
        for lw=1:nW
            bb=find(ntimeperiods==windowM(lw));
            whichcombosW(:,:,:,lw)=whichcombosRows((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:);
        end
    end
end
if ~has1 && ~isfield(simoptions,'whichcombos')
    whichcombos=zeros(numFnsToEvaluate,N_j,nwhichpages); % the single-period statistics are not wanted
end



%% The computation: never form the per-age transition matrix
% A measure over the age-jj states is pushed to age jj+1 in the two steps of Tan (2020, Economics Letters), as the
% stationary distribution iteration does: the policy step (Gammatranspose, a sparse matrix with one nonzero per state,
% or two with the grid-interpolation weights when gridinterplayer=1, times the number of semiz' a state can reach when
% there is a semi-exogenous state), then the shock step (times pi_z, then the kron with the next age's pi_e). The
% centered (signed) measures below go through the same two steps, since both are linear, so the AutoCovariance/
% AutoCorrelation are those of multiplying by the full transition matrix, at the memory cost of the policy matrix
% instead of the full matrix (about 2*N_z*N_e nonzeros per state).
% The states are ordered (a,semiz,z,e) [whichever are present]. The policy step maps a measure over (a,semiz,z,e) at
% age jj to a measure over (a',semiz',z): the semiz transition is applied with the policy step because it depends on the
% decision taken at the state, exactly as in StationaryDist_FHorz_Iteration_SemiExo_raw and its _e/_nProbs variants.
if simoptions.experienceasset>=1 || simoptions.inheritanceasset==1
    error('EvalFnOnAgentDist_AutoCorrTransProbs_FHorz: experience and inheritance assets are not yet implemented, ask on forum if you need this')
end
N_semizr=max(N_semiz,1); % so that N_a*N_semizr*N_zr*N_er equals N_a*N_semizze_reshape
N_zr=max(N_z,1);
N_er=max(N_e,1);
N_states=N_a*N_semizze_reshape;
%% The policy step: the indexes and weights of a' (and of the semi-exogenous transition) that each age's sparse matrix is built from
Policy=reshape(Policy,[size(Policy,1),N_a,N_semizze_reshape,N_j]);
if l_a==1
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:),1);
elseif l_a==2
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1),1);
elseif l_a==3
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1),1);
elseif l_a==4
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1),1);
else
    error('EvalFnOnAgentDist_AutoCorrTransProbs_FHorz cannot handle length(n_a)>4, contact me if you need this')
end
% Policy_aprime is [N_a,N_semizze_reshape,N_j]
if simoptions.gridinterplayer==1
    % two a' points per state: the lower grid point and the one above it, with the second-layer weights
    L2index=Policy(end-1,:,:,:); % L2 index (end-1 because end is L2flag)
    L2flag=Policy(end,:,:,:);
    L2index(L2flag==1)=1;                        % force all weight to lower grid point
    L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
    probupper=shiftdim((L2index-1)/(simoptions.ngridinterp+1),1); % [N_a,N_semizze_reshape,N_j]: probability of the upper grid point
end
%% The policy step of one age (a sparse gpuArray) is built when the age is reached, used by every push through that
% age, and dropped: the pushes run age by age. Everything in flight at age jj (the measures started at the origin ages
% jj-Kmax+1,...,jj of every function and page of the chunk) is pushed to age jj+1 in one sparse-times-dense product, one
% pagemtimes for the z step and one multiply by the next age's pi_e for the e step. So there are at most N_j-1 policy
% steps and N_j-1 pushes per chunk of functions, whatever the number of functions, restrictions, horizons and start ages,
% and the memory is one age's policy step plus the in-flight measures.
if N_semiz==0
    % Add the z index of each (z,e) column (z varies first) to get the index into (a',z)
    zindex=N_a*repmat(gpuArray(0:1:N_zr-1),1,N_er);
    Policy_aprimez=Policy_aprime+zindex; % [N_a,N_semizze_reshape,N_j], index into (a',z), on the gpu
    if simoptions.gridinterplayer==0
        IIind=gpuArray((1:1:N_states)');
    elseif simoptions.gridinterplayer==1
        IIind=gpuArray(repmat((1:1:N_states)',2,1));
    end
else
    % Semi-exogenous state (as StationaryDist_FHorz_Iteration_SemiExo_raw and its _e/_nProbs variants): from (a,semiz,z,e) to
    % (a',semiz',z) with the transition probabilities pi_semiz_J(semiz,semiz',dsemiz,jj) of the semi-exogenous decision
    % dsemiz taken at the state. Only the N_semizshort largest entries of each row of pi_semiz_J are kept (the sort puts
    % the zeros first), which is all of the nonzeros, so a state has N_semizshort (times two with gridinterplayer) entries.
    l_d1=l_d-simoptions.l_dsemiz; % the last l_dsemiz decision variables are the ones that influence semiz
    N_dsemiz=prod(n_d(l_d1+1:l_d));
    if simoptions.l_dsemiz==1
        Policy_dsemiexo=Policy(l_d1+1,:,:,:);
    elseif simoptions.l_dsemiz==2
        Policy_dsemiexo=Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*(Policy(l_d1+2,:,:,:)-1);
    elseif simoptions.l_dsemiz==3
        Policy_dsemiexo=Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*(Policy(l_d1+2,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*(Policy(l_d1+3,:,:,:)-1);
    elseif simoptions.l_dsemiz==4
        Policy_dsemiexo=Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*(Policy(l_d1+2,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*(Policy(l_d1+3,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*n_d(l_d1+3)*(Policy(l_d1+4,:,:,:)-1);
    end
    Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_states,1,N_j]));
    N_semizshort=max(max(max(sum((simoptions.pi_semiz_J>0),2))));
    [pi_semiz_J_short,idx]=sort(simoptions.pi_semiz_J,2); % puts the zeros on the left
    pi_semiz_J_short=gather(pi_semiz_J_short(:,end-N_semizshort+1:end,:,:)); % [N_semiz,N_semizshort,N_dsemiz,N_j-1]
    idxshort=gather(idx(:,end-N_semizshort+1:end,:,:)); % the semiz' each kept entry belongs to
    semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_zr*N_er,1)+N_semiz*(0:1:N_semizshort-1); % [N_states,N_semizshort]: the semiz of each state, offset to each column of pi_semiz_J_short
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
if N_z>0
    pi_z_J_gpu=gpuArray(pi_z_J); % the pushes are on the gpu
    pi_z_J_cpu=gather(pi_z_J); % the transition probabilities (section iv) are on the cpu
end
if N_e>0
    pi_e_J_gpu=gpuArray(simoptions.pi_e_J);
    pi_e_J_cpu=gather(simoptions.pi_e_J);
end
massJ=sum(StationaryDist,1); % 1 x N_j, the mass at each age (includes the age weights), on the gpu
massJ_cpu=gather(massJ);
hasmassJ=(massJ_cpu>0); % the ages with mass (an age with no mass, e.g. before the entry age of a cohort started using simoptions.jequaloneDistAge, gets NaN in every output that involves it)
if useCondlRest==1
    nRestr=length(CondlRestnFnNames);
else
    nRestr=0;
end

%% The outputs, for every function (NaN where not computed)
MeanV=nan(numFnsToEvaluate,N_j,'gpuArray');
StdDevV=nan(numFnsToEvaluate,N_j,'gpuArray');
AutoCov=cell(1,nhorizons);
AutoCorr=cell(1,nhorizons);
for hh=1:nhorizons
    AutoCov{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),'gpuArray');
    AutoCorr{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),'gpuArray');
end
if nRestr>0
    MeanR=nan(numFnsToEvaluate,N_j,nRestr,'gpuArray');
    StdDevR=nan(numFnsToEvaluate,N_j,nRestr,'gpuArray');
    AutoCovR=cell(1,nhorizons);
    AutoCorrR=cell(1,nhorizons);
    PairMass=cell(1,nhorizons);
    PairMean_j=cell(1,nhorizons);
    PairMean_jplusk=cell(1,nhorizons);
    PairStdDev_j=cell(1,nhorizons);
    PairStdDev_jplusk=cell(1,nhorizons);
    for hh=1:nhorizons
        AutoCovR{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,'gpuArray');
        AutoCorrR{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,'gpuArray');
        PairMass{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,'gpuArray');
        PairMean_j{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,'gpuArray');
        PairMean_jplusk{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,'gpuArray');
        PairStdDev_j{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,'gpuArray');
        PairStdDev_jplusk{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,'gpuArray');
    end
end
% The multi-period (window) outputs, the last dimension is the window length windowM(lw); computed as sums of the M periods
% (divided by M, and the covariances by M^2, when reporting averages)
if nW>0
    WMean=nan(numFnsToEvaluate,N_j,nW,'gpuArray');
    WStdDev=nan(numFnsToEvaluate,N_j,nW,'gpuArray');
    WAutoCov=cell(1,nhorizons);
    WAutoCorr=cell(1,nhorizons);
    for hh=1:nhorizons
        WAutoCov{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nW,'gpuArray');
        WAutoCorr{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nW,'gpuArray');
    end
    if nRestr>0
        WMeanR=nan(numFnsToEvaluate,N_j,nRestr,nW,'gpuArray');
        WStdDevR=nan(numFnsToEvaluate,N_j,nRestr,nW,'gpuArray');
        WAutoCovR=cell(1,nhorizons);
        WAutoCorrR=cell(1,nhorizons);
        WPairMass=cell(1,nhorizons);
        WPairMean_j=cell(1,nhorizons);
        WPairMean_jplusk=cell(1,nhorizons);
        WPairStdDev_j=cell(1,nhorizons);
        WPairStdDev_jplusk=cell(1,nhorizons);
        for hh=1:nhorizons
            WAutoCovR{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,nW,'gpuArray');
            WAutoCorrR{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,nW,'gpuArray');
            WPairMass{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,nW,'gpuArray');
            WPairMean_j{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,nW,'gpuArray');
            WPairMean_jplusk{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,nW,'gpuArray');
            WPairStdDev_j{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,nW,'gpuArray');
            WPairStdDev_jplusk{hh}=nan(numFnsToEvaluate,N_j-horizons(hh),nRestr,nW,'gpuArray');
        end
    end
end
% Transition probabilities between value bins (only when requested for a function, and only when its unrestricted page selects some start age): horizon 1, unrestricted
dotransprobs=zeros(numFnsToEvaluate,1);
for ff=1:numFnsToEvaluate
    if simoptions.transprobs(ff)==1 && any(whichcombos(ff,:,1))
        dotransprobs(ff)=1;
    end
end
if any(dotransprobs)
    if isempty(simoptions.transprobquantiles)
        % Default: unique values per age (size can differ across ages -> cell array)
        P_v_cell=cell(numFnsToEvaluate,1);
        fvals_j_cell=cell(numFnsToEvaluate,1); % unique fn values at jj (labels the rows of TransitionProbs{jj})
        fvals_jplus1_cell=cell(numFnsToEvaluate,1); % unique fn values at jj+1 (labels the columns)
        massbin_j_cell=cell(numFnsToEvaluate,1); % within-age mass of each origin bin (row)
        for ff=1:numFnsToEvaluate
            if dotransprobs(ff)==1
                P_v_cell{ff}=cell(N_j-1,1);
                fvals_j_cell{ff}=cell(N_j-1,1);
                fvals_jplus1_cell{ff}=cell(N_j-1,1);
                massbin_j_cell{ff}=cell(N_j-1,1);
            end
        end
    else
        % Quantile binning -> fixed-size (n_fvals, n_fvals, N_j-1) 3-D array
        n_fvals=simoptions.transprobquantiles;
        P_v_3d=cell(numFnsToEvaluate,1);
        for ff=1:numFnsToEvaluate
            if dotransprobs(ff)==1
                P_v_3d{ff}=nan(n_fvals,n_fvals,N_j-1);
            end
        end
    end
end

%% The computation, in chunks of functions
% A function that no (fn, start age, page) combination of simoptions.whichcombos selects is not evaluated at all. The in-flight
% measures of a function are at most Kmax origin ages times (one unrestricted column and three per restriction); the chunk of
% functions done together is sized so that these take at most about 1 GB of gpu memory (one chunk in all but huge models).
fnwantedmask=any(any(whichcombos,2),3);
if nW>0
    fnwantedmask=fnwantedmask | any(any(any(whichcombosW,2),3),4); % a function wanted only for some window length is evaluated too
end
fnwanted=find(reshape(fnwantedmask,[1,numFnsToEvaluate])); % the functions with some combination wanted, as a row
colsperfn=(1+3*nRestr)*Kmax+nW*(1+nRestr)*(3*(Kmax+Mmax)+6*nhorizons*Mmax); % the window measures: three per (origin, page) and six per (origin, page, horizon) (see below)
maxfnchunk=max(1,floor(1e9/(8*N_states*colsperfn)));
maxgroups=Kmax*(1+nRestr)+nW*(1+nRestr)*((Kmax+Mmax)+nhorizons*Mmax); % the most in-flight groups of columns a function can have (an origin age and a page is a group)
for fc=1:maxfnchunk:length(fnwanted)
    ffs=fnwanted(fc:min(fc+maxfnchunk-1,length(fnwanted))); % the functions of this chunk, as indexes into FnsToEvalNames
    nf=length(ffs);
    % The in-flight measures are the columns of inflight. Group g (an origin age grp_origin(g), the chunk's function
    % grp_fi(g), and a page: grp_rr(g)=0 the unrestricted signed measure distj.*Xc in column grp_col(g), grp_rr(g)=rr the
    % restricted mass, mass.*Xc and mass.*Xc.^2 of restriction rr in columns grp_col(g),grp_col(g)+1,grp_col(g)+2)
    % The window groups (multi-period sums, simoptions.ntimeperiods) are grp_kind(g)=1 and 2 (the single-period groups above
    % are grp_kind(g)=0); grp_w(g) is the number of columns of the group, grp_lw(g) the window length index, grp_hh(g) the
    % horizon index (kind 2), grp_last(g) the last age the group is needed at, grp_sel(g) whether its start age is selected (kind 1).
    % With x~_t the function at age t minus a centering constant c_t (the age-t mean of the page, see below), S1 the running
    % sum of x~ over the first window (ages j0,...,j0+M-1) and S2 over the second (ages j0+K,...,j0+K+M-1):
    %   kind 1, the first window of origin j0 (and a page): the measures m0, m0*S1, m0*S1^2 (three columns), where m0 is the
    %           mass at j0 pushed forward (unrestricted: the age-j0 distribution; restricted: the age-j0 mass satisfying the
    %           restriction). It gives the Mean and StdDeviation of W_j0 (read at age j0+M-1), and when the start age j0 is
    %           selected it starts a kind 2 group at each age j0+K.
    %   kind 2, the pair (j0, j0+K): m0, m0*S1, m0*S1^2, m0*S2, m0*S2^2, m0*S1*S2 (six columns), started at age j0+K from the
    %           kind 1 group (masked by the restriction at age j0+K), read at age j0+K+M-1.
    inflight=zeros(N_states,0,'gpuArray'); % the measures in flight, replaced once per age: [the kept columns, the new origin's columns] then the push
    ncols=0;
    ngrp=0;
    grp_origin=zeros(1,nf*maxgroups);
    grp_fi=zeros(1,nf*maxgroups);
    grp_rr=zeros(1,nf*maxgroups);
    grp_col=zeros(1,nf*maxgroups);
    grp_kind=zeros(1,nf*maxgroups);
    grp_w=zeros(1,nf*maxgroups);
    grp_lw=zeros(1,nf*maxgroups);
    grp_hh=zeros(1,nf*maxgroups);
    grp_last=zeros(1,nf*maxgroups);
    grp_sel=zeros(1,nf*maxgroups);
    % (i) Values of the chunk's functions, one age at a time, evaluated one age ahead (the transition probabilities of
    % age jj need the values at jj+1): Values_next holds age 1 here, and age jj+1 inside the loop
    Values_next=zeros(N_states,nf,'gpuArray');
    for fi=1:nf
        ff=ffs(fi);
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,1);
        if N_semizze==0
            Values_next(:,fi)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff},FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,1),l_daprime,n_a,n_semizze,a_gridvals,[]);
        else
            slice=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff},FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,1),l_daprime,n_a,n_semizze,a_gridvals,semizze_gridvals_J(:,:,1));
            Values_next(:,fi)=reshape(slice,[N_states,1]);
        end
    end
    for jj=1:N_j
        Values_j=Values_next; % N_states x nf, the values of the chunk's functions at age jj
        if jj<N_j
            Values_next=zeros(N_states,nf,'gpuArray');
            for fi=1:nf
                ff=ffs(fi);
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj+1);
                if N_semizze==0
                    Values_next(:,fi)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff},FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,jj+1),l_daprime,n_a,n_semizze,a_gridvals,[]);
                else
                    slice=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff},FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj+1),l_daprime,n_a,n_semizze,a_gridvals,semizze_gridvals_J(:,:,jj+1));
                    Values_next(:,fi)=reshape(slice,[N_states,1]);
                end
            end
        end

        % (ii) Age-jj Mean and StdDev (within-age conditional distribution), unrestricted and over those satisfying each
        % restriction; reported at every age (the horizon outputs starting at j need the means at j+k)
        if hasmassJ(jj)
            distj=StationaryDist(:,jj)./massJ(jj);
            MeanV(ffs,jj)=reshape(sum(distj.*Values_j,1),[nf,1]);
            StdDevV(ffs,jj)=reshape(sqrt(sum(distj.*(Values_j-reshape(MeanV(ffs,jj),[1,nf])).^2,1)),[nf,1]);
        end
        for rr=1:nRestr
            mr=StationaryDist(:,jj).*RestrictionValues(:,jj,rr);
            massr=sum(mr);
            if massr>0
                mr=mr./massr;
                MeanR(ffs,jj,rr)=reshape(sum(mr.*Values_j,1),[nf,1]);
                StdDevR(ffs,jj,rr)=reshape(sqrt(sum(mr.*(Values_j-reshape(MeanR(ffs,jj,rr),[1,nf])).^2,1)),[nf,1]);
            end
        end

        % (iii) Read the in-flight measures whose origin age is a horizon away from age jj. Centered form: more numerically
        % stable than E[XY]-EX*EY when X,Y are nearly constant (the raw-moment form cancels two large numbers into a noisy
        % tiny one). Under a restriction, masking the propagated measures with the restriction at age jj gives the pair
        % population (satisfies the restriction at both ages), and its mass, its means and variances of x_j and x_{j+k},
        % and their covariance follow. [E_pair[(x_j-c)(x_{j+k}-mu_y)] is the pair covariance for ANY constant c, since
        % E_pair[x_{j+k}-mu_y]=0, so centering x_j on the restricted origin-age mean rather than on the (not yet known) pair
        % mean is exact.]
        for g=1:ngrp
            if grp_kind(g)~=0
                continue % a window group (simoptions.ntimeperiods), see (iii-w)
            end
            kk=jj-grp_origin(g);
            hh=find(horizons==kk);
            if isempty(hh)
                continue
            end
            fi=grp_fi(g);
            ff=ffs(fi);
            jj0=grp_origin(g);
            if grp_rr(g)==0
                Yc=Values_j(:,fi)-MeanV(ff,jj);
                AutoCov{hh}(ff,jj0)=inflight(:,grp_col(g))'*Yc;
                denom=StdDevV(ff,jj0)*StdDevV(ff,jj);
                if denom>1e-15
                    AutoCorr{hh}(ff,jj0)=AutoCov{hh}(ff,jj0)/denom;
                end
            else
                rr=grp_rr(g);
                pairs=inflight(:,grp_col(g):grp_col(g)+2).*RestrictionValues(:,jj,rr); % N_states x 3: keep those who satisfy the restriction at age jj too
                pairmass=sum(pairs(:,1));
                PairMass{hh}(ff,jj0,rr)=pairmass;
                if pairmass>0
                    y=Values_j(:,fi);
                    d1=sum(pairs(:,2))/pairmass; % pair mean of x_j, minus MeanR at the origin age
                    muy=sum(pairs(:,1).*y)/pairmass; % pair mean of x_{j+k}
                    varx=sum(pairs(:,3))/pairmass-d1^2;
                    if varx<=64*eps*sum(pairs(:,3))/pairmass
                        varx=0; % below the rounding level of the difference it was computed as: x_j is constant in the pair population (else sqrt would turn the rounding into a std of about 1e-8, and the correlation would not be NaN)
                    end
                    vary=sum(pairs(:,1).*(y-muy).^2)/pairmass;
                    covxy=sum(pairs(:,2).*(y-muy))/pairmass;
                    PairMean_j{hh}(ff,jj0,rr)=MeanR(ff,jj0,rr)+d1;
                    PairMean_jplusk{hh}(ff,jj0,rr)=muy;
                    PairStdDev_j{hh}(ff,jj0,rr)=sqrt(max(varx,0));
                    PairStdDev_jplusk{hh}(ff,jj0,rr)=sqrt(vary);
                    AutoCovR{hh}(ff,jj0,rr)=covxy;
                    denom=PairStdDev_j{hh}(ff,jj0,rr)*PairStdDev_jplusk{hh}(ff,jj0,rr);
                    if denom>1e-15
                        AutoCorrR{hh}(ff,jj0,rr)=covxy/denom;
                    end
                end
            end
        end

        % (iii-w) The window groups (simoptions.ntimeperiods), see the description of kind 1 and kind 2 above. At age jj:
        % first start the kind 2 groups of the pairs (j0, jj) from the kind 1 groups (from their measures before age jj is
        % added), then add age jj to every window group, then read the groups whose windows end at age jj. All the updates
        % of a group at an age use the measures before that age is added: with x~ the age-jj value minus c_jj, a=1 when
        % age jj is in the first window and b=1 when it is in the second,
        %   m0*S1*S2 += b*x~*(m0*S1) + a*x~*(m0*S2) + a*b*x~^2*m0,  m0*S1^2 += a*(2*x~*(m0*S1) + x~^2*m0),
        %   m0*S2^2 += b*(2*x~*(m0*S2) + x~^2*m0),  m0*S1 += a*x~*m0,  m0*S2 += b*x~*m0
        % (an age in both windows, K<M, goes into both sums). The centering constant c_t of a page is its age-t mean (MeanV
        % unrestricted, MeanR restricted; the unrestricted one when the restricted one is NaN, else zero): any constants give
        % the same variances and covariances, and the means add them back. Means and variances are taken over the group's
        % population (sum(m0) is its mass), in the centered form of (iii).
        nspawn=0;
        spawncols=cell(1,0);
        spawn_origin=zeros(1,0);
        spawn_fi=zeros(1,0);
        spawn_rr=zeros(1,0);
        spawn_lw=zeros(1,0);
        spawn_hh=zeros(1,0);
        if nW>0
            for g=1:ngrp
                if grp_kind(g)==0
                    continue
                end
                fi=grp_fi(g);
                ff=ffs(fi);
                jj0=grp_origin(g);
                rr=grp_rr(g);
                lw=grp_lw(g);
                M=windowM(lw);
                c0=grp_col(g);
                % the centering constant of this page at age jj
                if rr==0
                    cjj=MeanV(ff,jj);
                else
                    cjj=MeanR(ff,jj,rr);
                    if isnan(cjj)
                        cjj=MeanV(ff,jj);
                    end
                end
                if isnan(cjj)
                    cjj=0;
                end
                xt=Values_j(:,fi)-cjj;
                a1=double(jj<=jj0+M-1); % age jj is in the first window
                if grp_kind(g)==1
                    % Start the pair (jj0, jj) when the start age jj0 is selected, jj-jj0 is a horizon and the second window fits:
                    % the kind 1 measures (masked by the restriction at age jj), with age jj added (b=1, a=a1)
                    if grp_sel(g)==1
                        hh=find(horizons==jj-jj0);
                        if ~isempty(hh) && jj+M-1<=N_j
                            if rr==0
                                m0=inflight(:,c0);
                                mS1=inflight(:,c0+1);
                                mS1sq=inflight(:,c0+2);
                            else
                                m0=inflight(:,c0).*RestrictionValues(:,jj,rr);
                                mS1=inflight(:,c0+1).*RestrictionValues(:,jj,rr);
                                mS1sq=inflight(:,c0+2).*RestrictionValues(:,jj,rr);
                            end
                            nspawn=nspawn+1;
                            spawncols{nspawn}=[m0, mS1+a1*xt.*m0, mS1sq+a1*(2*xt.*mS1+xt.^2.*m0), xt.*m0, xt.^2.*m0, xt.*mS1+a1*xt.^2.*m0]; % m0, m0*S1, m0*S1^2, m0*S2, m0*S2^2, m0*S1*S2
                            spawn_origin(nspawn)=jj0;
                            spawn_fi(nspawn)=fi;
                            spawn_rr(nspawn)=rr;
                            spawn_lw(nspawn)=lw;
                            spawn_hh(nspawn)=hh;
                        end
                    end
                    % Add age jj to the first window (the origin age was added when the group started)
                    if a1==1 && jj>jj0
                        mS1=inflight(:,c0+1);
                        inflight(:,c0+2)=inflight(:,c0+2)+2*xt.*mS1+xt.^2.*inflight(:,c0);
                        inflight(:,c0+1)=mS1+xt.*inflight(:,c0);
                    end
                    % The Mean and StdDeviation of W_jj0, at the end of the first window
                    if jj==jj0+M-1 && all(hasmassJ(jj0:jj))
                        mass=sum(inflight(:,c0));
                        if mass>0
                            csum=0; % the sum of the centering constants over the window
                            for tt=jj0:jj
                                if rr==0
                                    ctt=MeanV(ff,tt);
                                else
                                    ctt=MeanR(ff,tt,rr);
                                    if isnan(ctt)
                                        ctt=MeanV(ff,tt);
                                    end
                                end
                                if isnan(ctt)
                                    ctt=0;
                                end
                                csum=csum+ctt;
                            end
                            e1=sum(inflight(:,c0+1))/mass;
                            v1=sum(inflight(:,c0+2))/mass-e1^2;
                            if v1<=64*eps*sum(inflight(:,c0+2))/mass
                                v1=0; % below the rounding level of the difference it was computed as (W is constant in the population)
                            end
                            if rr==0
                                WMean(ff,jj0,lw)=csum+e1;
                                WStdDev(ff,jj0,lw)=sqrt(max(v1,0));
                            else
                                WMeanR(ff,jj0,rr,lw)=csum+e1;
                                WStdDevR(ff,jj0,rr,lw)=sqrt(max(v1,0));
                            end
                        end
                    end
                else
                    % kind 2, the pair (jj0, jj0+K): add age jj (its first age, jj0+K, was added when it started)
                    hh=grp_hh(g);
                    K=horizons(hh);
                    if jj>jj0+K
                        m0=inflight(:,c0);
                        mS1=inflight(:,c0+1);
                        mS2=inflight(:,c0+3);
                        inflight(:,c0+5)=inflight(:,c0+5)+xt.*mS1+a1*xt.*mS2+a1*xt.^2.*m0;
                        inflight(:,c0+2)=inflight(:,c0+2)+a1*(2*xt.*mS1+xt.^2.*m0);
                        inflight(:,c0+4)=inflight(:,c0+4)+2*xt.*mS2+xt.^2.*m0;
                        inflight(:,c0+1)=mS1+a1*xt.*m0;
                        inflight(:,c0+3)=mS2+xt.*m0;
                    end
                    % Read the pair at the end of the second window
                    if jj==jj0+K+M-1 && all(hasmassJ(jj0:jj0+M-1)) && all(hasmassJ(jj0+K:jj))
                        mass=sum(inflight(:,c0));
                        if rr>0
                            WPairMass{hh}(ff,jj0,rr,lw)=mass;
                        end
                        if mass>0
                            e1=sum(inflight(:,c0+1))/mass;
                            e2=sum(inflight(:,c0+3))/mass;
                            v1=sum(inflight(:,c0+2))/mass-e1^2;
                            v2=sum(inflight(:,c0+4))/mass-e2^2;
                            if v1<=64*eps*sum(inflight(:,c0+2))/mass
                                v1=0; % below the rounding level of the difference it was computed as (W_j0 is constant in the pair population)
                            end
                            if v2<=64*eps*sum(inflight(:,c0+4))/mass
                                v2=0; % likewise W_{j0+K}
                            end
                            covxy=sum(inflight(:,c0+5))/mass-e1*e2;
                            denom=sqrt(max(v1,0))*sqrt(max(v2,0));
                            if rr==0
                                WAutoCov{hh}(ff,jj0,lw)=covxy;
                                if denom>1e-15
                                    WAutoCorr{hh}(ff,jj0,lw)=covxy/denom;
                                end
                            else
                                csum1=0; % the sums of the centering constants over the two windows
                                for tt=jj0:jj0+M-1
                                    ctt=MeanR(ff,tt,rr);
                                    if isnan(ctt)
                                        ctt=MeanV(ff,tt);
                                    end
                                    if isnan(ctt)
                                        ctt=0;
                                    end
                                    csum1=csum1+ctt;
                                end
                                csum2=0;
                                for tt=jj0+K:jj
                                    ctt=MeanR(ff,tt,rr);
                                    if isnan(ctt)
                                        ctt=MeanV(ff,tt);
                                    end
                                    if isnan(ctt)
                                        ctt=0;
                                    end
                                    csum2=csum2+ctt;
                                end
                                WPairMean_j{hh}(ff,jj0,rr,lw)=csum1+e1;
                                WPairMean_jplusk{hh}(ff,jj0,rr,lw)=csum2+e2;
                                WPairStdDev_j{hh}(ff,jj0,rr,lw)=sqrt(max(v1,0));
                                WPairStdDev_jplusk{hh}(ff,jj0,rr,lw)=sqrt(max(v2,0));
                                WAutoCovR{hh}(ff,jj0,rr,lw)=covxy;
                                if denom>1e-15
                                    WAutoCorrR{hh}(ff,jj0,rr,lw)=covxy/denom;
                                end
                            end
                        end
                    end
                end
            end
        end

        if jj<=N_j-1
            % Drop the measures that have reached their last horizon (the columns are kept in the order of the groups, so
            % the kept columns are selected when the age's push input is assembled below)
            keepg=(jj-grp_origin(1:ngrp)<Kmax);
            if nW>0
                iswindow=(grp_kind(1:ngrp)~=0);
                keepg(iswindow)=(jj+1<=grp_last(iswindow)); % a window group is kept until its last age
            end
            keepcols=false(1,ncols);
            for g=1:ngrp
                if keepg(g)
                    if grp_kind(g)==0
                        keepcols(grp_col(g):grp_col(g)+2*(grp_rr(g)>0))=true;
                    else
                        keepcols(grp_col(g):grp_col(g)+grp_w(g)-1)=true;
                    end
                end
            end
            newcol=cumsum(keepcols); % the column index of each kept column after the drop
            nkeep=sum(keepg);
            grp_col(1:nkeep)=newcol(grp_col(keepg));
            grp_origin(1:nkeep)=grp_origin(keepg);
            grp_fi(1:nkeep)=grp_fi(keepg);
            grp_rr(1:nkeep)=grp_rr(keepg);
            grp_kind(1:nkeep)=grp_kind(keepg);
            grp_w(1:nkeep)=grp_w(keepg);
            grp_lw(1:nkeep)=grp_lw(keepg);
            grp_hh(1:nkeep)=grp_hh(keepg);
            grp_last(1:nkeep)=grp_last(keepg);
            grp_sel(1:nkeep)=grp_sel(keepg);
            ngrp=nkeep;
            nkeepcols=sum(keepcols);

            % Start the measures of origin age jj: for each function of the chunk, the unrestricted signed measure distj.*Xc
            % (when the unrestricted page selects jj) and the three restricted measures of each restriction that selects jj
            % and has mass at jj. An origin age with no mass starts nothing (its outputs stay NaN).
            newcols=zeros(N_states,nf*(1+3*nRestr)+nf*nW*3*(1+nRestr),'gpuArray');
            nnew=0;
            if hasmassJ(jj)
                distj=StationaryDist(:,jj)./massJ(jj);
                for fi=1:nf
                    ff=ffs(fi);
                    if whichcombos(ff,jj,1)==1
                        Xc=Values_j(:,fi)-MeanV(ff,jj);
                        nnew=nnew+1;
                        newcols(:,nnew)=distj.*Xc; % signed measure over the age-jj states
                        ngrp=ngrp+1;
                        grp_origin(ngrp)=jj;
                        grp_fi(ngrp)=fi;
                        grp_rr(ngrp)=0;
                        grp_col(ngrp)=nkeepcols+nnew;
                        grp_kind(ngrp)=0;
                    end
                    for rr=1:nRestr
                        if whichcombos(ff,jj,1+rr)==1
                            mr=StationaryDist(:,jj).*RestrictionValues(:,jj,rr); % restricted mass at age jj (not normalized: includes the age weight)
                            if sum(mr)==0
                                for hh=1:nhorizons
                                    if jj<=N_j-horizons(hh)
                                        PairMass{hh}(ff,jj,rr)=0; % nobody at age jj satisfies the restriction (the age itself has mass)
                                    end
                                end
                            else
                                Xc=Values_j(:,fi)-MeanR(ff,jj,rr);
                                newcols(:,nnew+1)=mr;
                                newcols(:,nnew+2)=mr.*Xc;
                                newcols(:,nnew+3)=mr.*Xc.^2;
                                ngrp=ngrp+1;
                                grp_origin(ngrp)=jj;
                                grp_fi(ngrp)=fi;
                                grp_rr(ngrp)=rr;
                                grp_col(ngrp)=nkeepcols+nnew+1;
                                grp_kind(ngrp)=0;
                                nnew=nnew+3;
                            end
                        end
                    end
                end
                % The window groups of origin age jj (kind 1), for each window length whose window fits and each page that
                % selects some start age of it (the Mean and StdDeviation of W_j are computed at every age, as for M=1)
                for fi=1:nf
                    ff=ffs(fi);
                    for lw=1:nW
                        M=windowM(lw);
                        if jj+M-1>N_j
                            continue % the window starting at jj runs past age N_j
                        end
                        Kfit=horizons(jj+horizons+M-1<=N_j); % the horizons whose second window fits
                        for pp=1:nwhichpages
                            if ~any(whichcombosW(ff,:,pp,lw))
                                continue
                            end
                            rr=pp-1;
                            sel=whichcombosW(ff,jj,pp,lw);
                            if rr==0
                                m0=distj;
                                cjj=MeanV(ff,jj);
                            else
                                m0=StationaryDist(:,jj).*RestrictionValues(:,jj,rr); % restricted mass at age jj (not normalized: includes the age weight)
                                if sum(m0)==0
                                    if sel==1
                                        for hh=1:nhorizons
                                            if jj+horizons(hh)+M-1<=N_j
                                                WPairMass{hh}(ff,jj,rr,lw)=0; % nobody at age jj satisfies the restriction (the age itself has mass)
                                            end
                                        end
                                    end
                                    continue
                                end
                                cjj=MeanR(ff,jj,rr);
                            end
                            if isnan(cjj)
                                cjj=0;
                            end
                            xt=Values_j(:,fi)-cjj;
                            newcols(:,nnew+1)=m0;
                            newcols(:,nnew+2)=m0.*xt;
                            newcols(:,nnew+3)=m0.*xt.^2;
                            ngrp=ngrp+1;
                            grp_origin(ngrp)=jj;
                            grp_fi(ngrp)=fi;
                            grp_rr(ngrp)=rr;
                            grp_col(ngrp)=nkeepcols+nnew+1;
                            grp_kind(ngrp)=1;
                            grp_w(ngrp)=3;
                            grp_lw(ngrp)=lw;
                            grp_hh(ngrp)=0;
                            grp_sel(ngrp)=sel;
                            if sel==1 && ~isempty(Kfit)
                                grp_last(ngrp)=max(jj+M-1,jj+max(Kfit)); % needed until it has started its last pair
                            else
                                grp_last(ngrp)=jj+M-1;
                            end
                            nnew=nnew+3;
                        end
                    end
                end
            end
            % The kind 2 groups started at age jj in (iii-w) go after the new origin's columns
            for ss=1:nspawn
                ngrp=ngrp+1;
                grp_origin(ngrp)=spawn_origin(ss);
                grp_fi(ngrp)=spawn_fi(ss);
                grp_rr(ngrp)=spawn_rr(ss);
                grp_col(ngrp)=nkeepcols+nnew+6*(ss-1)+1;
                grp_kind(ngrp)=2;
                grp_w(ngrp)=6;
                grp_lw(ngrp)=spawn_lw(ss);
                grp_hh(ngrp)=spawn_hh(ss);
                grp_sel(ngrp)=0;
                grp_last(ngrp)=spawn_origin(ss)+horizons(spawn_hh(ss))+windowM(spawn_lw(ss))-1;
            end
            % The push input of age jj: the kept columns and the new origin's columns (one copy; nothing when nothing changed)
            if nspawn>0
                inflight=[inflight(:,keepcols),newcols(:,1:nnew),spawncols{:}];
                ncols=nkeepcols+nnew+6*nspawn;
            elseif nkeepcols<ncols || nnew>0
                inflight=[inflight(:,keepcols),newcols(:,1:nnew)];
                ncols=nkeepcols+nnew;
            end

            % The transition probabilities of age jj (when requested) also use the age-jj policy step
            anytransprobs=0;
            if hasmassJ(jj)
                for fi=1:nf
                    if dotransprobs(ffs(fi))==1 && whichcombos(ffs(fi),jj,1)==1
                        anytransprobs=1;
                    end
                end
            end

            if ncols>0 || anytransprobs==1
                % The policy step of age jj: maps a measure over the age-jj states (a,semiz,z,e) to a measure over (a',semiz',z) [z kept, e summed out]
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
                        Policy_aprimesemizz_jj=repelem(Policy_aprime_cpu(:,1,jj),1,N_semizshort)+N_a*(idxshort(semizindex_short_jj)-1)+zprimeoffset; % [N_states,N_semizshort]: index into (a',semiz',z)
                        Gammatranspose_jj=sparse(gpuArray(Policy_aprimesemizz_jj),gpuArray(II2),gpuArray(pi_semiz_J_short(semizindex_short_jj)),N_a*N_semiz*N_zr,N_states); % sparse gpuArray
                    elseif simoptions.gridinterplayer==1
                        Policy_aprimesemizz_jj=repelem(Policy_aprime2_cpu(:,:,jj),1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short_jj)-1),1,2)+zprimeoffset; % [N_states,2*N_semizshort]: the two grid points, each with every semiz'
                        PolicyProbs_jj=repelem(PolicyProbs_cpu(:,:,jj),1,N_semizshort).*repmat(pi_semiz_J_short(semizindex_short_jj),1,2);
                        Gammatranspose_jj=sparse(gpuArray(Policy_aprimesemizz_jj),gpuArray(II2),gpuArray(PolicyProbs_jj),N_a*N_semiz*N_zr,N_states); % sparse gpuArray; sparse() accumulates repeated indexes (the two grid points coincide only at the top of the grid)
                    end
                end

                % The push from age jj to age jj+1 of everything in flight: the Tan step, all the measures as columns
                if ncols>0
                    temp=Gammatranspose_jj*inflight; % policy step: now over (a',semiz',z), (N_a*N_semizr*N_zr) x ncols
                    if N_z>0
                        temp=reshape(pagemtimes(reshape(temp,[N_a*N_semizr,N_zr,ncols]),pi_z_J_gpu(:,:,jj)),[N_a*N_semizr*N_zr,ncols]); % z step, every column at once
                    end
                    if N_e>0
                        temp=reshape(reshape(temp,[N_a*N_semizr*N_zr,1,ncols]).*reshape(pi_e_J_gpu(:,jj+1),[1,N_er]),[N_states,ncols]); % e step: the e realized at age jj+1 is the slowest index of the states (the layout of kron(pi_e,.))
                    end
                    inflight=temp; % now measures over the age-(jj+1) states (no copy)
                end

                % (iv) Transition probabilities between value bins at age jj (unrestricted, horizon 1), on the cpu as the
                % toolkit function does them: each origin bin's mass is pushed one age forward (the same Tan step, with the
                % origin bins as columns, in blocks of 64) and binned by the function's value at age jj+1; row b of
                % TransitionProbs{jj} is the destination distribution of origin bin b, so rows sum to one.
                if anytransprobs==1
                    Gammatranspose_jj_cpu=gather(Gammatranspose_jj);
                    distj_cpu=gather(StationaryDist(:,jj)./massJ(jj));
                    for fi=1:nf
                        ff=ffs(fi);
                        if dotransprobs(ff)==1 && whichcombos(ff,jj,1)==1
                            if isempty(simoptions.transprobquantiles)
                                Values_cpu_j=gather(Values_j(:,fi));
                                Values_cpu_jplus1=gather(Values_next(:,fi));
                                [fvals_j_cell{ff}{jj},~,idx_j]=unique(Values_cpu_j);
                                [fvals_jplus1_cell{ff}{jj},~,idx_jp]=unique(Values_cpu_jplus1);
                                n_fvals_j=max(idx_j);
                                n_fvals_jp=max(idx_jp);
                                massPerBin_j=accumarray(idx_j,distj_cpu,[n_fvals_j,1]);
                                S_jp=sparse(1:N_states,idx_jp,1,N_states,n_fvals_jp); % destination bin indicator
                                P_v=zeros(n_fvals_j,n_fvals_jp);
                                for b1=1:64:n_fvals_j
                                    b2=min(b1+63,n_fvals_j);
                                    inblock=(idx_j>=b1 & idx_j<=b2);
                                    M=sparse(find(inblock),idx_j(inblock)-b1+1,distj_cpu(inblock),N_states,b2-b1+1); % the origin bins' masses, one column each
                                    temp=full(Gammatranspose_jj_cpu*M);
                                    if N_z>0
                                        for cc=1:size(temp,2)
                                            temp(:,cc)=reshape(reshape(temp(:,cc),[N_a*N_semizr,N_zr])*pi_z_J_cpu(:,:,jj),[N_a*N_semizr*N_zr,1]);
                                        end
                                    end
                                    if N_e>0
                                        temp=kron(pi_e_J_cpu(:,jj+1),temp);
                                    end
                                    massPerBin_block=massPerBin_j(b1:b2);
                                    massPerBin_block(massPerBin_block==0)=1; % an origin bin with no mass gets a row of zeros (nothing to normalise); any positive mass, however small, normalises its row to one
                                    P_v(b1:b2,:)=(S_jp'*temp)'./massPerBin_block;
                                end
                                P_v_cell{ff}{jj}=P_v;
                                massbin_j_cell{ff}{jj}=massPerBin_j;
                            elseif hasmassJ(jj+1)
                                % Quantile bins of the within-age distributions: bin q holds the values up to the first sorted value whose
                                % cumulative mass exceeds q/n_fvals (the same definition as the InfHorz command), at age jj and at age jj+1
                                distjplus1=StationaryDist(:,jj+1)./massJ(jj+1);
                                [SortedValues,sortindex]=sort(Values_j(:,fi));
                                CumSortedDist=cumsum(distj(sortindex));
                                quantilecutoffs=nan(n_fvals-1,1,'gpuArray');
                                for qq=1:n_fvals-1
                                    [~,qqind]=max(CumSortedDist>qq*1/n_fvals);
                                    quantilecutoffs(qq)=SortedValues(qqind);
                                end
                                idx_j=ones(N_states,1,'gpuArray');
                                for qq=2:n_fvals
                                    idx_j(Values_j(:,fi)>quantilecutoffs(qq-1))=qq;
                                end
                                idx_j=gather(idx_j);
                                [SortedValues,sortindex]=sort(Values_next(:,fi));
                                CumSortedDist=cumsum(distjplus1(sortindex));
                                quantilecutoffs=nan(n_fvals-1,1,'gpuArray');
                                for qq=1:n_fvals-1
                                    [~,qqind]=max(CumSortedDist>qq*1/n_fvals);
                                    quantilecutoffs(qq)=SortedValues(qqind);
                                end
                                idx_jp=ones(N_states,1,'gpuArray');
                                for qq=2:n_fvals
                                    idx_jp(Values_next(:,fi)>quantilecutoffs(qq-1))=qq;
                                end
                                idx_jp=gather(idx_jp);
                                massPerBin_j=accumarray(idx_j,distj_cpu,[n_fvals,1]);
                                S_jp=sparse(1:N_states,idx_jp,1,N_states,n_fvals);
                                M=sparse(1:N_states,idx_j,distj_cpu,N_states,n_fvals); % the origin bins' masses, one column each
                                temp=full(Gammatranspose_jj_cpu*M);
                                if N_z>0
                                    for cc=1:size(temp,2)
                                        temp(:,cc)=reshape(reshape(temp(:,cc),[N_a*N_semizr,N_zr])*pi_z_J_cpu(:,:,jj),[N_a*N_semizr*N_zr,1]);
                                    end
                                end
                                if N_e>0
                                    temp=kron(pi_e_J_cpu(:,jj+1),temp);
                                end
                                massPerBin_safe=massPerBin_j;
                                massPerBin_safe(massPerBin_j==0)=1; % an origin bin with no mass gets a row of zeros (nothing to normalise)
                                P_v_3d{ff}(:,:,jj)=(S_jp'*temp)'./massPerBin_safe;
                            end
                        end
                    end
                end
            end
        end
    end
end

%% Report
for ff=1:numFnsToEvaluate
    fn=FnsToEvalNames{ff};
    if ~any(whichcombos(ff,:,:),'all') % no combination of this function is wanted
        continue
    end
    if any(whichcombos(ff,:,1)) % the unrestricted outputs of this function are wanted (the horizon outputs at the selected start ages; the rest stay NaN)
        CorrTransProbs.(fn).Mean=MeanV(ff,:); % reported at every age: computed anyway (the horizon outputs starting at j need the means at j+k)
        CorrTransProbs.(fn).StdDeviation=StdDevV(ff,:);
        for hh=1:nhorizons
            CorrTransProbs.(fn).(['AutoCovariance',horizonstr{hh}])=AutoCov{hh}(ff,:);
            CorrTransProbs.(fn).(['AutoCorrelation',horizonstr{hh}])=AutoCorr{hh}(ff,:);
        end
    end
    if useCondlRest==1
        for rr=1:nRestr
            if ~any(whichcombos(ff,:,1+rr)) % this restriction is not wanted for this function
                continue
            end
            CorrTransProbs.(CondlRestnFnNames{rr}).(fn).Mean=MeanR(ff,:,rr); % reported at every age (computed anyway)
            CorrTransProbs.(CondlRestnFnNames{rr}).(fn).StdDeviation=StdDevR(ff,:,rr);
            for hh=1:nhorizons
                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['AutoCovariance',horizonstr{hh}])=AutoCovR{hh}(ff,:,rr);
                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['AutoCorrelation',horizonstr{hh}])=AutoCorrR{hh}(ff,:,rr);
                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMass',horizonstr{hh}])=PairMass{hh}(ff,:,rr);
                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMean_j',horizonstr{hh}])=PairMean_j{hh}(ff,:,rr);
                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMean_jplusk',horizonstr{hh}])=PairMean_jplusk{hh}(ff,:,rr);
                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairStdDeviation_j',horizonstr{hh}])=PairStdDev_j{hh}(ff,:,rr);
                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairStdDeviation_jplusk',horizonstr{hh}])=PairStdDev_jplusk{hh}(ff,:,rr);
            end
        end
    end
    if dotransprobs(ff)==1
        if isempty(simoptions.transprobquantiles)
            CorrTransProbs.(fn).TransitionProbs=P_v_cell{ff};
            CorrTransProbs.(fn).TransitionValues_j=fvals_j_cell{ff};
            CorrTransProbs.(fn).TransitionValues_jplus1=fvals_jplus1_cell{ff};
            CorrTransProbs.(fn).TransitionMass_j=massbin_j_cell{ff};
        else
            CorrTransProbs.(fn).TransitionProbs=P_v_3d{ff};
        end
    end
end

% The multi-period (window) outputs, under [fnname,'_',M,'periodAverage'] (or 'periodSum'); computed as sums, so for averages
% the means and std devs are divided by M and the covariances by M^2 (the correlations and pair masses are the same)
for lw=1:nW
    M=windowM(lw);
    if strcmp(simoptions.ntimeperiodstype,'average')
        scale1=1/M;
    else
        scale1=1;
    end
    for ff=1:numFnsToEvaluate
        if ~any(whichcombosW(ff,:,:,lw),'all') % no combination of this function and window length is wanted
            continue
        end
        wname=[FnsToEvalNames{ff},'_',num2str(M),windowstr];
        if any(whichcombosW(ff,:,1,lw)) % the unrestricted outputs are wanted
            CorrTransProbs.(wname).Mean=scale1*WMean(ff,:,lw);
            CorrTransProbs.(wname).StdDeviation=scale1*WStdDev(ff,:,lw);
            for hh=1:nhorizons
                CorrTransProbs.(wname).(['AutoCovariance',horizonstr{hh}])=scale1^2*WAutoCov{hh}(ff,:,lw);
                CorrTransProbs.(wname).(['AutoCorrelation',horizonstr{hh}])=WAutoCorr{hh}(ff,:,lw);
            end
        end
        for rr=1:nRestr
            if ~any(whichcombosW(ff,:,1+rr,lw)) % this restriction is not wanted for this function and window length
                continue
            end
            CorrTransProbs.(CondlRestnFnNames{rr}).(wname).Mean=scale1*WMeanR(ff,:,rr,lw);
            CorrTransProbs.(CondlRestnFnNames{rr}).(wname).StdDeviation=scale1*WStdDevR(ff,:,rr,lw);
            for hh=1:nhorizons
                CorrTransProbs.(CondlRestnFnNames{rr}).(wname).(['AutoCovariance',horizonstr{hh}])=scale1^2*WAutoCovR{hh}(ff,:,rr,lw);
                CorrTransProbs.(CondlRestnFnNames{rr}).(wname).(['AutoCorrelation',horizonstr{hh}])=WAutoCorrR{hh}(ff,:,rr,lw);
                CorrTransProbs.(CondlRestnFnNames{rr}).(wname).(['PairMass',horizonstr{hh}])=WPairMass{hh}(ff,:,rr,lw);
                CorrTransProbs.(CondlRestnFnNames{rr}).(wname).(['PairMean_j',horizonstr{hh}])=scale1*WPairMean_j{hh}(ff,:,rr,lw);
                CorrTransProbs.(CondlRestnFnNames{rr}).(wname).(['PairMean_jplusk',horizonstr{hh}])=scale1*WPairMean_jplusk{hh}(ff,:,rr,lw);
                CorrTransProbs.(CondlRestnFnNames{rr}).(wname).(['PairStdDeviation_j',horizonstr{hh}])=scale1*WPairStdDev_j{hh}(ff,:,rr,lw);
                CorrTransProbs.(CondlRestnFnNames{rr}).(wname).(['PairStdDeviation_jplusk',horizonstr{hh}])=scale1*WPairStdDev_jplusk{hh}(ff,:,rr,lw);
            end
        end
    end
end

CorrTransProbs.Notes='Mean and StdDeviation are 1xN_j. AutoCovariance and AutoCorrelation are 1x(N_j-1), with index jj corresponding to the transition from age jj to age jj+1; AutoCovariance_kK and AutoCorrelation_kK (for K in simoptions.timehorizons) are 1x(N_j-K), index jj is the pair of ages jj and jj+K. Under a conditional restriction the auto-covariances are over the pairs that satisfy the restriction at both ages, centered on the pair means (PairMean_j, PairMean_jplusk), and PairMass is the population mass of those pairs. TransitionProbs (when requested) is a cell {N_j-1} of (possibly varying-size) matrices, or a 3-D (nquantiles, nquantiles, N_j-1) array when simoptions.transprobquantiles is set. TransitionValues_j and TransitionValues_jplus1 (cells {N_j-1}) give the unique function values labelling the rows and columns of TransitionProbs{jj} respectively, and TransitionMass_j (cell {N_j-1}) gives the within-age mass of each origin bin/row (none of these are provided when using transprobquantiles, where bins are quantiles rather than values).';
if nW>0
    CorrTransProbs.Notes=[CorrTransProbs.Notes,' The fields fnname_MperiodAverage (or fnname_MperiodSum), for each M>1 in simoptions.ntimeperiods, are the same statistics of the average (sum) W_j of the function over the M ages j,...,j+M-1: Mean and StdDeviation of W_j, and the auto-covariances/-correlations between W_j and W_{j+K} (the windows starting at ages j and j+K, K in model periods; the windows overlap when K<M); under a conditional restriction the pairs satisfy the restriction at the two window start ages j and j+K. The last M-1 entries are NaN (the window runs past age N_j).'];
end

end
