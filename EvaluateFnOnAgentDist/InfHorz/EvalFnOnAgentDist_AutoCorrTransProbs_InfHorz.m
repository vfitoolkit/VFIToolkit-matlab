function CorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz(StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, simoptions)
% Returns stats on (auto) correlation and transition probabilities
% You must input the names for the FnsToEvaluate that you want the transition probabilities for (by default it won't do any)
% Done as simoptions.transprobs
%
% simoptions optional inputs
%   simoptions.timehorizons=[2,5]: also the K-period auto-covariance/-correlation (and transition probabilities, if
%                                   requested), reported under CorrTransProbs.(fnname).tperiodsK (the 1-period is always computed)
%   simoptions.transprobquantiles=5: transition probabilities between quantile bins instead of between unique values
%   simoptions.n_e, e_grid, pi_e: an iid e shock (kept apart from the markov z throughout)
%
% Outputs:
% Mean (as it has to be calculated anyway as an intermediate step to correlation)
% StdDeviation (as it has to be calculated anyway as an intermediate step to correlation)
% AutoCovariance
% AutoCorrelation
% TransitionProbs (optional)
%
% simoptions.conditionalrestrictions (structure of functions, same form as FnsToEvaluate, returning 0/1): everything also
% conditional on each restriction, under CorrTransProbs.(restrictionname). The Mean and StdDeviation are over those that
% satisfy the restriction, and the auto-covariance at horizon k is over those that satisfy it both now and k periods later
% (the 'pairs'), as in EvalFnOnAgentDist_AutoCorrTransProbs_FHorz:
%   .RestrictedSampleMass   the mass that satisfies the restriction
%   .(fnname).Mean, .StdDeviation   over those that satisfy the restriction
%   .(fnname).AutoCovariance, .AutoCorrelation   over the pairs, centered on the pair means (the covariance/correlation of the pair population)
%   .(fnname).PairMass   the mass of the pairs (an agent counts if it satisfies the restriction now and will k periods later)
%   .(fnname).PairMean_t, .PairMean_tplusk, .PairStdDeviation_t, .PairStdDeviation_tplusk   the means/std devs of x_t and x_{t+k}
%                    in the pair population
%   and the same under .(fnname).tperiodsK for each K in simoptions.timehorizons.
% A restriction of zero mass gives a warning, NaN (and a PairMass of zero). TransitionProbs are not computed under conditional
% restrictions.
%
% simoptions.whichcombos ([numFnsToEvaluate, 1+number of conditional restrictions] of zeros/ones) selects which (fn, restriction)
% combinations are computed; see the whichcombos section below.

%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.transprobs=zeros(length(fieldnames(FnsToEvaluate)),1); % which variables to calculate transition probabilities for (input as cell of FnsToEval names)
    simoptions.timehorizons=[]; % calculate the correlation and trans prob at these time horizons in addition to the one-period ones (can be a vector, e.g. [5,10] calculates the 1, 5 and 10 period correlations and transition probabilities; the 1 period is always calculated regardless)
    simoptions.transprobquantiles=[]; % e.g., =5 will calculate the transition probabilities for quintiles
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
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'transprobs')
        simoptions.transprobs=zeros(length(fieldnames(FnsToEvaluate)),1); % which variables to calculate transition probabilities for (input as cell of FnsToEval names)
    end
    if ~isfield(simoptions,'timehorizons')
        simoptions.timehorizons=[]; % calculate the correlation and trans prob at these time horizons in addition to the one-period ones (can be a vector, e.g. [5,10] calculates the 1, 5 and 10 period correlations and transition probabilities; the 1 period is always calculated regardless)
    end
    if ~isfield(simoptions,'transprobquantiles')
        simoptions.transprobquantiles=[]; % e.g., =5 will calculate the transition probabilities for quintiles
    end
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
end

N_d=prod(n_d);
N_a=prod(n_a);

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

a_gridvals=CreateGridvals(n_a,a_grid,1);
if prod(simoptions.n_semiz)>0
    error('Have not yet implemented semiz variables for InfHorz AutoCorrTransProbs, ask on forum if you need this')
end

%% Exogenous shocks: the markov z and the iid e are kept apart throughout (as in the FHorz command)
% The functions are evaluated on the combined (z,e) grid built here (z varies first, then e), while the push below
% applies pi_z to the z index and pi_e to the e index separately. [This command used to fold e into z with
% CreateGridvals_FnsToEvaluate_InfHorz and then recover the split from quantities captured before the fold.]
N_z=prod(n_z);
N_e=prod(simoptions.n_e);
[z_gridvals, pi_z, simoptions]=ExogShockSetup_InfHorz(n_z,z_grid,pi_z,Parameters,simoptions,3,0); % also gives simoptions.e_gridvals and simoptions.pi_e
if N_e==0
    n_ze=n_z;
    ze_gridvals=z_gridvals;
else
    if N_z==0
        n_ze=simoptions.n_e;
        ze_gridvals=simoptions.e_gridvals;
    else
        n_ze=[n_z,simoptions.n_e];
        ze_gridvals=[repmat(z_gridvals,N_e,1),repelem(simoptions.e_gridvals,N_z,1)];
    end
end
N_ze=prod(n_ze);
if N_ze==0
    l_ze=0;
else
    l_ze=length(n_ze);
end
N_ze_reshape=max(N_ze,1); % so that N_a*N_ze_reshape is the number of states whether or not there are shocks
N_states=N_a*N_ze_reshape;

CorrTransProbs=struct();

%%
StationaryDist=reshape(StationaryDist,[N_states,1]);

% Make sure things are on the gpu (they should already be)
StationaryDist=gpuArray(StationaryDist);
Policy=gpuArray(Policy);

% Switch to PolicyValues, and permute
PolicyValues=PolicyInd2Val_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid,simoptions); % PolicyInd2Val_InfHorz handles simoptions.n_e itself
if N_ze==0
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a]),[2,1]); %[N_a,l_d+l_a]
else
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_ze]),[2,3,1]); %[N_a,N_ze,l_d+l_a]
end
l_daprime=size(PolicyValues,1);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluate_copy=FnsToEvaluate; % keep a copy in case needed for conditional restrictions
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_ze)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_ze+1:end}}; % the first inputs will always be (d,aprime,a,z,e)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

%% Conditional restrictions: evaluate each restriction on the grid (0/1)
% RestrictionValues(:,rr) is 1 on the states that satisfy restriction rr.
useCondlRest=0;
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);
    RestrictionValues=false(N_states,length(CondlRestnFnNames)); % logical masks, on the cpu (where the measures are propagated)
    for rr=1:length(CondlRestnFnNames)
        CondlRestnFn=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});
        temp=getAnonymousFnInputNames(CondlRestnFn);
        if length(temp)>(l_daprime+l_a+l_ze)
            CondlRestnFnParamNames={temp{l_daprime+l_a+l_ze+1:end}}; % the first inputs will always be (d,aprime,a,z,e)
        else
            CondlRestnFnParamNames={};
        end
        CondlRestnFnParamsCell=CreateCellFromParams(Parameters,CondlRestnFnParamNames);
        RestrictionValues(:,rr)=reshape(gather(logical(EvalFnOnAgentDist_Grid(CondlRestnFn,CondlRestnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_ze,a_gridvals,ze_gridvals))),[N_states,1]);
        CorrTransProbs.(CondlRestnFnNames{rr}).RestrictedSampleMass=sum(StationaryDist.*RestrictionValues(:,rr));
        if CorrTransProbs.(CondlRestnFnNames{rr}).RestrictedSampleMass==0
            warning('One of the conditional restrictions evaluates to a zero mass')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
        end
    end
end

%% Convert simoptions.transprobs from names to 0-1
if iscell(simoptions.transprobs)
    temp=simoptions.transprobs;
    simoptions.transprobs=zeros(length(FnsToEvalNames),1);
    for ff=1:length(FnsToEvalNames)
        if any(strcmp(temp,FnsToEvalNames{ff}))
            simoptions.transprobs(ff)=1;
        end
    end
end

%% simoptions.whichcombos: which (fn, restriction) combinations to compute
% [numFnsToEvaluate, 1+number of conditional restrictions] of zeros/ones ([numFnsToEvaluate,1] without restrictions): page 1 is the
% unrestricted outputs (Mean, StdDeviation, the auto-covariances/-correlations at every horizon, TransitionProbs if requested), pages
% 2:end the restrictions in the fieldnames order of simoptions.conditionalrestrictions (all their outputs, every horizon). Ones are
% computed, zeros skipped: a skipped (fn, page) has no output fields at all, and a function with nothing selected on any page is not
% evaluated. RestrictedSampleMass is always filled. Default all ones. A vector of length numFnsToEvaluate with restrictions is applied
% to every page. (As EvalFnOnAgentDist_AutoCorrTransProbs_FHorz, without its start-age dimension.)
if useCondlRest==1
    nwhichpages=1+length(CondlRestnFnNames);
else
    nwhichpages=1;
end
numFnsToEvaluate=length(FnsToEvalNames);
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,nwhichpages);
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if isvector(whichcombos) && numel(whichcombos)==numFnsToEvaluate
        whichcombos=repmat(whichcombos(:),[1,nwhichpages]); % one entry per function: apply to every page
    end
    if ~isequal(size(whichcombos),[numFnsToEvaluate,nwhichpages])
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(nwhichpages),'] (number of FnsToEvaluate, 1+number of conditional restrictions)'])
    end
    whichcombos=double(whichcombos);
end

%% The computation: never form the transition matrix
% Cov(x_t,x_{t+k}) is the centered signed measure (dist.*(x-mean)) pushed k periods forward and then integrated against
% x. The push is the two-step of Tan (2020, Economics Letters) that the stationary distribution iteration uses: the
% policy step (Gammatranspose, a sparse N_a*N_z by N_a*N_z*N_e matrix with one nonzero per state, or two with the grid
% interpolation weights), then the shock step (times pi_z, then the kron with pi_e). The transition probabilities between
% value bins push each origin bin's mass the same way.
N_zr=max(N_z,1);
N_er=max(N_e,1);
Policy=reshape(Policy,[size(Policy,1),N_a,N_ze_reshape]);
if l_a==1
    Policy_aprime=shiftdim(Policy(l_d+1,:,:),1);
elseif l_a==2
    Policy_aprime=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1),1);
elseif l_a==3
    Policy_aprime=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1),1);
elseif l_a==4
    Policy_aprime=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1),1);
else
    error('EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
end
% Policy_aprime is [N_a,N_ze_reshape]; add the markov-z index of each (z,e) column (z varies first) to get the index into (a',z)
zindex=N_a*repmat(gpuArray(0:1:N_zr-1),1,N_er);
if simoptions.gridinterplayer==0
    Policy_aprimez=gather(Policy_aprime+zindex);
    Gammatranspose=sparse(reshape(Policy_aprimez,[],1),(1:1:N_states)',ones(N_states,1),N_a*N_zr,N_states);
elseif simoptions.gridinterplayer==1
    % two a' points per state: the lower grid point and the one above it, with the second-layer weights
    Policy_aprimez=gather(cat(3,Policy_aprime,Policy_aprime+1)+zindex); % [N_a,N_ze_reshape,2]
    L2index=Policy(end-1,:,:); % L2 index (end-1 because end is L2flag)
    L2flag=Policy(end,:,:);
    L2index(L2flag==1)=1;                        % force all weight to lower grid point
    L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
    probupper=shiftdim((L2index-1)/(simoptions.ngridinterp+1),1); % [N_a,N_ze_reshape]: probability of the upper grid point
    PolicyProbs=gather(cat(3,1-probupper,probupper)); % [N_a,N_ze_reshape,2]
    Gammatranspose=sparse(reshape(Policy_aprimez,[],1),repmat((1:1:N_states)',2,1),reshape(PolicyProbs,[],1),N_a*N_zr,N_states);
end
if N_z>0
    pi_z_cpu=gather(pi_z);
end
if N_e>0
    pi_e_cpu=gather(simoptions.pi_e(:));
end
if ~isempty(simoptions.timehorizons)
    maxhorizon=max([1,simoptions.timehorizons(:)']);
else
    maxhorizon=1;
end

%%
for ff=1:length(FnsToEvalNames)
    if ~any(whichcombos(ff,:)) % no combination of this function is wanted
        continue
    end
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
    Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_ze,a_gridvals,ze_gridvals);
    Values=reshape(Values,[N_states,1]);
    Values_cpu=gather(Values);
    if whichcombos(ff,1)==1 % the unrestricted outputs of this function are wanted
        %% Mean and standard deviation
        meanV=sum(StationaryDist.*Values);
        stddevV=sqrt(sum(StationaryDist.*(Values-meanV).^2));
        CorrTransProbs.(FnsToEvalNames{ff}).Mean=meanV;
        CorrTransProbs.(FnsToEvalNames{ff}).StdDeviation=stddevV;
        %% Auto-covariance and auto-correlation at horizons 1 and simoptions.timehorizons
        % Correlation(x,y)=Cov(x,y)/(stddev(x)*stddev(y)), with Cov(x_t,x_{t+k}) from the centered measure pushed k periods
        Xc=Values_cpu-gather(meanV);
        propagated=gather(StationaryDist).*Xc; % N_states x 1 signed measure, on the cpu
        for kk=1:maxhorizon
            % Tan step: one period forward
            temp=Gammatranspose*propagated; % policy step: now over (a',z)
            if N_z>0
                temp=reshape(reshape(temp,[N_a,N_zr])*pi_z_cpu,[N_a*N_zr,1]); % z step
            end
            if N_e>0
                temp=kron(pi_e_cpu,temp); % e step
            end
            propagated=temp;
            if kk==1 || any(simoptions.timehorizons==kk)
                Covar=propagated'*Xc;
                Corr=Covar/(stddevV*stddevV);
                if kk==1
                    CorrTransProbs.(FnsToEvalNames{ff}).AutoCovariance=Covar;
                    CorrTransProbs.(FnsToEvalNames{ff}).AutoCorrelation=Corr;
                else
                    CorrTransProbs.(FnsToEvalNames{ff}).(['tperiods',num2str(kk)]).AutoCovariance=Covar;
                    CorrTransProbs.(FnsToEvalNames{ff}).(['tperiods',num2str(kk)]).AutoCorrelation=Corr;
                end
            end
        end
    end

    %% Conditional restrictions: the Mean/StdDeviation over those satisfying the restriction, and the auto-covariance over the
    % pairs that satisfy it now and k periods later. Three signed measures are propagated together (three columns of one Tan
    % step): the restricted mass m, m.*Xc and m.*Xc.^2 (Xc centered on the restricted mean). Masking the propagated measures with
    % the restriction gives the pair population, and its mass, its means and variances of x_t and x_{t+k}, and their covariance
    % follow. [E_pair[(x_t-c)(x_{t+k}-mu_y)] is the pair covariance for ANY constant c, since E_pair[x_{t+k}-mu_y]=0, so centering
    % x_t on the restricted mean rather than on the (not yet known) pair mean is exact.]
    if useCondlRest==1
        dist_cpu=gather(StationaryDist);
        for rr=1:length(CondlRestnFnNames)
            if whichcombos(ff,1+rr)==0 % this restriction is not wanted for this function
                continue
            end
            rname=CondlRestnFnNames{rr};
            mr=dist_cpu.*RestrictionValues(:,rr); % restricted mass (not normalized)
            massr=sum(mr);
            MeanR=NaN; StdDevR=NaN;
            if massr>0
                MeanR=sum(mr.*Values_cpu)/massr;
                StdDevR=sqrt(sum(mr.*(Values_cpu-MeanR).^2)/massr);
            end
            CorrTransProbs.(rname).(FnsToEvalNames{ff}).Mean=MeanR;
            CorrTransProbs.(rname).(FnsToEvalNames{ff}).StdDeviation=StdDevR;
            if massr>0
                XcR=Values_cpu-MeanR;
                propagated=[mr, mr.*XcR, mr.*XcR.^2]; % N_states x 3, on the cpu
            end
            for kk=1:maxhorizon
                if massr>0
                    % Tan step: one period forward, the three measures as three columns
                    temp=Gammatranspose*propagated; % policy step: now over (a',z)
                    if N_z>0
                        for cc=1:3
                            temp(:,cc)=reshape(reshape(temp(:,cc),[N_a,N_zr])*pi_z_cpu,[N_a*N_zr,1]); % z step
                        end
                    end
                    if N_e>0
                        temp=kron(pi_e_cpu,temp); % e step
                    end
                    propagated=temp;
                end
                if ~(kk==1 || any(simoptions.timehorizons==kk))
                    continue % a horizon that is not reported
                end
                PairMass=0; PairMean_t=NaN; PairMean_tplusk=NaN; PairStdDev_t=NaN; PairStdDev_tplusk=NaN; AutoCovR=NaN; AutoCorrR=NaN;
                if massr>0
                    pairs=propagated.*RestrictionValues(:,rr); % keep those who satisfy the restriction k periods later too
                    PairMass=sum(pairs(:,1));
                    if PairMass>0
                        d1=sum(pairs(:,2))/PairMass; % pair mean of x_t, minus MeanR
                        muy=sum(pairs(:,1).*Values_cpu)/PairMass; % pair mean of x_{t+k}
                        varx=sum(pairs(:,3))/PairMass-d1^2;
                        vary=sum(pairs(:,1).*(Values_cpu-muy).^2)/PairMass;
                        AutoCovR=sum(pairs(:,2).*(Values_cpu-muy))/PairMass;
                        PairMean_t=MeanR+d1;
                        PairMean_tplusk=muy;
                        PairStdDev_t=sqrt(max(varx,0));
                        PairStdDev_tplusk=sqrt(vary);
                        if PairStdDev_t*PairStdDev_tplusk>1e-15
                            AutoCorrR=AutoCovR/(PairStdDev_t*PairStdDev_tplusk);
                        end
                    end
                end
                if kk==1
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).AutoCovariance=AutoCovR;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).AutoCorrelation=AutoCorrR;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).PairMass=PairMass;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).PairMean_t=PairMean_t;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).PairMean_tplusk=PairMean_tplusk;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).PairStdDeviation_t=PairStdDev_t;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).PairStdDeviation_tplusk=PairStdDev_tplusk;
                else
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).(['tperiods',num2str(kk)]).AutoCovariance=AutoCovR;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).(['tperiods',num2str(kk)]).AutoCorrelation=AutoCorrR;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).(['tperiods',num2str(kk)]).PairMass=PairMass;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).(['tperiods',num2str(kk)]).PairMean_t=PairMean_t;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).(['tperiods',num2str(kk)]).PairMean_tplusk=PairMean_tplusk;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).(['tperiods',num2str(kk)]).PairStdDeviation_t=PairStdDev_t;
                    CorrTransProbs.(rname).(FnsToEvalNames{ff}).(['tperiods',num2str(kk)]).PairStdDeviation_tplusk=PairStdDev_tplusk;
                end
            end
        end
    end
    %% Transition probabilities between value bins (horizon 1 and simoptions.timehorizons): each origin bin's mass is
    % pushed forward (origin bins as columns, in blocks of 64) and binned by the function's value; row b is the
    % destination distribution of origin bin b, so rows sum to one
    if simoptions.transprobs(ff)==1 && whichcombos(ff,1)==1
        if isempty(simoptions.transprobquantiles)
            [~,~,indexes]=unique(Values_cpu);
            n_fvals=max(indexes); % number of unique values of the FnsToEvaluate{ff}
        else
            % Convert from values into quantile indexes
            [SortedValues,sortindex]=sort(Values);
            SortedDist=StationaryDist(sortindex);
            CumSortedDist=cumsum(SortedDist);
            quantilecutoffs=nan(simoptions.transprobquantiles-1,1,'gpuArray');
            for qq=1:simoptions.transprobquantiles-1
                [~,qqind]=max(CumSortedDist>qq*1/simoptions.transprobquantiles);
                quantilecutoffs(qq)=SortedValues(qqind);
            end
            quantilesindicator=zeros(size(Values),'gpuArray');
            quantilesindicator(Values<=quantilecutoffs(1))=1;
            for qq=2:simoptions.transprobquantiles-1
                quantilesindicator(logical((Values>quantilecutoffs(qq-1)).*(Values<=quantilecutoffs(qq))))=qq;
            end
            quantilesindicator(Values>quantilecutoffs(end))=simoptions.transprobquantiles;
            indexes=gather(quantilesindicator);
            n_fvals=simoptions.transprobquantiles;
        end
        dist_cpu=gather(StationaryDist);
        massPerBin=accumarray(indexes,dist_cpu,[n_fvals,1]);
        S_dest=sparse(1:N_states,indexes,1,N_states,n_fvals); % destination bin indicator (the same bins at every horizon)
        P_v=zeros(n_fvals,n_fvals,maxhorizon); % the third index is the horizon (only 1 and simoptions.timehorizons are filled)
        for b1=1:64:n_fvals
            b2=min(b1+63,n_fvals);
            inblock=(indexes>=b1 & indexes<=b2);
            temp=sparse(find(inblock),indexes(inblock)-b1+1,dist_cpu(inblock),N_states,b2-b1+1); % the origin bins' masses, one column each
            for kk=1:maxhorizon
                temp=full(Gammatranspose*temp);
                if N_z>0
                    for cc=1:size(temp,2)
                        temp(:,cc)=reshape(reshape(temp(:,cc),[N_a,N_zr])*pi_z_cpu,[N_a*N_zr,1]);
                    end
                end
                if N_e>0
                    temp=kron(pi_e_cpu,temp);
                end
                if kk==1 || any(simoptions.timehorizons==kk)
                    P_v(b1:b2,:,kk)=(S_dest'*temp)'./massPerBin(b1:b2);
                end
            end
        end
        CorrTransProbs.(FnsToEvalNames{ff}).TransitionProbs=P_v(:,:,1);
        for kk=2:maxhorizon
            if any(simoptions.timehorizons==kk)
                CorrTransProbs.(FnsToEvalNames{ff}).(['tperiods',num2str(kk)]).TransitionProbs=P_v(:,:,kk);
            end
        end
    end
end

end
