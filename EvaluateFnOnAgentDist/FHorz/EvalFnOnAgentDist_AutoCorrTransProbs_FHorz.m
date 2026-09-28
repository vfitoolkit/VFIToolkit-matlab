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
%                     restriction at j and will satisfy it at j+1; includes the age-j weight)
%   .(fnname).PairMean_j, .PairMean_jplusk, .PairStdDeviation_j, .PairStdDeviation_jplusk   1 x (N_j-1),
%                     the means/std devs of x_j and of x_{j+1} in the pair population
%   and the same with suffix _kK for each K in simoptions.timehorizons (all 1 x (N_j-K))
%   TransitionProbs are not computed under conditional restrictions.
%
% Note: an age with zero mass (e.g. before the entry age of a cohort started using
% simoptions.jequaloneDistAge) gets NaN in every output that involves it.
%
% Not yet implemented (will error):
%   simoptions.agegroupings non-default  -- error

%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.transprobs=zeros(length(fieldnames(FnsToEvaluate)),1);
    simoptions.timehorizons=[]; % multi-period horizons (horizon 1 is always calculated)
    simoptions.transprobquantiles=[];
    simoptions.agegroupings=1:1:N_j; % age bins -- not yet implemented (default = each age separately)
    simoptions.lowmemory=0; % =1 use less memory, but slower
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    % Other endogenous states
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
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0; % =1 use less memory, but slower
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
    RestrictionValues=zeros(N_a*N_semizze_reshape,N_j,length(CondlRestnFnNames),'gpuArray');
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
        if all(CorrTransProbs.(CondlRestnFnNames{rr}).RestrictedSampleMass==0)
            warning('One of the conditional restrictions evaluates to a zero mass (at all j)')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
        end
    end
end



if simoptions.lowmemory==0
    %% Per-age transition matrices P_jj (jj=1..N_j-1)
    if N_e==0
        if N_z==0
            if N_semiz==0 % none
                P_cell=CreatePTransitionMatrix_J(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],[],[],Parameters,simoptions);
            else % semiz
                P_cell=CreatePTransitionMatrix_J(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J,[],[],Parameters,simoptions);
            end
        else % z
            if N_semiz==0
                P_cell=CreatePTransitionMatrix_J(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],pi_z_J,[],Parameters,simoptions);
            else % semiz,z
                P_cell=CreatePTransitionMatrix_J(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J,pi_z_J,[],Parameters,simoptions);
            end
        end
    else
        if N_z==0
            if N_semiz==0 % e
                P_cell=CreatePTransitionMatrix_J(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],[],simoptions.pi_e_J,Parameters,simoptions);
            else % semiz,e
                P_cell=CreatePTransitionMatrix_J(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J,[],simoptions.pi_e_J,Parameters,simoptions);
            end
        else
            if N_semiz==0 % z,e
                P_cell=CreatePTransitionMatrix_J(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],pi_z_J,simoptions.pi_e_J,Parameters,simoptions);
            else % semiz,z,e
                P_cell=CreatePTransitionMatrix_J(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters,simoptions);
            end
        end
    end


    %% Per-function computation
    for ff=1:length(FnsToEvalNames)
        fn=FnsToEvalNames{ff};

        % (i) Per-age Values, shape (N_a*N_semizze, N_j)
        Values=nan(N_a*N_semizze_reshape,N_j,'gpuArray');
        if N_semizze==0
            for jj=1:N_j
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                slice=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff},FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,n_semizze,a_gridvals,[]);
                Values(:,jj)=slice;
            end
        else
            for jj=1:N_j
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                slice=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff},FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_semizze,a_gridvals,semizze_gridvals_J(:,:,jj));
                Values(:,jj)=reshape(slice,[N_a*N_semizze_reshape,1]);
            end
        end

        % (ii) Per-age Mean and StdDev (within-age conditional distribution)
        MeanV=nan(1,N_j,'gpuArray');
        StdDevV=nan(1,N_j,'gpuArray');
        for jj=1:N_j
            massj=sum(StationaryDist(:,jj));
            if massj>0
                distj=StationaryDist(:,jj)./massj;
                MeanV(jj)=sum(distj.*Values(:,jj));
                StdDevV(jj)=sqrt(sum(distj.*(Values(:,jj)-MeanV(jj)).^2));
            end
        end

        % (iii) Per-age AutoCov and AutoCorr at each horizon (transition j -> j+k)
        % Use the centered form for AutoCov: more numerically stable than E[XY]-EX*EY
        % when X,Y are nearly constant (the raw-moment form cancels two large numbers
        % into a noisy tiny one).
        % The signed measure (distj.*Xc) over the age-jj states is propagated forward one
        % age at a time (a row vector times the sparse transition matrix, never a product
        % of transition matrices), and at each horizon that was requested the covariance
        % with the centered age-(jj+k) values is read off.
        AutoCov=cell(1,nhorizons);
        AutoCorr=cell(1,nhorizons);
        for hh=1:nhorizons
            AutoCov{hh}=nan(1,N_j-horizons(hh),'gpuArray');
            AutoCorr{hh}=nan(1,N_j-horizons(hh),'gpuArray');
        end
        for jj=1:N_j-1
            massj=sum(StationaryDist(:,jj));
            if massj>0
                distj=StationaryDist(:,jj)./massj;
                Xc=Values(:,jj)-MeanV(jj);
                propagated=(distj.*Xc)';
                for kk=1:min(Kmax,N_j-jj)
                    propagated=propagated*P_cell{jj+kk-1}; % now a signed measure over the age-(jj+kk) states
                    hh=find(horizons==kk);
                    if ~isempty(hh)
                        Yc=Values(:,jj+kk)-MeanV(jj+kk);
                        AutoCov{hh}(jj)=full(propagated*Yc);
                        denom=StdDevV(jj)*StdDevV(jj+kk);
                        % Threshold guards against "0/0" for variables that are constant within
                        % an age (e.g. an agej-only fn): StdDev there is floating-point noise
                        % (~1e-15), so denom can be ~1e-30 and the ratio explodes. 1e-15 is far
                        % below any real-world variance and well above numerical noise.
                        if denom>1e-15
                            AutoCorr{hh}(jj)=AutoCov{hh}(jj)/denom;
                        end
                    end
                end
            end
        end

        CorrTransProbs.(fn).Mean=MeanV;
        CorrTransProbs.(fn).StdDeviation=StdDevV;
        for hh=1:nhorizons
            CorrTransProbs.(fn).(['AutoCovariance',horizonstr{hh}])=AutoCov{hh};
            CorrTransProbs.(fn).(['AutoCorrelation',horizonstr{hh}])=AutoCorr{hh};
        end

        %% (iii-b) Conditional restrictions: means/std devs at each age over those satisfying the
        % restriction, and the auto-covariance over the pairs that satisfy it at both ages.
        % Three signed measures over the age-jj states are propagated together: the restricted
        % mass m, m.*Xc and m.*Xc.^2 (Xc centered on the restricted age-jj mean). Masking the
        % propagated measures with the restriction at age jj+k gives the pair population, and
        % its mass, its means and variances of x_j and x_{j+k}, and their covariance follow.
        % [E_pair[(x_j-c)(x_{j+k}-mu_y)] is the pair covariance for ANY constant c, since
        % E_pair[x_{j+k}-mu_y]=0, so centering x_j on the restricted age-jj mean rather than
        % on the (not yet known) pair mean is exact.]
        if useCondlRest==1
            for rr=1:length(CondlRestnFnNames)
                MeanR=nan(1,N_j,'gpuArray');
                StdDevR=nan(1,N_j,'gpuArray');
                for jj=1:N_j
                    mr=StationaryDist(:,jj).*RestrictionValues(:,jj,rr);
                    massr=sum(mr);
                    if massr>0
                        mr=mr./massr;
                        MeanR(jj)=sum(mr.*Values(:,jj));
                        StdDevR(jj)=sqrt(sum(mr.*(Values(:,jj)-MeanR(jj)).^2));
                    end
                end
                AutoCovR=cell(1,nhorizons);
                AutoCorrR=cell(1,nhorizons);
                PairMass=cell(1,nhorizons);
                PairMean_j=cell(1,nhorizons);
                PairMean_jplusk=cell(1,nhorizons);
                PairStdDev_j=cell(1,nhorizons);
                PairStdDev_jplusk=cell(1,nhorizons);
                for hh=1:nhorizons
                    AutoCovR{hh}=nan(1,N_j-horizons(hh),'gpuArray');
                    AutoCorrR{hh}=nan(1,N_j-horizons(hh),'gpuArray');
                    PairMass{hh}=nan(1,N_j-horizons(hh),'gpuArray');
                    PairMean_j{hh}=nan(1,N_j-horizons(hh),'gpuArray');
                    PairMean_jplusk{hh}=nan(1,N_j-horizons(hh),'gpuArray');
                    PairStdDev_j{hh}=nan(1,N_j-horizons(hh),'gpuArray');
                    PairStdDev_jplusk{hh}=nan(1,N_j-horizons(hh),'gpuArray');
                end
                for jj=1:N_j-1
                    mr=StationaryDist(:,jj).*RestrictionValues(:,jj,rr); % restricted mass at age jj (not normalized: includes the age weight)
                    if sum(mr)>0
                        Xc=Values(:,jj)-MeanR(jj);
                        propagated=[mr, mr.*Xc, mr.*Xc.^2]'; % 3 x N_states
                        for kk=1:min(Kmax,N_j-jj)
                            propagated=propagated*P_cell{jj+kk-1}; % now over the age-(jj+kk) states
                            hh=find(horizons==kk);
                            if ~isempty(hh)
                                pairs=full(propagated).*RestrictionValues(:,jj+kk,rr)'; % keep those who satisfy the restriction at age jj+kk too
                                pairmass=sum(pairs(1,:));
                                PairMass{hh}(jj)=pairmass;
                                if pairmass>0
                                    y=Values(:,jj+kk)';
                                    d1=sum(pairs(2,:))/pairmass; % pair mean of x_j, minus MeanR(jj)
                                    muy=sum(pairs(1,:).*y)/pairmass; % pair mean of x_{j+k}
                                    varx=sum(pairs(3,:))/pairmass-d1^2;
                                    vary=sum(pairs(1,:).*(y-muy).^2)/pairmass;
                                    covxy=sum(pairs(2,:).*(y-muy))/pairmass;
                                    PairMean_j{hh}(jj)=MeanR(jj)+d1;
                                    PairMean_jplusk{hh}(jj)=muy;
                                    PairStdDev_j{hh}(jj)=sqrt(max(varx,0));
                                    PairStdDev_jplusk{hh}(jj)=sqrt(vary);
                                    AutoCovR{hh}(jj)=covxy;
                                    denom=PairStdDev_j{hh}(jj)*PairStdDev_jplusk{hh}(jj);
                                    if denom>1e-15
                                        AutoCorrR{hh}(jj)=covxy/denom;
                                    end
                                end
                            end
                        end
                    end
                end
                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).Mean=MeanR;
                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).StdDeviation=StdDevR;
                for hh=1:nhorizons
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['AutoCovariance',horizonstr{hh}])=AutoCovR{hh};
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['AutoCorrelation',horizonstr{hh}])=AutoCorrR{hh};
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMass',horizonstr{hh}])=PairMass{hh};
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMean_j',horizonstr{hh}])=PairMean_j{hh};
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMean_jplusk',horizonstr{hh}])=PairMean_jplusk{hh};
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairStdDeviation_j',horizonstr{hh}])=PairStdDev_j{hh};
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairStdDeviation_jplusk',horizonstr{hh}])=PairStdDev_jplusk{hh};
                end
            end
        end

        %% (iv) Transition probabilities (only when requested for this fn)
        if simoptions.transprobs(ff)==1
            if isempty(simoptions.transprobquantiles)
                % Default: unique values per age (size can differ across ages -> cell array)
                P_v_cell=cell(N_j-1,1);
                fvals_j_cell=cell(N_j-1,1); % unique fn values at jj (labels the rows of TransitionProbs{jj})
                fvals_jplus1_cell=cell(N_j-1,1); % unique fn values at jj+1 (labels the columns)
                massbin_j_cell=cell(N_j-1,1); % within-age mass of each origin bin (row)
                for jj=1:N_j-1
                    massj=sum(StationaryDist(:,jj));
                    if massj==0
                        continue
                    end
                    [fvals_j_cell{jj},~,idx_j]=unique(gather(Values(:,jj)));
                    [fvals_jplus1_cell{jj},~,idx_jp]=unique(gather(Values(:,jj+1)));
                    n_fvals_j=max(idx_j);
                    n_fvals_jp=max(idx_jp);

                    distj_cpu=gather(StationaryDist(:,jj)./massj);
                    P_jj=P_cell{jj};

                    % Bin columns of P_jj by idx_jp (right-multiply by sparse indicator)
                    % and mass-weighted-aggregate rows by idx_j (left-multiply)
                    S_jp=sparse(1:N_a*N_semizze_reshape,idx_jp,1,N_a*N_semizze_reshape,n_fvals_jp);
                    S_j=sparse(1:N_a*N_semizze_reshape,idx_j,1,N_a*N_semizze_reshape,n_fvals_j);
                    massPerBin_j=accumarray(idx_j,distj_cpu,[n_fvals_j,1]);
                    P_v=full(S_j'*(distj_cpu.*(P_jj*S_jp)))./max(massPerBin_j,eps);
                    P_v_cell{jj}=P_v;
                    massbin_j_cell{jj}=massPerBin_j;
                end
                CorrTransProbs.(fn).TransitionProbs=P_v_cell;
                CorrTransProbs.(fn).TransitionValues_j=fvals_j_cell;
                CorrTransProbs.(fn).TransitionValues_jplus1=fvals_jplus1_cell;
                CorrTransProbs.(fn).TransitionMass_j=massbin_j_cell;
            else
                % Quantile binning -> fixed-size (n_fvals, n_fvals, N_j-1) 3-D array
                n_fvals=simoptions.transprobquantiles;
                P_v_3d=nan(n_fvals,n_fvals,N_j-1);
                for jj=1:N_j-1
                    massj=sum(StationaryDist(:,jj));
                    massjplus1=sum(StationaryDist(:,jj+1));
                    if massj==0 || massjplus1==0
                        continue
                    end
                    distj=StationaryDist(:,jj)./massj;
                    distjplus1=StationaryDist(:,jj+1)./massjplus1;

                    idx_j=gather(LocalQuantileIndex(Values(:,jj),distj,n_fvals));
                    idx_jp=gather(LocalQuantileIndex(Values(:,jj+1),distjplus1,n_fvals));

                    distj_cpu=gather(distj);
                    P_jj=P_cell{jj};
                    % Bin columns of P_jj by idx_jp (right-multiply by sparse indicator)
                    % and mass-weighted-aggregate rows by idx_j (left-multiply)
                    S_jp=sparse(1:N_a*N_semizze_reshape,idx_jp,1,N_a*N_semizze_reshape,n_fvals);
                    S_j=sparse(1:N_a*N_semizze_reshape,idx_j,1,N_a*N_semizze_reshape,n_fvals);
                    massPerBin_j=accumarray(idx_j,distj_cpu,[n_fvals,1]);
                    P_v_3d(:,:,jj)=full(S_j'*(distj_cpu.*(P_jj*S_jp)))./max(massPerBin_j,eps);
                end
                CorrTransProbs.(fn).TransitionProbs=P_v_3d;
            end
        end
    end

elseif simoptions.lowmemory==1
    % Setup some output shapes
    % (the per-age stats must live in CorrTransProbs from the start: local nan-vectors
    % recreated inside the (jj,ff) loops would be wiped and overwritten every iteration,
    % leaving only the final age in the output)
    for ff=1:length(FnsToEvalNames)
        fn=FnsToEvalNames{ff};
        CorrTransProbs.(fn).Mean=nan(1,N_j,'gpuArray');
        CorrTransProbs.(fn).StdDeviation=nan(1,N_j,'gpuArray');
        for hh=1:nhorizons
            CorrTransProbs.(fn).(['AutoCovariance',horizonstr{hh}])=nan(1,N_j-horizons(hh),'gpuArray');
            CorrTransProbs.(fn).(['AutoCorrelation',horizonstr{hh}])=nan(1,N_j-horizons(hh),'gpuArray');
        end
        if useCondlRest==1
            for rr=1:length(CondlRestnFnNames)
                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).Mean=nan(1,N_j,'gpuArray');
                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).StdDeviation=nan(1,N_j,'gpuArray');
                for hh=1:nhorizons
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['AutoCovariance',horizonstr{hh}])=nan(1,N_j-horizons(hh),'gpuArray');
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['AutoCorrelation',horizonstr{hh}])=nan(1,N_j-horizons(hh),'gpuArray');
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMass',horizonstr{hh}])=nan(1,N_j-horizons(hh),'gpuArray');
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMean_j',horizonstr{hh}])=nan(1,N_j-horizons(hh),'gpuArray');
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMean_jplusk',horizonstr{hh}])=nan(1,N_j-horizons(hh),'gpuArray');
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairStdDeviation_j',horizonstr{hh}])=nan(1,N_j-horizons(hh),'gpuArray');
                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairStdDeviation_jplusk',horizonstr{hh}])=nan(1,N_j-horizons(hh),'gpuArray');
                end
            end
        end
        if simoptions.transprobs(ff)==1
            if isempty(simoptions.transprobquantiles)
                % Default: unique values per age (size can differ across ages -> cell array)
                CorrTransProbs.(fn).TransitionProbs=cell(N_j-1,1);
                CorrTransProbs.(fn).TransitionValues_j=cell(N_j-1,1); % unique fn values at jj (labels the rows of TransitionProbs{jj})
                CorrTransProbs.(fn).TransitionValues_jplus1=cell(N_j-1,1); % unique fn values at jj+1 (labels the columns)
                CorrTransProbs.(fn).TransitionMass_j=cell(N_j-1,1); % within-age mass of each origin bin (row)
            else
                % Quantile binning -> fixed-size (n_fvals, n_fvals, N_j-1) 3-D array
                n_fvals=simoptions.transprobquantiles;
                CorrTransProbs.(fn).TransitionProbs=nan(n_fvals,n_fvals,N_j-1);
            end
        end
    end

    nFns=length(FnsToEvalNames);
    if useCondlRest==1
        nRest=length(CondlRestnFnNames);
    else
        nRest=0;
    end

    % Age-jj and age-(jj+1) Values per function (so at jj>=2 each function recycles its own
    % age-jj values; a single shared variable would hand it whichever function was
    % evaluated last in the ff loop)
    Values_now=zeros(N_a*N_semizze_reshape,nFns,'gpuArray');
    Values_last=zeros(N_a*N_semizze_reshape,nFns,'gpuArray');

    % Signed measures in flight, being propagated from their start age j0 towards horizon Kmax.
    % Only the current P_jj is ever held, so a measure started at j0 is multiplied by P_j0, then
    % P_j0+1, ... as the loop over jj reaches them; the buffers hold one row per start age,
    % circularly (row mod(j0-1,Kmax)+1), and at iteration jj the row of j0=jj-Kmax has just been
    % read off at horizon Kmax and is overwritten by the new start age jj (a start age with zero
    % mass leaves its row zero and inactive, so its outputs stay NaN).
    % Unrestricted: one row (distj.*Xc)' per start age; restricted: three rows (m, m.*Xc, m.*Xc.^2)'.
    % All the rows in flight (every function, every restriction) are propagated by ONE product
    % with P_jj per age, as that product is the expensive step (P_jj is large and sparse).
    Ubuf=zeros(Kmax,N_a*N_semizze_reshape,nFns,'gpuArray');
    Uactive=false(Kmax,nFns);
    if useCondlRest==1
        Rbuf=zeros(3,N_a*N_semizze_reshape,Kmax,nFns,nRest,'gpuArray');
        Ractive=false(Kmax,nFns,nRest);
    end

    % Loop over jj=1:N_j to minimize having to store the large P transition matrices
    for jj=1:N_j-1

        if jj==1
            massj=sum(StationaryDist(:,jj));
            distj=StationaryDist(:,jj)./massj;
        else
            massj=massjplus1;
            distj=distjplus1;
        end
        massjplus1=sum(StationaryDist(:,jj+1));
        distjplus1=StationaryDist(:,jj+1)./massjplus1;

        if N_e==0
            if N_z==0
                if N_semiz==0 % none
                    P_jj=CreatePTransitionMatrix(Policy(:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],[],[],Parameters,simoptions);
                else % semiz
                    P_jj=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J(:,:,:,jj),[],[],Parameters,simoptions);
                end
            else % z
                if N_semiz==0
                    P_jj=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],pi_z_J(:,:,jj),[],Parameters,simoptions);
                else % semiz,z
                    P_jj=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J(:,:,:,jj),pi_z_J(:,:,jj),[],Parameters,simoptions);
                end
            end
        else
            if N_z==0
                if N_semiz==0 % e
                    P_jj=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],[],simoptions.pi_e_J(:,jj+1),Parameters,simoptions);
                else % semiz,e
                    P_jj=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J(:,:,:,jj),[],simoptions.pi_e_J(:,jj+1),Parameters,simoptions);
                end
            else
                if N_semiz==0 % z,e
                    P_jj=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],pi_z_J(:,:,jj),simoptions.pi_e_J(:,jj+1),Parameters,simoptions);
                else % semiz,z,e
                    P_jj=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J(:,:,:,jj),pi_z_J(:,:,jj),simoptions.pi_e_J(:,jj+1),Parameters,simoptions);
                end
            end
        end

        rowjj=mod(jj-1,Kmax)+1; % the buffer row of the measures started at age jj (it held j0=jj-Kmax until it was read off at horizon Kmax in the previous iteration)

        %% Per-function: values, means, and start the measures of age jj
        for ff=1:nFns
            fn=FnsToEvalNames{ff};

            if jj==1
                % (i) Per-age Values, shape (N_a*N_semizze, 1)
                if N_semizze==0
                    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                    slice=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff},FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,n_semizze,a_gridvals,[]);
                    Values_jj=slice;
                else
                    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                    slice=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff},FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_semizze,a_gridvals,semizze_gridvals_J(:,:,jj));
                    Values_jj=reshape(slice,[N_a*N_semizze_reshape,1]);
                end
            else
                Values_jj=Values_last(:,ff);
            end
            Values_now(:,ff)=Values_jj;

            % (i) Per-age Values, shape (N_a*N_semizze, 1)
            if N_semizze==0
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj+1);
                slice=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff},FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,jj+1),l_daprime,n_a,n_semizze,a_gridvals,[]);
                Values_jjplus1=slice;
            else
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj+1);
                slice=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff},FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj+1),l_daprime,n_a,n_semizze,a_gridvals,semizze_gridvals_J(:,:,jj+1));
                Values_jjplus1=reshape(slice,[N_a*N_semizze_reshape,1]);
            end
            Values_last(:,ff)=Values_jjplus1;

            % (ii) Per-age Mean and StdDev (within-age conditional distribution)
            if jj==1
                if massj>0
                    CorrTransProbs.(fn).Mean(jj)=sum(distj.*Values_jj);
                    CorrTransProbs.(fn).StdDeviation(jj)=sqrt(sum(distj.*(Values_jj-CorrTransProbs.(fn).Mean(jj)).^2));
                end
            end

            if massjplus1>0
                CorrTransProbs.(fn).Mean(jj+1)=sum(distjplus1.*Values_jjplus1);
                CorrTransProbs.(fn).StdDeviation(jj+1)=sqrt(sum(distjplus1.*(Values_jjplus1-CorrTransProbs.(fn).Mean(jj+1)).^2));
            end

            % (iii-a) Start the measure of age jj (the centered form, see the lowmemory=0 branch)
            if massj>0
                Ubuf(rowjj,:,ff)=(distj.*(Values_jj-CorrTransProbs.(fn).Mean(jj)))';
                Uactive(rowjj,ff)=true;
            else
                Ubuf(rowjj,:,ff)=0;
                Uactive(rowjj,ff)=false;
            end

            % (iii-b) Conditional restrictions: restricted mean and std dev at age jj (at jj=1) and at age jj+1, and start the three measures of age jj
            if useCondlRest==1
                for rr=1:nRest
                    if jj==1
                        mr=StationaryDist(:,jj).*RestrictionValues(:,jj,rr);
                        massr=sum(mr);
                        if massr>0
                            mr=mr./massr;
                            CorrTransProbs.(CondlRestnFnNames{rr}).(fn).Mean(jj)=sum(mr.*Values_jj);
                            CorrTransProbs.(CondlRestnFnNames{rr}).(fn).StdDeviation(jj)=sqrt(sum(mr.*(Values_jj-CorrTransProbs.(CondlRestnFnNames{rr}).(fn).Mean(jj)).^2));
                        end
                    end
                    mr=StationaryDist(:,jj+1).*RestrictionValues(:,jj+1,rr);
                    massr=sum(mr);
                    if massr>0
                        mr=mr./massr;
                        CorrTransProbs.(CondlRestnFnNames{rr}).(fn).Mean(jj+1)=sum(mr.*Values_jjplus1);
                        CorrTransProbs.(CondlRestnFnNames{rr}).(fn).StdDeviation(jj+1)=sqrt(sum(mr.*(Values_jjplus1-CorrTransProbs.(CondlRestnFnNames{rr}).(fn).Mean(jj+1)).^2));
                    end
                    mr=StationaryDist(:,jj).*RestrictionValues(:,jj,rr); % restricted mass at age jj (not normalized: includes the age weight)
                    if sum(mr)>0
                        Xc=Values_jj-CorrTransProbs.(CondlRestnFnNames{rr}).(fn).Mean(jj);
                        Rbuf(:,:,rowjj,ff,rr)=[mr, mr.*Xc, mr.*Xc.^2]';
                        Ractive(rowjj,ff,rr)=true;
                    else
                        Rbuf(:,:,rowjj,ff,rr)=0;
                        Ractive(rowjj,ff,rr)=false;
                    end
                end
            end
        end

        %% Propagate every measure in flight one age (jj -> jj+1), all in one product with P_jj
        Wall=reshape(permute(Ubuf,[1,3,2]),[Kmax*nFns,N_a*N_semizze_reshape]);
        if useCondlRest==1
            Wall=[Wall; reshape(permute(Rbuf,[1,3,4,5,2]),[3*Kmax*nFns*nRest,N_a*N_semizze_reshape])];
        end
        Wall=full(Wall*P_jj); % now signed measures over the age-(jj+1) states
        Ubuf=permute(reshape(Wall(1:Kmax*nFns,:),[Kmax,nFns,N_a*N_semizze_reshape]),[1,3,2]);
        if useCondlRest==1
            Rbuf=permute(reshape(Wall(Kmax*nFns+1:end,:),[3,Kmax,nFns,nRest,N_a*N_semizze_reshape]),[1,5,2,3,4]);
        end

        %% Per-function: read off the measures that reached a requested horizon, then transition probabilities
        for ff=1:nFns
            fn=FnsToEvalNames{ff};
            Values_jj=Values_now(:,ff);
            Values_jjplus1=Values_last(:,ff);

            % (iii) Per-age AutoCov and AutoCorr at each horizon (transition j0 -> j0+k, read off when jj+1=j0+k)
            for j0=max(1,jj-Kmax+1):jj
                row=mod(j0-1,Kmax)+1;
                if Uactive(row,ff)
                    kk=jj+1-j0;
                    hh=find(horizons==kk);
                    if ~isempty(hh)
                        Yc=Values_jjplus1-CorrTransProbs.(fn).Mean(jj+1);
                        CorrTransProbs.(fn).(['AutoCovariance',horizonstr{hh}])(j0)=Ubuf(row,:,ff)*Yc;
                        denom=CorrTransProbs.(fn).StdDeviation(j0)*CorrTransProbs.(fn).StdDeviation(jj+1);
                        % Threshold guards against "0/0" for variables that are constant within
                        % an age (e.g. an agej-only fn): StdDev there is floating-point noise
                        % (~1e-15), so denom can be ~1e-30 and the ratio explodes. 1e-15 is far
                        % below any real-world variance and well above numerical noise.
                        if denom>1e-15
                            CorrTransProbs.(fn).(['AutoCorrelation',horizonstr{hh}])(j0)=CorrTransProbs.(fn).(['AutoCovariance',horizonstr{hh}])(j0)/denom;
                        end
                    end
                end
            end

            %% (iii-b) Conditional restrictions (see the lowmemory=0 branch for the formulas)
            if useCondlRest==1
                for rr=1:nRest
                    for j0=max(1,jj-Kmax+1):jj
                        row=mod(j0-1,Kmax)+1;
                        if Ractive(row,ff,rr)
                            kk=jj+1-j0;
                            hh=find(horizons==kk);
                            if ~isempty(hh)
                                pairs=Rbuf(:,:,row,ff,rr).*RestrictionValues(:,jj+1,rr)'; % keep those who satisfy the restriction at age jj+1 too
                                pairmass=sum(pairs(1,:));
                                CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMass',horizonstr{hh}])(j0)=pairmass;
                                if pairmass>0
                                    y=Values_jjplus1';
                                    d1=sum(pairs(2,:))/pairmass; % pair mean of x_j0, minus the restricted Mean(j0)
                                    muy=sum(pairs(1,:).*y)/pairmass; % pair mean of x_{jj+1}
                                    varx=sum(pairs(3,:))/pairmass-d1^2;
                                    vary=sum(pairs(1,:).*(y-muy).^2)/pairmass;
                                    covxy=sum(pairs(2,:).*(y-muy))/pairmass;
                                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMean_j',horizonstr{hh}])(j0)=CorrTransProbs.(CondlRestnFnNames{rr}).(fn).Mean(j0)+d1;
                                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairMean_jplusk',horizonstr{hh}])(j0)=muy;
                                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairStdDeviation_j',horizonstr{hh}])(j0)=sqrt(max(varx,0));
                                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['PairStdDeviation_jplusk',horizonstr{hh}])(j0)=sqrt(vary);
                                    CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['AutoCovariance',horizonstr{hh}])(j0)=covxy;
                                    denom=sqrt(max(varx,0))*sqrt(vary);
                                    if denom>1e-15
                                        CorrTransProbs.(CondlRestnFnNames{rr}).(fn).(['AutoCorrelation',horizonstr{hh}])(j0)=covxy/denom;
                                    end
                                end
                            end
                        end
                    end
                end
            end

            %% (iv) Transition probabilities (only when requested for this fn)
            if simoptions.transprobs(ff)==1
                if isempty(simoptions.transprobquantiles)
                    % Default: unique values per age (size can differ across ages -> cell array)
                    % P_v_cell=cell(N_j-1,1);

                    if massj==0
                        continue
                    end
                    [fvals_j,~,idx_j]=unique(gather(Values_jj));
                    [fvals_jplus1,~,idx_jp]=unique(gather(Values_jjplus1));
                    n_fvals_j=max(idx_j);
                    n_fvals_jp=max(idx_jp);

                    distj_cpu=gather(StationaryDist(:,jj)./massj);
                    % Bin columns of P_jj by idx_jp (right-multiply by sparse indicator) and mass-weighted-aggregate rows by idx_j (left-multiply)
                    S_jp=sparse(1:N_a*N_semizze_reshape,idx_jp,1,N_a*N_semizze_reshape,n_fvals_jp);
                    S_j=sparse(1:N_a*N_semizze_reshape,idx_j,1,N_a*N_semizze_reshape,n_fvals_j);
                    massPerBin_j=accumarray(idx_j,distj_cpu,[n_fvals_j,1]);
                    P_v=full(S_j'*(distj_cpu.*(P_jj*S_jp)))./max(massPerBin_j,eps);

                    CorrTransProbs.(fn).TransitionProbs{jj}=P_v;
                    CorrTransProbs.(fn).TransitionValues_j{jj}=fvals_j;
                    CorrTransProbs.(fn).TransitionValues_jplus1{jj}=fvals_jplus1;
                    CorrTransProbs.(fn).TransitionMass_j{jj}=massPerBin_j;

                else
                    % Quantile binning -> fixed-size (n_fvals, n_fvals, N_j-1) 3-D array
                    % n_fvals=simoptions.transprobquantiles;
                    % P_v_3d=nan(n_fvals,n_fvals,N_j-1);

                    if massj==0 || massjplus1==0
                        continue
                    end
                    idx_j=gather(LocalQuantileIndex(Values_jj,distj,n_fvals));
                    idx_jp=gather(LocalQuantileIndex(Values_jjplus1,distjplus1,n_fvals));

                    distj_cpu=gather(distj);
                    % Bin columns of P_jj by idx_jp (right-multiply by sparse indicator) and mass-weighted-aggregate rows by idx_j (left-multiply)
                    S_jp=sparse(1:N_a*N_semizze_reshape,idx_jp,1,N_a*N_semizze_reshape,n_fvals);
                    S_j=sparse(1:N_a*N_semizze_reshape,idx_j,1,N_a*N_semizze_reshape,n_fvals);
                    massPerBin_j=accumarray(idx_j,distj_cpu,[n_fvals,1]);
                    CorrTransProbs.(fn).TransitionProbs(:,:,jj)=full(S_j'*(distj_cpu.*(P_jj*S_jp)))./max(massPerBin_j,eps);
                end
            end
        end
    end

end


CorrTransProbs.Notes='Mean and StdDeviation are 1xN_j. AutoCovariance and AutoCorrelation are 1x(N_j-1), with index jj corresponding to the transition from age jj to age jj+1; AutoCovariance_kK and AutoCorrelation_kK (for K in simoptions.timehorizons) are 1x(N_j-K), index jj is the pair of ages jj and jj+K. Under a conditional restriction the auto-covariances are over the pairs that satisfy the restriction at both ages, centered on the pair means (PairMean_j, PairMean_jplusk), and PairMass is the population mass of those pairs. TransitionProbs (when requested) is a cell {N_j-1} of (possibly varying-size) matrices, or a 3-D (nquantiles, nquantiles, N_j-1) array when simoptions.transprobquantiles is set. TransitionValues_j and TransitionValues_jplus1 (cells {N_j-1}) give the unique function values labelling the rows and columns of TransitionProbs{jj} respectively, and TransitionMass_j (cell {N_j-1}) gives the within-age mass of each origin bin/row (none of these are provided when using transprobquantiles, where bins are quantiles rather than values).';

end


function idx=LocalQuantileIndex(Values_jj,distj,n_fvals)
% Map Values_jj into 1..n_fvals bins by quantiles of the within-age distribution distj
[SortedValues,sortindex]=sort(Values_jj);
SortedDist=distj(sortindex);
CumSortedDist=cumsum(SortedDist);
quantilecutoffs=nan(n_fvals-1,1,'gpuArray');
for qq=1:n_fvals-1
    [~,qqind]=max(CumSortedDist>qq*1/n_fvals);
    quantilecutoffs(qq)=SortedValues(qqind);
end
idx=ones(size(Values_jj),'gpuArray');
idx(Values_jj<=quantilecutoffs(1))=1;
for qq=2:n_fvals-1
    idx(logical((Values_jj>quantilecutoffs(qq-1)).*(Values_jj<=quantilecutoffs(qq))))=qq;
end
idx(Values_jj>quantilecutoffs(end))=n_fvals;
end
