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
        if all(CorrTransProbs.(CondlRestnFnNames{rr}).RestrictedSampleMass==0)
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
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,N_j,nwhichpages);
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ismatrix(whichcombos) && isequal(size(whichcombos),[numFnsToEvaluate,nwhichpages]) && nwhichpages~=N_j
        whichcombos=repmat(reshape(whichcombos,[numFnsToEvaluate,1,nwhichpages]),[1,N_j,1]); % (fn, page): apply to every start age
    elseif ismatrix(whichcombos) && isequal(size(whichcombos),[numFnsToEvaluate,N_j])
        whichcombos=repmat(whichcombos,[1,1,nwhichpages]); % (fn, start age): apply to every page
    end
    if ~isequal(size(whichcombos,1:3),[numFnsToEvaluate,N_j,nwhichpages])
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(N_j),',',num2str(nwhichpages),'] (number of FnsToEvaluate, N_j, 1+number of conditional restrictions)'])
    end
    whichcombos=double(whichcombos);
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
%% The policy step: Gammatranspose_cell{jj} maps a measure over the age-jj states (a,semiz,z,e) to a measure over (a',semiz',z) [z kept, e summed out]
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
Gammatranspose_cell=cell(N_j-1,1);
if N_semiz==0
    % Add the z index of each (z,e) column (z varies first) to get the index into (a',z)
    zindex=N_a*repmat(gpuArray(0:1:N_zr-1),1,N_er);
    if simoptions.gridinterplayer==0
        Policy_aprimez=gather(Policy_aprime+zindex);
        IIind=(1:1:N_states)';
        for jj=1:N_j-1
            Gammatranspose_cell{jj}=sparse(reshape(Policy_aprimez(:,:,jj),[],1),IIind,ones(N_states,1),N_a*N_zr,N_states);
        end
    elseif simoptions.gridinterplayer==1
        Policy_aprimez=gather(cat(4,Policy_aprime,Policy_aprime+1)+zindex); % [N_a,N_semizze_reshape,N_j,2]
        PolicyProbs=gather(cat(4,1-probupper,probupper)); % [N_a,N_semizze_reshape,N_j,2]
        IIind=repmat((1:1:N_states)',2,1);
        for jj=1:N_j-1
            Gammatranspose_cell{jj}=sparse(reshape(Policy_aprimez(:,:,jj,:),[],1),IIind,reshape(PolicyProbs(:,:,jj,:),[],1),N_a*N_zr,N_states);
        end
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
        for jj=1:N_j-1
            semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % linear index into pi_semiz_J_short and idxshort
            Policy_aprimesemizz_jj=repelem(Policy_aprime_cpu(:,1,jj),1,N_semizshort)+N_a*(idxshort(semizindex_short_jj)-1)+zprimeoffset; % [N_states,N_semizshort]: index into (a',semiz',z)
            Gammatranspose_cell{jj}=sparse(Policy_aprimesemizz_jj,II2,pi_semiz_J_short(semizindex_short_jj),N_a*N_semiz*N_zr,N_states);
        end
    elseif simoptions.gridinterplayer==1
        Policy_aprime2_cpu=cat(2,Policy_aprime_cpu,Policy_aprime_cpu+1); % [N_states,2,N_j]: the lower grid point and the one above it
        probupper_cpu=gather(reshape(probupper,[N_states,1,N_j]));
        PolicyProbs_cpu=cat(2,1-probupper_cpu,probupper_cpu); % [N_states,2,N_j]
        II2=repelem((1:1:N_states)',1,2*N_semizshort);
        for jj=1:N_j-1
            semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1);
            Policy_aprimesemizz_jj=repelem(Policy_aprime2_cpu(:,:,jj),1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short_jj)-1),1,2)+zprimeoffset; % [N_states,2*N_semizshort]: the two grid points, each with every semiz'
            PolicyProbs_jj=repelem(PolicyProbs_cpu(:,:,jj),1,N_semizshort).*repmat(pi_semiz_J_short(semizindex_short_jj),1,2);
            Gammatranspose_cell{jj}=sparse(Policy_aprimesemizz_jj,II2,PolicyProbs_jj,N_a*N_semiz*N_zr,N_states); % sparse() accumulates repeated indexes (the two grid points coincide only at the top of the grid)
        end
    end
end
if N_z>0
    pi_z_J_cpu=gather(pi_z_J);
end
if N_e>0
    pi_e_J_cpu=gather(simoptions.pi_e_J);
end
%% Per-function computation
for ff=1:length(FnsToEvalNames)
    fn=FnsToEvalNames{ff};
    if ~any(whichcombos(ff,:,:),'all') % no combination of this function is wanted
        continue
    end
    selU=reshape(whichcombos(ff,:,1)==1,[1,N_j]); % the start ages wanted for the unrestricted outputs of this function

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
    Values_cpu=gather(Values);

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

    % (iii) Per-age AutoCov and AutoCorr at each horizon (transition j -> j+k). Centered form: more numerically stable than
    % E[XY]-EX*EY when X,Y are nearly constant (the raw-moment form cancels two large numbers into a noisy tiny one).
    AutoCov=cell(1,nhorizons);
    AutoCorr=cell(1,nhorizons);
    for hh=1:nhorizons
        AutoCov{hh}=nan(1,N_j-horizons(hh),'gpuArray');
        AutoCorr{hh}=nan(1,N_j-horizons(hh),'gpuArray');
    end
    for jj=1:N_j-1
        if ~selU(jj) % this start age is not wanted
            continue
        end
        massj=sum(StationaryDist(:,jj));
        if massj>0
            distj=StationaryDist(:,jj)./massj;
            Xc=Values(:,jj)-MeanV(jj);
            propagated=gather(distj.*Xc)'; % 1 x N_states signed measure over the age-jj states, on the cpu
            for kk=1:min(Kmax,N_j-jj)
                % Tan step from age jj+kk-1 to age jj+kk
                temp=Gammatranspose_cell{jj+kk-1}*propagated'; % policy step: now over (a',semiz',z)
                if N_z>0
                    temp=reshape(reshape(temp,[N_a*N_semizr,N_zr])*pi_z_J_cpu(:,:,jj+kk-1),[N_a*N_semizr*N_zr,1]); % z step
                end
                if N_e>0
                    temp=kron(pi_e_J_cpu(:,jj+kk),temp); % e step: the e realized at age jj+kk
                end
                propagated=temp'; % now a signed measure over the age-(jj+kk) states
                hh=find(horizons==kk);
                if ~isempty(hh)
                    Yc=Values_cpu(:,jj+kk)-gather(MeanV(jj+kk));
                    AutoCov{hh}(jj)=propagated*Yc;
                    denom=StdDevV(jj)*StdDevV(jj+kk);
                    if denom>1e-15
                        AutoCorr{hh}(jj)=AutoCov{hh}(jj)/denom;
                    end
                end
            end
        end
    end

    if any(selU) % the unrestricted outputs of this function are wanted (the horizon outputs at the selected start ages; the rest stay NaN)
    CorrTransProbs.(fn).Mean=MeanV; % reported at every age: computed anyway (the horizon outputs starting at j need the means at j+k)
    CorrTransProbs.(fn).StdDeviation=StdDevV;
    for hh=1:nhorizons
        CorrTransProbs.(fn).(['AutoCovariance',horizonstr{hh}])=AutoCov{hh};
        CorrTransProbs.(fn).(['AutoCorrelation',horizonstr{hh}])=AutoCorr{hh};
    end
    end % any(selU)

    %% (iii-b) Conditional restrictions: means/std devs at each age over those satisfying the restriction, and the
    % auto-covariance over the pairs that satisfy it at both ages. Three signed measures over the age-jj states are
    % propagated together (three columns of one Tan step): the restricted mass m, m.*Xc and m.*Xc.^2 (Xc centered on the
    % restricted age-jj mean). Masking the propagated measures with the restriction at age jj+k gives the pair population,
    % and its mass, its means and variances of x_j and x_{j+k}, and their covariance follow. [E_pair[(x_j-c)(x_{j+k}-mu_y)]
    % is the pair covariance for ANY constant c, since E_pair[x_{j+k}-mu_y]=0, so centering x_j on the restricted age-jj
    % mean rather than on the (not yet known) pair mean is exact.]
    if useCondlRest==1
        for rr=1:length(CondlRestnFnNames)
            if ~any(whichcombos(ff,:,1+rr)) % this restriction is not wanted for this function
                continue
            end
            selR=reshape(whichcombos(ff,:,1+rr)==1,[1,N_j]); % the start ages wanted for this (function, restriction)
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
            RestrictionValues_cpu=gather(RestrictionValues(:,:,rr));
            for jj=1:N_j-1
                if ~selR(jj) % this start age is not wanted
                    continue
                end
                mr=StationaryDist(:,jj).*RestrictionValues(:,jj,rr); % restricted mass at age jj (not normalized: includes the age weight)
                if sum(mr)==0 && sum(StationaryDist(:,jj))>0
                    for hh=1:nhorizons
                        if jj<=N_j-horizons(hh)
                            PairMass{hh}(jj)=0;
                        end
                    end
                end
                if sum(mr)>0
                    Xc=Values(:,jj)-MeanR(jj);
                    propagated=gather([mr, mr.*Xc, mr.*Xc.^2])'; % 3 x N_states, on the cpu
                    for kk=1:min(Kmax,N_j-jj)
                        % Tan step from age jj+kk-1 to age jj+kk, the three measures as three columns
                        temp=Gammatranspose_cell{jj+kk-1}*propagated'; % (N_a*N_semizr*N_zr) x 3
                        if N_z>0
                            for cc=1:3
                                temp(:,cc)=reshape(reshape(temp(:,cc),[N_a*N_semizr,N_zr])*pi_z_J_cpu(:,:,jj+kk-1),[N_a*N_semizr*N_zr,1]);
                            end
                        end
                        if N_e>0
                            temp=kron(pi_e_J_cpu(:,jj+kk),temp);
                        end
                        propagated=temp'; % now over the age-(jj+kk) states
                        hh=find(horizons==kk);
                        if ~isempty(hh)
                            pairs=propagated.*RestrictionValues_cpu(:,jj+kk)'; % keep those who satisfy the restriction at age jj+kk too
                            pairmass=sum(pairs(1,:));
                            PairMass{hh}(jj)=pairmass;
                            if pairmass>0
                                y=Values_cpu(:,jj+kk)';
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
            CorrTransProbs.(CondlRestnFnNames{rr}).(fn).Mean=MeanR; % reported at every age (computed anyway)
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
    %% (iv) Transition probabilities between value bins (only when requested for this fn): horizon 1, unrestricted
    % Each origin bin's mass is pushed one age forward (the same Tan step, with the origin bins as columns, in blocks of
    % 64) and binned by the function's value at age jj+1; row b of TransitionProbs{jj} is the destination distribution
    % of origin bin b, so rows sum to one.
    if simoptions.transprobs(ff)==1 && any(selU)
        if isempty(simoptions.transprobquantiles)
            % Default: unique values per age (size can differ across ages -> cell array)
            P_v_cell=cell(N_j-1,1);
            fvals_j_cell=cell(N_j-1,1); % unique fn values at jj (labels the rows of TransitionProbs{jj})
            fvals_jplus1_cell=cell(N_j-1,1); % unique fn values at jj+1 (labels the columns)
            massbin_j_cell=cell(N_j-1,1); % within-age mass of each origin bin (row)
            for jj=1:N_j-1
        if ~selU(jj) % this start age is not wanted
            continue
        end
                massj=sum(StationaryDist(:,jj));
                if massj==0
                    continue
                end
                [fvals_j_cell{jj},~,idx_j]=unique(Values_cpu(:,jj));
                [fvals_jplus1_cell{jj},~,idx_jp]=unique(Values_cpu(:,jj+1));
                n_fvals_j=max(idx_j);
                n_fvals_jp=max(idx_jp);
                distj_cpu=gather(StationaryDist(:,jj)./massj);
                massPerBin_j=accumarray(idx_j,distj_cpu,[n_fvals_j,1]);
                S_jp=sparse(1:N_a*N_semizze_reshape,idx_jp,1,N_a*N_semizze_reshape,n_fvals_jp); % destination bin indicator
                P_v=zeros(n_fvals_j,n_fvals_jp);
                for b1=1:64:n_fvals_j
                    b2=min(b1+63,n_fvals_j);
                    inblock=(idx_j>=b1 & idx_j<=b2);
                    M=sparse(find(inblock),idx_j(inblock)-b1+1,distj_cpu(inblock),N_a*N_semizze_reshape,b2-b1+1); % the origin bins' masses, one column each
                    temp=full(Gammatranspose_cell{jj}*M);
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
        if ~selU(jj) % this start age is not wanted
            continue
        end
                massj=sum(StationaryDist(:,jj));
                massjplus1=sum(StationaryDist(:,jj+1));
                if massj==0 || massjplus1==0
                    continue
                end
                distj=StationaryDist(:,jj)./massj;
                distjplus1=StationaryDist(:,jj+1)./massjplus1;
                % Quantile bins of the within-age distributions: bin q holds the values up to the first sorted value whose
                % cumulative mass exceeds q/n_fvals (the same definition as the InfHorz command), at age jj and at age jj+1
                [SortedValues,sortindex]=sort(Values(:,jj));
                CumSortedDist=cumsum(distj(sortindex));
                quantilecutoffs=nan(n_fvals-1,1,'gpuArray');
                for qq=1:n_fvals-1
                    [~,qqind]=max(CumSortedDist>qq*1/n_fvals);
                    quantilecutoffs(qq)=SortedValues(qqind);
                end
                idx_j=ones(N_a*N_semizze_reshape,1,'gpuArray');
                for qq=2:n_fvals
                    idx_j(Values(:,jj)>quantilecutoffs(qq-1))=qq;
                end
                idx_j=gather(idx_j);
                [SortedValues,sortindex]=sort(Values(:,jj+1));
                CumSortedDist=cumsum(distjplus1(sortindex));
                quantilecutoffs=nan(n_fvals-1,1,'gpuArray');
                for qq=1:n_fvals-1
                    [~,qqind]=max(CumSortedDist>qq*1/n_fvals);
                    quantilecutoffs(qq)=SortedValues(qqind);
                end
                idx_jp=ones(N_a*N_semizze_reshape,1,'gpuArray');
                for qq=2:n_fvals
                    idx_jp(Values(:,jj+1)>quantilecutoffs(qq-1))=qq;
                end
                idx_jp=gather(idx_jp);
                distj_cpu=gather(distj);
                massPerBin_j=accumarray(idx_j,distj_cpu,[n_fvals,1]);
                S_jp=sparse(1:N_a*N_semizze_reshape,idx_jp,1,N_a*N_semizze_reshape,n_fvals);
                M=sparse(1:N_a*N_semizze_reshape,idx_j,distj_cpu,N_a*N_semizze_reshape,n_fvals); % the origin bins' masses, one column each
                temp=full(Gammatranspose_cell{jj}*M);
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
                P_v_3d(:,:,jj)=(S_jp'*temp)'./massPerBin_safe;
            end
            CorrTransProbs.(fn).TransitionProbs=P_v_3d;
        end
    end
end


CorrTransProbs.Notes='Mean and StdDeviation are 1xN_j. AutoCovariance and AutoCorrelation are 1x(N_j-1), with index jj corresponding to the transition from age jj to age jj+1; AutoCovariance_kK and AutoCorrelation_kK (for K in simoptions.timehorizons) are 1x(N_j-K), index jj is the pair of ages jj and jj+K. Under a conditional restriction the auto-covariances are over the pairs that satisfy the restriction at both ages, centered on the pair means (PairMean_j, PairMean_jplusk), and PairMass is the population mass of those pairs. TransitionProbs (when requested) is a cell {N_j-1} of (possibly varying-size) matrices, or a 3-D (nquantiles, nquantiles, N_j-1) array when simoptions.transprobquantiles is set. TransitionValues_j and TransitionValues_jplus1 (cells {N_j-1}) give the unique function values labelling the rows and columns of TransitionProbs{jj} respectively, and TransitionMass_j (cell {N_j-1}) gives the within-age mass of each origin bin/row (none of these are provided when using transprobquantiles, where bins are quantiles rather than values).';

end
