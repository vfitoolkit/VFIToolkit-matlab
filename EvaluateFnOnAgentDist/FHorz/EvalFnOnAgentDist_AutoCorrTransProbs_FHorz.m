function CorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_FHorz(StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, pi_z, simoptions)
% Returns stats on (auto) correlation and transition probabilities for FHorz models.
% Auto-correlation/-covariance are reported per-age (j -> j+1), so length N_j-1.
% Mean and StdDeviation are reported per-age, so length N_j.
%
% Use simoptions.transprobs={'name1','name2',...} (cell of FnsToEval names) to
% request transition probabilities for those functions (none by default).
%
% Outputs (per FnsToEvaluate field):
%   .Mean             1 x N_j
%   .StdDeviation     1 x N_j
%   .AutoCovariance   1 x (N_j-1)   transition j -> j+1
%   .AutoCorrelation  1 x (N_j-1)   transition j -> j+1
%   .TransitionProbs  cell {N_j-1} of n_fvals_j x n_fvals_{j+1} matrices (default),
%                     or n_fvals x n_fvals x (N_j-1) array when simoptions.transprobquantiles is set
%
% Not yet implemented (will error or be ignored):
%   simoptions.conditionalrestrictions  -- warn-and-ignore
%   simoptions.n_e>0 or simoptions.n_semiz>0  -- error
%   simoptions.timehorizons non-empty  -- error
%   simoptions.agegroupings non-default  -- error

%%
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.transprobs=zeros(length(fieldnames(FnsToEvaluate)),1);
    simoptions.timehorizons=[]; % multi-period horizons -- not yet implemented
    simoptions.transprobquantiles=[];
    simoptions.agegroupings=1:1:N_j; % age bins -- not yet implemented (default = each age separately)
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
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
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
end

if isfield(simoptions,'conditionalrestrictions')
    warning('Have not yet implemented simoptions.conditionalrestrictions for AutoCorrTransProbs_FHorz so ignoring them, ask on forum if you need this')
end
if prod(simoptions.n_semiz)>0
    error('AutoCorrTransProbs_FHorz: semi-exogenous (semiz) shocks not yet implemented; ask on forum if you need this')
end
if prod(simoptions.n_e)>0
    error('AutoCorrTransProbs_FHorz: iid (e) shocks not yet implemented; ask on forum if you need this')
end
if ~isempty(simoptions.timehorizons)
    error('AutoCorrTransProbs_FHorz: simoptions.timehorizons (multi-period autocorrelations) not yet implemented; will implement later')
end
if ~isequal(simoptions.agegroupings,1:1:N_j)
    error('AutoCorrTransProbs_FHorz: simoptions.agegroupings (age bins) not yet implemented; will implement later')
end

%%
N_a=prod(n_a);
N_z=prod(n_z);

if isempty(n_d) || prod(n_d)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

%% Exogenous shock grids (z only; semiz/e disallowed above)
[n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% Implement new way of handling FnsToEvaluate
l_daprime=size(Policy,1)-2*simoptions.gridinterplayer;

if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}};
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

%% Reshape StationaryDist, build PolicyValues with trailing-j axis
StationaryDist=gpuArray(reshape(StationaryDist,[N_a*N_z,N_j]));
Policy=gpuArray(Policy);

PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
PolicyValuesPermuteJ=permute(PolicyValues,[2,3,1,4]); % (N_a,N_z,l_daprime,N_j)

%% Per-age transition matrices P_jj (jj=1..N_j-1), via the existing helper
CTP_simoptions=struct();
CTP_simoptions.experienceasset=0;
CTP_simoptions.experienceassetz=0;
CTP_simoptions.experienceassete=0;
CTP_simoptions.experienceassetze=0;
CTP_simoptions.inheritanceasset=0;
CTP_simoptions.gridinterplayer=simoptions.gridinterplayer;
if simoptions.gridinterplayer==1
    CTP_simoptions.ngridinterp=simoptions.ngridinterp;
end

P_cell=cell(N_j-1,1);
for jj=1:N_j-1
    P_cell{jj}=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,0,N_z,0,[],pi_z,[],Parameters,CTP_simoptions);
end

%% Output
CorrTransProbs=struct();

%% Per-function computation
for ff=1:length(FnsToEvalNames)
    fn=FnsToEvalNames{ff};

    % (i) Per-age Values, shape (N_a*N_z, N_j)
    Values=nan(N_a*N_z,N_j,'gpuArray');
    for jj=1:N_j
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
        slice=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff},FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
        Values(:,jj)=reshape(slice,[N_a*N_z,1]);
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

    % (iii) Per-age AutoCov and AutoCorr (transition j -> j+1)
    % Use the centered form for AutoCov: more numerically stable than E[XY]-EX*EY
    % when X,Y are nearly constant (the raw-moment form cancels two large numbers
    % into a noisy tiny one).
    AutoCov=nan(1,N_j-1,'gpuArray');
    AutoCorr=nan(1,N_j-1,'gpuArray');
    for jj=1:N_j-1
        massj=sum(StationaryDist(:,jj));
        if massj>0
            distj=StationaryDist(:,jj)./massj;
            Xc=Values(:,jj)-MeanV(jj);
            Yc=Values(:,jj+1)-MeanV(jj+1);
            AutoCov(jj)=full((distj.*Xc)'*P_cell{jj}*Yc);
            denom=StdDevV(jj)*StdDevV(jj+1);
            % Threshold guards against "0/0" for variables that are constant within
            % an age (e.g. an agej-only fn): StdDev there is floating-point noise
            % (~1e-15), so denom can be ~1e-30 and the ratio explodes. 1e-15 is far
            % below any real-world variance and well above numerical noise.
            if denom>1e-15
                AutoCorr(jj)=AutoCov(jj)/denom;
            end
        end
    end

    CorrTransProbs.(fn).Mean=MeanV;
    CorrTransProbs.(fn).StdDeviation=StdDevV;
    CorrTransProbs.(fn).AutoCovariance=AutoCov;
    CorrTransProbs.(fn).AutoCorrelation=AutoCorr;

    %% (iv) Transition probabilities (only when requested for this fn)
    if simoptions.transprobs(ff)==1
        if isempty(simoptions.transprobquantiles)
            % Default: unique values per age (size can differ across ages -> cell array)
            P_v_cell=cell(N_j-1,1);
            for jj=1:N_j-1
                massj=sum(StationaryDist(:,jj));
                if massj==0
                    continue
                end
                Values_j=gather(Values(:,jj));
                Values_jp=gather(Values(:,jj+1));
                [~,~,idx_j]=unique(Values_j);
                [~,~,idx_jp]=unique(Values_jp);
                n_fvals_j=max(idx_j);
                n_fvals_jp=max(idx_jp);

                distj_cpu=gather(StationaryDist(:,jj)./massj);
                P_jj=P_cell{jj};

                % Sum P_jj columns into n_fvals_jp bins (by idx_jp), per source row ii
                Pintermediate=zeros(N_a*N_z,n_fvals_jp);
                for ii=1:N_a*N_z
                    Pintermediate(ii,:)=accumarray(idx_jp,full(P_jj(ii,:))',[n_fvals_jp,1])';
                end
                % Mass-weighted aggregation by source bin (idx_j)
                Pintermediate=distj_cpu.*Pintermediate;
                massPerBin_j=accumarray(idx_j,distj_cpu,[n_fvals_j,1]);
                P_v=zeros(n_fvals_j,n_fvals_jp);
                for kk=1:n_fvals_jp
                    P_v(:,kk)=accumarray(idx_j,Pintermediate(:,kk),[n_fvals_j,1])./max(massPerBin_j,eps);
                end
                P_v_cell{jj}=P_v;
            end
            CorrTransProbs.(fn).TransitionProbs=P_v_cell;
        else
            % Quantile binning -> fixed-size (n_fvals, n_fvals, N_j-1) 3-D array
            n_fvals=simoptions.transprobquantiles;
            P_v_3d=nan(n_fvals,n_fvals,N_j-1);
            for jj=1:N_j-1
                massj=sum(StationaryDist(:,jj));
                massjp=sum(StationaryDist(:,jj+1));
                if massj==0 || massjp==0
                    continue
                end
                distj=StationaryDist(:,jj)./massj;
                distjp=StationaryDist(:,jj+1)./massjp;

                idx_j=gather(LocalQuantileIndex(Values(:,jj),distj,n_fvals));
                idx_jp=gather(LocalQuantileIndex(Values(:,jj+1),distjp,n_fvals));

                distj_cpu=gather(distj);
                P_jj=P_cell{jj};
                Pintermediate=zeros(N_a*N_z,n_fvals);
                for ii=1:N_a*N_z
                    Pintermediate(ii,:)=accumarray(idx_jp,full(P_jj(ii,:))',[n_fvals,1])';
                end
                Pintermediate=distj_cpu.*Pintermediate;
                massPerBin_j=accumarray(idx_j,distj_cpu,[n_fvals,1]);
                for kk=1:n_fvals
                    P_v_3d(:,kk,jj)=accumarray(idx_j,Pintermediate(:,kk),[n_fvals,1])./max(massPerBin_j,eps);
                end
            end
            CorrTransProbs.(fn).TransitionProbs=P_v_3d;
        end
    end
end

CorrTransProbs.Notes='Mean and StdDeviation are 1xN_j. AutoCovariance and AutoCorrelation are 1x(N_j-1), with index jj corresponding to the transition from age jj to age jj+1. TransitionProbs (when requested) is a cell {N_j-1} of (possibly varying-size) matrices, or a 3-D (nquantiles, nquantiles, N_j-1) array when simoptions.transprobquantiles is set.';

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
