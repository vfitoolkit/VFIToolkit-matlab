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
%   .TransitionValues_j, .TransitionValues_jplus1  cells {N_j-1} of the unique function
%                     values labelling the rows/columns of TransitionProbs{jj}
%                     (not provided when simoptions.transprobquantiles is set)
%   .TransitionMass_j cell {N_j-1}, within-age mass of each origin bin (row) of
%                     TransitionProbs{jj}; multiply by the age weight to get population
%                     mass (not provided when simoptions.transprobquantiles is set)
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

if isfield(simoptions,'conditionalrestrictions')
    warning('Have not yet implemented simoptions.conditionalrestrictions for AutoCorrTransProbs_FHorz so ignoring them, ask on forum if you need this')
end
if ~isempty(simoptions.timehorizons)
    error('AutoCorrTransProbs_FHorz: simoptions.timehorizons (multi-period autocorrelations) not yet implemented; will implement later')
end
if ~isequal(simoptions.agegroupings,1:1:N_j)
    error('AutoCorrTransProbs_FHorz: simoptions.agegroupings (age bins) not yet implemented; will implement later')
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
[z_gridvals_J, pi_z_J, simoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,simoptions,3);
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



if simoptions.lowmemory==0
    %% Per-age transition matrices P_jj (jj=1..N_j-1), via the existing helper
    disp('Old Version')
    P_cell=cell(N_j-1,1);
    if N_e==0
        if N_z==0
            if N_semiz==0 % none
                for jj=1:N_j-1
                    P_cell{jj}=CreatePTransitionMatrix(Policy(:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],[],[],Parameters,simoptions);
                end
            else % semiz
                for jj=1:N_j-1
                    P_cell{jj}=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J(:,:,:,jj),[],[],Parameters,simoptions);
                end
            end
        else % z
            if N_semiz==0
                for jj=1:N_j-1
                    P_cell{jj}=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],pi_z_J(:,:,jj),[],Parameters,simoptions);
                end
            else % semiz,z
                for jj=1:N_j-1
                    P_cell{jj}=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J(:,:,:,jj),pi_z_J(:,:,jj),[],Parameters,simoptions);
                end
            end
        end
    else
        if N_z==0
            if N_semiz==0 % e
                for jj=1:N_j-1
                    P_cell{jj}=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],[],simoptions.pi_e_J(:,jj+1),Parameters,simoptions);
                end
            else % semiz,e
                for jj=1:N_j-1
                    P_cell{jj}=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J(:,:,:,jj),[],simoptions.pi_e_J(:,jj+1),Parameters,simoptions);
                end
            end
        else
            if N_semiz==0 % z,e
                for jj=1:N_j-1
                    P_cell{jj}=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,[],pi_z_J(:,:,jj),simoptions.pi_e_J(:,jj+1),Parameters,simoptions);
                end
            else % semiz,z,e
                for jj=1:N_j-1
                    P_cell{jj}=CreatePTransitionMatrix(Policy(:,:,:,jj),l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J(:,:,:,jj),pi_z_J(:,:,jj),simoptions.pi_e_J(:,jj+1),Parameters,simoptions);
                end
            end
        end
    end

    disp('New Version')
    P_cell_new=CreatePTransitionMatrix_J(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,simoptions.pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters,simoptions);


    disp('Compare')
    for jj=1:N_j-1
        max(abs(P_cell{jj}-P_cell_new{jj})) % should all be zero
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
        CorrTransProbs.(fn).AutoCovariance=nan(1,N_j-1,'gpuArray');
        CorrTransProbs.(fn).AutoCorrelation=nan(1,N_j-1,'gpuArray');
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

    % Last evaluated Values per function (so at jj>=2 each function recycles its own
    % age-jj values; a single shared variable would hand it whichever function was
    % evaluated last in the ff loop)
    Values_last=zeros(N_a*N_semizze_reshape,length(FnsToEvalNames),'gpuArray');

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

        %% Per-function computation
        for ff=1:length(FnsToEvalNames)
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

            % (iii) Per-age AutoCov and AutoCorr (transition j -> j+1)
            % Use the centered form for AutoCov: more numerically stable than E[XY]-EX*EY
            % when X,Y are nearly constant (the raw-moment form cancels two large numbers
            % into a noisy tiny one).
            if massj>0
                Xc=Values_jj-CorrTransProbs.(fn).Mean(jj);
                Yc=Values_jjplus1-CorrTransProbs.(fn).Mean(jj+1);
                CorrTransProbs.(fn).AutoCovariance(jj)=full((distj.*Xc)'*P_jj*Yc);
                denom=CorrTransProbs.(fn).StdDeviation(jj)*CorrTransProbs.(fn).StdDeviation(jj+1);
                % Threshold guards against "0/0" for variables that are constant within
                % an age (e.g. an agej-only fn): StdDev there is floating-point noise
                % (~1e-15), so denom can be ~1e-30 and the ratio explodes. 1e-15 is far
                % below any real-world variance and well above numerical noise.
                if denom>1e-15
                    CorrTransProbs.(fn).AutoCorrelation(jj)=CorrTransProbs.(fn).AutoCovariance(jj)/denom;
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


CorrTransProbs.Notes='Mean and StdDeviation are 1xN_j. AutoCovariance and AutoCorrelation are 1x(N_j-1), with index jj corresponding to the transition from age jj to age jj+1. TransitionProbs (when requested) is a cell {N_j-1} of (possibly varying-size) matrices, or a 3-D (nquantiles, nquantiles, N_j-1) array when simoptions.transprobquantiles is set. TransitionValues_j and TransitionValues_jplus1 (cells {N_j-1}) give the unique function values labelling the rows and columns of TransitionProbs{jj} respectively, and TransitionMass_j (cell {N_j-1}) gives the within-age mass of each origin bin/row (none of these are provided when using transprobquantiles, where bins are quantiles rather than values).';

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
