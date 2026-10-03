function CorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz(StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, simoptions)
% Returns stats on (auto) correlation and transition probabilities
% You must input the names for the FnsToEvaluate that you want the transition probabilities for (by default it won't do any)
% Done as simoptions.transprobs
%
% simoptions optional inputs
%
% Outputs:
% Mean (as it has to be calculated anyway as an intermediate step to correlation)
% StdDeviation (as it has to be calculated anyway as an intermediate step to correlation)
% AutoCovariance
% AutoCorrelation
% TransitionProbs (optional)
%
% Note: simoptions.conditionalrestrictions is not yet implemented

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
    % Internal options
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0;
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
    % Internal options
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
end

if isfield(simoptions,'conditionalrestrictions')
    warning('Have not yet implemented simoptions.conditionalrestrictions for CorrTransProbs_InfHorz so ignoring them, ask on forum if you need this')
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
% Keep the iid e shock (if any) for the transition step below, before e is folded into z
N_e_orig=prod(simoptions.n_e);
if N_e_orig>0
    pi_e_orig=simoptions.pi_e;
end
% Switch to z_gridvals (folding e and semiz into z if appropriate)
[n_z,z_gridvals,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_InfHorz(n_z,z_grid,simoptions,Parameters);

CorrTransProbs=struct();

%% I want to do some things now, so that they can be used in setting up conditional restrictions
StationaryDist=reshape(StationaryDist,[N_a*max(N_z,1),1]);

% Make sure things are on the gpu (they should already be)
StationaryDist=gpuArray(StationaryDist);
Policy=gpuArray(Policy);

% Switch to PolicyValues, and permute
PolicyValues=PolicyInd2Val_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid,simoptions);
if N_z==0
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a]),[2,1]); %[N_a,l_d+l_a]
else
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]
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
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
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

%% The computation: never form the transition matrix
% Cov(x_t,x_{t+k}) is the centered signed measure (dist.*(x-mean)) pushed k periods forward and then integrated against
% x. The push is the two-step of Tan (2020, Economics Letters) that the stationary distribution iteration uses: the
% policy step (Gammatranspose, a sparse N_a*N_z by N_a*N_z*N_e matrix with one nonzero per state, or two with the grid
% interpolation weights), then the shock step (times pi_z, then the kron with pi_e). The transition probabilities between
% value bins push each origin bin's mass the same way.
% [e was folded into z above by CreateGridvals_FnsToEvaluate_InfHorz (z varies first, then e): N_z now counts both,
% and the push keeps the markov z (N_zr states, pi_z) and the iid e (N_er states, pi_e) apart.]
N_er=max(N_e_orig,1);
N_zr=max(N_z,1)/N_er; % the markov z states
Policy=reshape(Policy,[size(Policy,1),N_a,max(N_z,1)]);
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
% Policy_aprime is [N_a,N_z]; add the markov-z index of each (z,e) column to get the index into (a',z)
zindex=N_a*repmat(gpuArray(0:1:N_zr-1),1,N_er);
N_states=N_a*max(N_z,1);
if simoptions.gridinterplayer==0
    Policy_aprimez=gather(Policy_aprime+zindex);
    Gammatranspose=sparse(reshape(Policy_aprimez,[],1),(1:1:N_states)',ones(N_states,1),N_a*N_zr,N_states);
elseif simoptions.gridinterplayer==1
    % two a' points per state: the lower grid point and the one above it, with the second-layer weights
    Policy_aprimez=gather(cat(3,Policy_aprime,Policy_aprime+1)+zindex); % [N_a,N_z,2]
    L2index=Policy(end-1,:,:); % L2 index (end-1 because end is L2flag)
    L2flag=Policy(end,:,:);
    L2index(L2flag==1)=1;                        % force all weight to lower grid point
    L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
    probupper=shiftdim((L2index-1)/(simoptions.ngridinterp+1),1); % [N_a,N_z]: probability of the upper grid point
    PolicyProbs=gather(cat(3,1-probupper,probupper)); % [N_a,N_z,2]
    Gammatranspose=sparse(reshape(Policy_aprimez,[],1),repmat((1:1:N_states)',2,1),reshape(PolicyProbs,[],1),N_a*N_zr,N_states);
end
if N_zr>1
    pi_z_cpu=gather(pi_z);
end
if N_er>1
    pi_e_cpu=gather(pi_e_orig(:));
end
if ~isempty(simoptions.timehorizons)
    maxhorizon=max([1,simoptions.timehorizons(:)']);
else
    maxhorizon=1;
end

%%
for ff=1:length(FnsToEvalNames)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
    Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
    Values=reshape(Values,[N_states,1]);
    Values_cpu=gather(Values);
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
        if N_zr>1
            temp=reshape(reshape(temp,[N_a,N_zr])*pi_z_cpu,[N_a*N_zr,1]); % z step
        end
        if N_er>1
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
    %% Transition probabilities between value bins (horizon 1 and simoptions.timehorizons): each origin bin's mass is
    % pushed forward (origin bins as columns, in blocks of 64) and binned by the function's value; row b is the
    % destination distribution of origin bin b, so rows sum to one
    if simoptions.transprobs(ff)==1
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
                if N_zr>1
                    for cc=1:size(temp,2)
                        temp(:,cc)=reshape(reshape(temp(:,cc),[N_a,N_zr])*pi_z_cpu,[N_a*N_zr,1]);
                    end
                end
                if N_er>1
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
