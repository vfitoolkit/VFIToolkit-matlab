function CorrTransProbsPath=EvalFnOnTransPath_AutoCorrTransProbs_InfHorz(FnsToEvaluate,AgentDistPath,PolicyPath,PricePath,ParamPath, Parameters, T, n_d, n_a, n_z, d_grid, a_grid,z_grid, pi_z,transpathoptions,simoptions)
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
%
% No z is n_z=0 (z_grid=[], pi_z=[]). iid e shocks are given in simoptions (n_e, e_grid, pi_e), as for
% EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz; pi_e may be [N_e,T], in which case pi_e(:,t) is the distribution of
% the e realized IN period t. z_grid and pi_z may vary over the path (pi_z(:,:,t) is the transition from t to t+1).
% AgentDistPath is [n_a,(n_z),(n_e),T] and PolicyPath [rows,n_a,(n_z),(n_e),T].
% Semi-exogenous states are given in simoptions (n_semiz, semiz_grid, SemiExoStateFn or pi_semiz, l_dsemiz), as for
% EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz; then AgentDistPath is [n_a,n_semiz,(n_z),(n_e),T], PolicyPath
% [rows,n_a,n_semiz,(n_z),(n_e),T], and FnsToEvaluate inputs are (d,aprime,a,semiz,z,e,...). pi_semiz may vary over the
% path (SemiExoStateFn depending on a ParamPath parameter, or a 4-D pi_semiz), slice t being the transition from t to t+1.

if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.transprobs=zeros(length(fieldnames(FnsToEvaluate)),1); % intended that user inputs names, and this is converted to vector internally
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
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions,'transprobs')
        simoptions.transprobs=zeros(length(fieldnames(FnsToEvaluate)),1); % intended that user inputs names, and this is converted to vector internally
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
end

if isfield(simoptions,'conditionalrestrictions')
    warning('Have not yet implemented simoptions.conditionalrestrictions for CorrTransProbs_InfHorz, ask on forum if you need this')
end

l_d=length(n_d);
if n_d(1)==0
    l_d=0;
end
l_a=length(n_a);

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end
% Read here, because ExogShockSetup_InfHorz_TPath below removes n_e from simoptions
n_e=simoptions.n_e;
N_e=prod(n_e);
if N_e==0
    l_e=0;
else
    l_e=length(n_e);
end
% The push below moves the mass with the a' index of the policy, which does not include an experience asset (that
% needs aprimeFn), so an experience asset would silently give the wrong answer
if simoptions.experienceasset>=1
    error('EvalFnOnTransPath_AutoCorrTransProbs_InfHorz: experienceasset is not yet implemented')
end
% (z,e) are kept as one exogenous dimension, z first, which is a singleton when there is neither
N_zr=max(N_z,1);
N_er=max(N_e,1);
N_ze=N_zr*N_er;
if N_z==0 && N_e==0
    n_ze=0;
elseif N_z>0 && N_e==0
    n_ze=n_z;
elseif N_z==0 && N_e>0
    n_ze=n_e;
elseif N_z>0 && N_e>0
    n_ze=[n_z,n_e];
end
% Semi-exogenous states (simoptions.n_semiz) go in front of (z,e), semiz fastest: from here on the exogenous dimension
% (n_ze, N_ze) is (semiz,z,e). FnsToEvaluate inputs are (d,aprime,a,semiz,z,e,...).
N_semiz=prod(simoptions.n_semiz);
N_semizr=max(N_semiz,1);
N_zeonly=N_ze; % (z,e) alone, without semiz
if N_semiz==0
    l_semiz=0;
else
    l_semiz=length(simoptions.n_semiz);
    if n_d(1)==0
        error('EvalFnOnTransPath_AutoCorrTransProbs_InfHorz: semi-exogenous states (simoptions.n_semiz) require a decision variable: the last simoptions.l_dsemiz decision variables determine the semiz transitions')
    end
    N_ze=N_semiz*N_ze;
    if N_z==0 && N_e==0
        n_ze=simoptions.n_semiz;
    else
        n_ze=[simoptions.n_semiz,n_ze];
    end
end


%%
% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);

%%
FnsToEvalNames=fieldnames(FnsToEvaluate);
for ff=1:length(FnsToEvalNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
    if length(temp)>(l_d+l_a+l_a+l_semiz+l_z+l_e)
        FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_semiz+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,semiz,z,e)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluateCell{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
end
% For the subfunctions we want the following
simoptions.outputasstructure=0;
simoptions.AggVarNames=FnsToEvalNames;

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

%% Check if using _tminus1 and/or _tplus1 variables.
[tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames,ParamPathNames,{},transpathoptions);

%% The _tminus1 and _tplus1 values come from transpathoptions.initialvalues
% Checked here rather than at first use, so that a missing initialvalues names itself
% instead of surfacing as 'Unrecognized field name' from inside the loop over t.
if (use_tminus1price==1 || use_tminus1params==1 || use_tminus1AggVars==1) && ~isfield(transpathoptions,'initialvalues')
    error('EvalFnOnTransPath_AutoCorrTransProbs_InfHorz: a _tminus1 input is used, so transpathoptions.initialvalues must be given (it supplies the period-1 values)')
end

%%
% d_gridvals=CreateGridvals(n_d,d_grid,1);
a_gridvals=CreateGridvals(n_a,a_grid,1);
if simoptions.experienceasset==0
    if simoptions.gridinterplayer==1
        if isscalar(n_a)
            N_aprime=N_a+(N_a-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,N_a,N_a)',a_grid(1:n_a(1)),linspace(1,N_a,N_aprime)');
            aprime_grid=temp;
            % n_aprime=n_a;
        else
            N_a1prime=n_a(1)+(n_a(1)-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,n_a(1),n_a(1))',a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)');
            aprime_grid=[temp; a_grid(n_a(1)+1:end)];
            % n_aprime=[N_a1prime,n_a(2:end)];
        end
        % aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
        simoptions.policyind2val_finegridinput=1; % aprime_grid contains the fine grid for the first asset (tells PolicyInd2Val_InfHorz_TPath)
    else
        aprime_grid=a_grid;
        % aprime_gridvals=a_gridvals;
    end
elseif simoptions.experienceasset>=1
    % omit a2 from aprime_gridvals
    if simoptions.gridinterplayer==1
        N_a1prime=n_a(1)+(n_a(1)-1)*simoptions.ngridinterp;
        temp=interp1(linspace(1,n_a(1),n_a(1))',a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)');
        if length(n_a)==2
            aprime_grid=temp; % omit a2
            % n_aprime=N_a1prime; % omit a2
        elseif length(n_a)>2 % more than one a1
            aprime_grid=[temp; a_grid(n_a(1)+1:end-1)];
            % n_aprime=[N_a1prime,n_a(2:end-1)];
        end
        % aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
        simoptions.policyind2val_finegridinput=1; % aprime_grid contains the fine grid for the first asset (tells PolicyInd2Val_InfHorz_TPath)
    else
        aprime_grid=a_grid;
        % aprime_gridvals=CreateGridvals(n_a(1:end-1),a_grid(1:sum(n_a(1:end-1))),1); % omit a2
    end
end
% gridpiboth=4: both the grids (for the functions) and the transition probabilities (for the push). The setup writes
% zpathtrivial/epathtrivial/zepathtrivial into transpathoptions, and when z or e varies over the path the paths are in
% transpathoptions.pi_z_T, pi_e_T and ze_gridvals_T (ze_gridvals is the (z,e) grid, z varies first).
[~, pi_z, ~, ~, pi_e, ~, ze_gridvals, transpathoptions, simoptions]=ExogShockSetup_InfHorz_TPath(n_z,z_grid,pi_z,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,simoptions,4);
if N_semiz>0
    % The semiz grid (fixed over the path), put in front of the (z,e) grid, semiz fastest, and pi_semiz
    % [N_semiz,N_semiz,N_dsemiz] (gpu); when transpathoptions.semizpathtrivial=0 the path is in transpathoptions.pi_semiz_T
    % [N_semiz,N_semiz,N_dsemiz,T], slice t being the transition from t to t+1
    [semiz_gridvals, pi_semiz, transpathoptions, simoptions]=SemiExogShockSetup_InfHorz_TPath(n_d,d_grid,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,simoptions,4);
    if N_z==0 && N_e==0
        ze_gridvals=semiz_gridvals;
    else
        ze_gridvals=[repmat(semiz_gridvals,N_zeonly,1),repelem(ze_gridvals,N_semiz,1)];
    end
end


%%
PolicyPath=gpuArray(reshape(PolicyPath,[size(PolicyPath,1),N_a,N_ze,T]));
% (PolicyInd2Val_InfHorz_TPath takes the exogenous sizes, (semiz,z,e) together, as its n_z input)
PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_ze,T,d_grid,aprime_grid,simoptions,1);
l_daprime=size(PolicyValuesPath,1);
if N_z==0 && N_e==0 && N_semiz==0
    PolicyValuesPermutePath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,T]),[2,1,3]); %[N_a,l_d+l_a,T]
else
    PolicyValuesPermutePath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_ze,T]),[2,3,1,4]); %[N_a,N_z*N_e,l_d+l_a,T]
end

N_states=N_a*N_ze;
AgentDistPath=gpuArray(reshape(AgentDistPath,[N_states,T]));
numFnsToEvaluate=length(FnsToEvalNames);

% preallocate
for ff=1:numFnsToEvaluate
    CorrTransProbsPath.(FnsToEvalNames{ff}).Mean=zeros(1,T);
    CorrTransProbsPath.(FnsToEvalNames{ff}).StdDeviation=zeros(1,T);
    CorrTransProbsPath.(FnsToEvalNames{ff}).AutoCovariance=zeros(1,T);
    CorrTransProbsPath.(FnsToEvalNames{ff}).AutoCorrelation=zeros(1,T);
    % cannot preallocate for TransProbs, as do not yet know size, instead do it in t==2 of loop over tt below
end

%% The push from period tt-1 to tt is the Tan (2020) two-step: the policy step with the period tt-1 policy (a sparse
% gpuArray built from the gpu policy slice, one nonzero per state, or two with the grid-interpolation weights), then
% pi_z, then pi_e along a new dimension (kron(pi_e,.) without forming the kron); the full transition matrix is never
% formed. The policy step goes from (a,z,e) to (a',z), so is N_a*N_z by N_a*N_z*N_e. The z step uses the period tt-1
% pi_z (the transition from tt-1 to tt) and the e step the distribution of the e realized in period tt.
% The centered lag measures of every function are the columns of one matrix and go through the step in one product;
% the transition probabilities push each origin bin's mass the same way, in blocks of 64 bins, on the gpu.
zindex=N_a*repmat(gpuArray(0:1:N_zr-1),1,N_er); % the z index of each (z,e) column (z varies first), to get the index into (a',z)
if simoptions.gridinterplayer==0
    IIind=gpuArray((1:1:N_states)');
elseif simoptions.gridinterplayer==1
    IIind=gpuArray(repmat((1:1:N_states)',2,1));
end
if N_semiz>0
    % With semiz the policy step goes from (a,semiz,z,e) to (a',semiz',z), with probabilities pi_semiz(semiz,semiz',d2) at
    % the period tt-1 policy d2 (the last simoptions.l_dsemiz decision variables), N_semiz nonzeros per state (twice that
    % with the grid interpolation layer); the z and e steps follow as without semiz. [As EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz.]
    l_dsemiz=simoptions.setup_semiexo.l_d2;
    n_dsemiz=simoptions.setup_semiexo.n_d2;
    semizind=gpuArray(repmat(repelem((1:1:N_semiz)',N_a,1),N_zr*N_er,1)); % semiz of each state (a,semiz,z,e)
    zind=gpuArray(repmat(repelem((1:1:N_zr)',N_a*N_semiz,1),N_er,1)); % z of each state (a,semiz,z,e)
    if simoptions.gridinterplayer==0
        IIind=gpuArray(repmat((1:1:N_states)',N_semiz,1));
    elseif simoptions.gridinterplayer==1
        IIind=gpuArray(repmat((1:1:N_states)',2*N_semiz,1));
    end
    pi_semiz_gpu=gpuArray(pi_semiz);
end
if N_z>1
    pi_z_gpu=gpuArray(pi_z);
end
if N_e>0
    pi_e_gpu=gpuArray(reshape(pi_e,[1,N_e]));
end
% This period's values, means and standard deviations of every function (one column each), kept as the lag for the next period
Values_now=zeros(N_states,numFnsToEvaluate,'gpuArray');
vv_now=cell(numFnsToEvaluate,1); % the distinct values and the bin index of each state (functions with transition probabilities), kept as the lag
indexes_now=cell(numFnsToEvaluate,1);
meanV_now=zeros(1,numFnsToEvaluate,'gpuArray');
stddevV_now=zeros(1,numFnsToEvaluate,'gpuArray');

%%
for tt=1:T
    % The _tminus1 values first, while Parameters still holds the previous period's prices and parameters
    % (setting them after period tt is written in would make every _tminus1 equal to period tt)
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            if tt>1
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
            else
                Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
    end
    if use_tminus1params==1
        for pp=1:length(tminus1paramNames)
            if tt>1
                Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
            else
                Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
            end
        end
    end
    if use_tminus1AggVars==1
        for pp=1:length(tminus1AggVarsNames)
            if tt>1
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=CorrTransProbsPath.(tminus1AggVarsNames{pp}).Mean(tt-1);
            else
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
            end
        end
    end

    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            % Period T is the final stationary eqm, so the price after it is the same as at T
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(min(tt+1,T),PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
    end

    if N_z==0 && N_e==0 && N_semiz==0
        PolicyValuesPermute=PolicyValuesPermutePath(:,:,tt);
    else
        PolicyValuesPermute=PolicyValuesPermutePath(:,:,:,tt);
    end
    AgentDist=AgentDistPath(:,tt); % [N_states,1]
    if transpathoptions.zepathtrivial==0
        if N_semiz==0
            ze_gridvals=transpathoptions.ze_gridvals_T(:,:,tt);
        else
            ze_gridvals=[repmat(semiz_gridvals,N_zeonly,1),repelem(transpathoptions.ze_gridvals_T(:,:,tt),N_semiz,1)]; % (semiz,z,e), semiz fastest
        end
    end

    if tt>1
        AgentDist_lag=AgentDistPath(:,tt-1);
        Values_lag=Values_now;
        meanV_lag=meanV_now;
        stddevV_lag=stddevV_now;
        vv_lag=vv_now;
        indexes_lag_all=indexes_now;
    end

    %% This period's values of every function, and their means and standard deviations (one reduction each over the columns)
    Values_now=zeros(N_states,numFnsToEvaluate,'gpuArray');
    for ff=1:numFnsToEvaluate
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
        Values=EvalFnOnAgentDist_Grid(FnsToEvaluateCell{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_ze,a_gridvals,ze_gridvals);
        Values_now(:,ff)=reshape(Values,[N_states,1]);
    end
    meanV_now=sum(AgentDist.*Values_now,1);
    stddevV_now=sqrt(sum(AgentDist.*(Values_now-meanV_now).^2,1));
    for ff=1:numFnsToEvaluate
        if simoptions.transprobs(ff)==1
            [vv_now{ff},~,indexes_now{ff}]=unique(Values_now(:,ff)); % on the gpu; the bin of each state
        end
    end
    meanV_cpu=gather(meanV_now);
    stddevV_cpu=gather(stddevV_now);
    for ff=1:numFnsToEvaluate
        CorrTransProbsPath.(FnsToEvalNames{ff}).Mean(tt)=meanV_cpu(ff);
        CorrTransProbsPath.(FnsToEvalNames{ff}).StdDeviation(tt)=stddevV_cpu(ff);
    end

    %% Can only calculate the rest from period 2 on (tt is treated as next period, tt-1 as this period)
    if tt>1
        % The shock steps: the transition of z from tt-1 to tt, and the distribution of the e realized in tt
        if N_z>1 && transpathoptions.zpathtrivial==0
            pi_z_gpu=transpathoptions.pi_z_T(:,:,tt-1);
        end
        if N_e>0 && transpathoptions.epathtrivial==0
            pi_e_gpu=reshape(transpathoptions.pi_e_T(:,tt),[1,N_e]);
        end
        if N_semiz>0 && transpathoptions.semizpathtrivial==0
            pi_semiz_gpu=transpathoptions.pi_semiz_T(:,:,:,tt-1); % the transition of semiz from tt-1 to tt
        end
        % The policy step of period tt-1
        Policy_lag=PolicyPath(:,:,:,tt-1); % [size(PolicyPath,1),N_a,N_z*N_e]
        if l_a==1
            Policy_aprime=shiftdim(Policy_lag(l_d+1,:,:),1);
        elseif l_a==2
            Policy_aprime=shiftdim(Policy_lag(l_d+1,:,:)+n_a(1)*(Policy_lag(l_d+2,:,:)-1),1);
        elseif l_a==3
            Policy_aprime=shiftdim(Policy_lag(l_d+1,:,:)+n_a(1)*(Policy_lag(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy_lag(l_d+3,:,:)-1),1);
        elseif l_a==4
            Policy_aprime=shiftdim(Policy_lag(l_d+1,:,:)+n_a(1)*(Policy_lag(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy_lag(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy_lag(l_d+4,:,:)-1),1);
        else
            error('EvalFnOnTransPath_AutoCorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
        end
        if N_semiz==0
            Policy_aprimez=reshape(Policy_aprime+zindex,[],1); % [N_states,1], index into (a',z), on the gpu
            if simoptions.gridinterplayer==0
                Gammatranspose_lag=sparse(Policy_aprimez,IIind,ones(N_states,1,'gpuArray'),N_a*N_zr,N_states); % sparse gpuArray
            elseif simoptions.gridinterplayer==1
                % two a' points per state: the lower grid point and the one above it, with the second-layer weights
                L2index=Policy_lag(end-1,:,:); % L2 index (end-1 because end is L2flag)
                L2flag=Policy_lag(end,:,:);
                L2index(L2flag==1)=1;                        % force all weight to lower grid point
                L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
                probupper=reshape((L2index-1)/(simoptions.ngridinterp+1),[],1); % [N_states,1]: probability of the upper grid point
                Gammatranspose_lag=sparse([Policy_aprimez;Policy_aprimez+1],IIind,[1-probupper;probupper],N_a*N_zr,N_states); % sparse gpuArray
            end
        else
            % The joint d2 index of the period tt-1 policy (d2 are the last l_dsemiz decision variable rows)
            Policy_dsemiz=reshape(Policy_lag(l_d-l_dsemiz+1,:,:),[N_states,1]);
            for ii=2:l_dsemiz
                Policy_dsemiz=Policy_dsemiz+prod(n_dsemiz(1:ii-1))*(reshape(Policy_lag(l_d-l_dsemiz+ii,:,:),[N_states,1])-1);
            end
            semiztransitions=pi_semiz_gpu(semizind+N_semiz*gpuArray(0:1:N_semiz-1)+N_semiz*N_semiz*(Policy_dsemiz-1)); % [N_states,N_semiz], pi_semiz(semiz,semiz',d2) at the policy d2
            Policy_aprimesemizz=reshape(Policy_aprime,[N_states,1])+N_a*gpuArray(0:1:N_semiz-1)+N_a*N_semiz*(zind-1); % [N_states,N_semiz], index of (a',semiz',z)
            if simoptions.gridinterplayer==0
                Gammatranspose_lag=sparse(reshape(Policy_aprimesemizz,[],1),IIind,reshape(semiztransitions,[],1),N_a*N_semiz*N_zr,N_states); % sparse gpuArray
            elseif simoptions.gridinterplayer==1
                % two a' points per state: the lower grid point and the one above it, with the second-layer weights, each with every semiz'
                L2index=Policy_lag(end-1,:,:); % L2 index (end-1 because end is L2flag)
                L2flag=Policy_lag(end,:,:);
                L2index(L2flag==1)=1;                        % force all weight to lower grid point
                L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
                probupper=reshape((L2index-1)/(simoptions.ngridinterp+1),[],1); % [N_states,1]: probability of the upper grid point
                Gammatranspose_lag=sparse([reshape(Policy_aprimesemizz,[],1);reshape(Policy_aprimesemizz,[],1)+1],IIind,[reshape((1-probupper).*semiztransitions,[],1);reshape(probupper.*semiztransitions,[],1)],N_a*N_semiz*N_zr,N_states); % sparse gpuArray
            end
        end

        %% Auto-covariance and auto-correlation between periods tt-1 and tt, every function at once
        % Covar is E[(x_{t-1}-mean_{t-1})(y_t-mean_t)] over the joint distribution of the state pair, which is the lag
        % distribution pushed one period: so the lag distribution is the weight here (weighting by AgentDist instead would
        % apply the step twice). The centered lag measures AgentDist_lag.*(x_{t-1}-mean) are pushed one period and
        % integrated against (x_t-mean).
        propagated=Gammatranspose_lag*(AgentDist_lag.*(Values_lag-meanV_lag)); % N_a*N_semiz*N_z x numFnsToEvaluate
        if N_z>1
            propagated=reshape(pagemtimes(reshape(propagated,[N_a*N_semizr,N_z,numFnsToEvaluate]),pi_z_gpu),[N_a*N_semizr*N_z,numFnsToEvaluate]);
        end
        if N_e>0
            propagated=reshape(reshape(propagated,[N_a*N_semizr*N_zr,1,numFnsToEvaluate]).*pi_e_gpu,[N_states,numFnsToEvaluate]); % e step: e is the slowest index of the states
        end
        Covar=sum(propagated.*(Values_now-meanV_now),1);
        Corr=Covar./(stddevV_lag.*stddevV_now);
        Covar_cpu=gather(Covar);
        Corr_cpu=gather(Corr);
        for ff=1:numFnsToEvaluate
            CorrTransProbsPath.(FnsToEvalNames{ff}).AutoCovariance(tt-1)=Covar_cpu(ff);
            CorrTransProbsPath.(FnsToEvalNames{ff}).AutoCorrelation(tt-1)=Corr_cpu(ff);
        end

        %% Transition probabilities between the value bins of period tt-1 and period tt
        for ff=1:numFnsToEvaluate
            if simoptions.transprobs(ff)==1
                vv=vv_now{ff};
                indexes=indexes_now{ff};
                vv2=vv_lag{ff};
                indexes_lag=indexes_lag_all{ff};
                if isequal(vv,vv2) % cannot handle the case where it is not just same list of values every period (e.g., cannot handle that in some period we don't see one of the values)
                    n_fvals=length(vv); % number of unique values of the FnsToEvaluate{ff}
                    % Each origin bin's lag mass is pushed one period (origin bins as columns, in blocks of 64) and binned by
                    % the period-tt value; row b of P_v is the destination distribution of origin bin b (conditioned on the
                    % period tt-1 value bin, so sum(P_v,2) is one where the bin has mass)
                    massPerBin=gather(accumarray(indexes_lag,AgentDist_lag,[n_fvals,1]));
                    S_dest=sparse(gpuArray((1:1:N_states)'),indexes,ones(N_states,1,'gpuArray'),N_states,n_fvals); % destination bin indicator, sparse gpuArray
                    P_v=zeros(n_fvals,n_fvals); % transition probabilities for the values
                    for b1=1:64:n_fvals
                        b2=min(b1+63,n_fvals);
                        inblock=find(indexes_lag>=b1 & indexes_lag<=b2); % on the gpu
                        temp=zeros(N_states,b2-b1+1,'gpuArray'); % the origin bins' lag masses, one column each
                        temp(inblock+N_states*(indexes_lag(inblock)-b1))=AgentDist_lag(inblock);
                        temp=Gammatranspose_lag*temp;
                        if N_z>1
                            temp=reshape(pagemtimes(reshape(temp,[N_a*N_semizr,N_z,b2-b1+1]),pi_z_gpu),[N_a*N_semizr*N_z,b2-b1+1]);
                        end
                        if N_e>0
                            temp=reshape(reshape(temp,[N_a*N_semizr*N_zr,1,b2-b1+1]).*pi_e_gpu,[N_states,b2-b1+1]); % e step
                        end
                        P_v(b1:b2,:)=gather((S_dest'*temp)')./massPerBin(b1:b2);
                    end
                    if tt==2
                        CorrTransProbsPath.(FnsToEvalNames{ff}).TransitionProbs=repmat(P_v,1,1,T-1);
                    else
                        CorrTransProbsPath.(FnsToEvalNames{ff}).TransitionProbs(:,:,tt-1)=P_v;
                    end
                else
                    warning('EvalFnOnTransPath_AutoCorrTransProbs_InfHorz: not returning TransitionProbs for %s, as the distinct values it takes in period %i differ from those in period %i (not implemented)',FnsToEvalNames{ff},tt,tt-1)
                end
            end
        end
    end
end
