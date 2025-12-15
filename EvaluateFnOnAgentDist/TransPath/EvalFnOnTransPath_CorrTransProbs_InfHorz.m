function CorrTransProbsPath=EvalFnOnTransPath_CorrTransProbs_InfHorz(FnsToEvaluate,AgentDistPath,PolicyPath,PricePath,ParamPath, Parameters, T, n_d, n_a, n_z, d_grid, a_grid,z_grid, pi_z,simoptions)
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

if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.transprobs=zeros(length(fieldnames(FnsToEvaluate)),1);
    simoptions.gridinterplayer=0;
    % Model setup
    simoptions.experienceasset=0;
    simoptions.n_e=0;
    simoptions.n_semiz=0;
    % Internal options
    simoptions.alreadygridvals=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions, 'transprobs')
        simoptions.transprobs=zeros(length(fieldnames(FnsToEvaluate)),1);
    end
    % Model solution
    if ~isfield(simoptions, 'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    % Model setup
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    % Internal options
    if ~isfield(simoptions, 'alreadygridvals')
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
l_z=length(n_z);

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);


%%
% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);

%%
FnsToEvalNames=fieldnames(FnsToEvaluate);
for ff=1:length(FnsToEvalNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
    if length(temp)>(l_d+l_a+l_a+l_z)
        FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
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
if isstruct(FnsToEvaluate)
    [tplus1priceNames,tminus1priceNames,~,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames);
    % tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk.
    % But omit tminus1AggVarsNames as AggVars are anyway not allowed to take AggVars as inputs.
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tplus1pricePathkk=[];
end

use_tplus1price=0;
if ~isempty(tplus1priceNames)
    use_tplus1price=1;
end
use_tminus1price=0;
if ~isempty(tminus1priceNames)
    use_tminus1price=1;
    for tt=1:length(tminus1priceNames)
        if ~isfield(simoptions.initialvalues,tminus1priceNames{tt})
            dbstack
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{tt})
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

%%
d_gridvals=CreateGridvals(n_d,d_grid,1);
a_gridvals=CreateGridvals(n_a,a_grid,1);
if simoptions.experienceasset==0
    if simoptions.gridinterplayer==1
        if isscalar(n_a)
            N_aprime=N_a+(N_a-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,N_a,N_a)',a_grid(1:n_a(1)),linspace(1,N_a,N_aprime)');
            aprime_grid=temp;
            n_aprime=n_a;
        else
            N_a1prime=n_a(1)+(n_a(1)-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,n_a(1),n_a(1))',a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)');
            aprime_grid=[temp; a_grid(n_a(1)+1:end)];
            n_aprime=[N_a1prime,n_a(2:end)];
        end
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
    else
        aprime_gridvals=a_gridvals;
    end
elseif simoptions.experienceasset==1
    % omit a2 from aprime_gridvals
    if simoptions.gridinterplayer==1
        N_a1prime=n_a(1)+(n_a(1)-1)*simoptions.ngridinterp;
        temp=interp1(linspace(1,n_a(1),n_a(1))',a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)');
        if length(n_a)==2
            aprime_grid=temp; % omit a2
            n_aprime=N_a1prime; % omit a2
        elseif length(n_a)>2 % more than one a1
            aprime_grid=[temp; a_grid(n_a(1)+1:end-1)];
            n_aprime=[N_a1prime,n_a(2:end-1)];
        end
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
    else
        aprime_gridvals=CreateGridvals(n_a(1:end-1),a_grid(1:sum(n_a(1:end-1))),1); % omit a2
    end

end
z_gridvals=CreateGridvals(n_z,z_grid,1);


%%
PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);
PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_z,T,d_gridvals,aprime_gridvals,simoptions,1);
PolicyValuesPermutePath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_z,T]),[2,3,1,4]); %[N_a,N_z,l_d+l_a,T]

AgentDistPath=reshape(AgentDistPath,[N_a,N_z,T]);

% preallocate
for ff=1:length(FnsToEvalNames)
    CorrTransProbsPath.(FnsToEvalNames{ff}).Mean=zeros(1,T);
    CorrTransProbsPath.(FnsToEvalNames{ff}).StdDeviation=zeros(1,T);
    CorrTransProbsPath.(FnsToEvalNames{ff}).AutoCovariance=zeros(1,T);
    CorrTransProbsPath.(FnsToEvalNames{ff}).AutoCorrelation=zeros(1,T);
    % cannot preallocate for TransProbs, as do not yet know size, instead do it in t==2 of loop over tt below
end



%%
for tt=1:T
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            if tt>1
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
            else
                Parameters.([tminus1priceNames{pp},'_tminus1'])=simoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
    end
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
        end
    end
    
    PolicyValuesPermute=PolicyValuesPermutePath(:,:,:,tt);
    AgentDist=AgentDistPath(:,:,tt);

    if tt>1
        AgentDist_lag=AgentDistPath(:,:,tt-1);
    end

    %% Create big transition matrix P
    N_semiz=0; % NOT YET IMPLEMENTED
    N_e=0; % NOT YET IMPLEMENTED
    pi_semiz=[];
    pi_e=[];
    if tt>1
        P_lag=CreatePTransitionMatrix(PolicyPath(:,:,:,tt-1),l_d,l_a,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z,pi_e,simoptions);
        % Note: I suspect keeping P and P_lag would run out of memory.
        % This only works because Parameters is not used here as it would contain tt
    end
    
    %% Can only calculate most stats from period 2 on
    for ff=1:length(FnsToEvalNames)
        if tt>1
            Values_lag=Values;
            meanV_lag=meanV;
            stddevV_lag=stddevV;
        end

        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
        Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
        Values=reshape(Values,[N_a*N_z,1]);

        
        %% Calculate the correlation (tt is treated as next period, tt-1 as this period)
        % Correlation(x,y)=Cov(x,u)/(stddev(x)*stddev(y))
        % So first calculate the covariance and the two standard deviations

        % We don't need lag for some basics
        meanV=sum(AgentDist.*Values);
        stddevV=sqrt(sum(AgentDist.*(Values-meanV).^2));
        CorrTransProbsPath.(FnsToEvalNames{ff}).Mean(tt)=meanV;
        CorrTransProbsPath.(FnsToEvalNames{ff}).StdDeviation(tt)=stddevV;

        % For autocovar and autocorr we can do them from tt=2 on
        if tt>1
            % Calculate covariance between this period and next period values
            Covar=(AgentDist_lag.*Values_lag)'*P_lag*Values - meanV_lag*meanV; % AgentDist or AgentDist_lag??
            % Calculate the correlation
            Corr=Covar/(stddevV_lag*stddevV);
            CorrTransProbsPath.(FnsToEvalNames{ff}).AutoCovariance(tt-1)=Covar;
            CorrTransProbsPath.(FnsToEvalNames{ff}).AutoCorrelation(tt-1)=Corr;
        end
        

        %% Calculate transition probabilties (tt is treated as next period, tt-1 as this period)
        if tt>1
            if simoptions.transprobs(ff)==1
                [vv,~,indexes]=unique(Values);
                [vv2,~,indexes_lag]=unique(Values_lag);
                if all(vv==vv2) % cannot handle the case where it is not just same list of values every period (e.g., cannot handle that in some period we don't see one of the values)
                    n_fvals=length(vv); % number of unique values of the FnsToEvaluate{ff}
                    % Pintermediate: sum transition probabilities for next period based accumulating the unique values
                    Pintermediate=zeros(N_a*N_z,n_fvals);
                    for ii=1:N_a*N_z
                        Pintermediate(ii,:)=accumarray(indexes,full(P_lag(ii,:)));
                    end
                    % Final: weighted sum of rows based on this period weights
                    P_v=zeros(n_fvals,n_fvals); % transition probabilities for the values
                    Pintermediate=AgentDist_lag.*Pintermediate;
                    for kk=1:n_fvals
                        P_v(:,kk)=accumarray(indexes,Pintermediate(:,kk))./accumarray(indexes_lag,AgentDist_lag);
                    end

                    if tt==2
                        CorrTransProbs.(FnsToEvalNames{ff}).TransitionProbs=repmat(P_v,1,1,T-1);
                    else
                        CorrTransProbs.(FnsToEvalNames{ff}).TransitionProbs(:,:,tt-1)=P_v;
                    end
                end
            end
        end
    end
end