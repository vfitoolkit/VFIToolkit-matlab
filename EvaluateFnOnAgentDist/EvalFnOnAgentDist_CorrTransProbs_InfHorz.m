function CorrTransProbs=EvalFnOnAgentDist_CorrTransProbs_InfHorz(StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, simoptions)
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
    simoptions.n_e=0;
    simoptions.n_semiz=0;
    % Internal options
    simoptions.alreadygridvals=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions, 'transprobs')
        simoptions.transprobs=zeros(length(fieldnames(FnsToEvaluate)),1); % which variables to calculate transition probabilities for (input as cell of FnsToEval names)
    end
    if ~isfield(simoptions, 'timehorizons')
        simoptions.timehorizons=[]; % calculate the correlation and trans prob at these time horizons in addition to the one-period ones (can be a vector, e.g. [5,10] calculates the 1, 5 and 10 period correlations and transition probabilities; the 1 period is always calculated regardless)
    end
    if ~isfield(simoptions, 'transprobquantiles')
        simoptions.transprobquantiles=[]; % e.g., =5 will calculate the transition probabilities for quintiles
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

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

l_daprime=size(Policy,1);
if simoptions.gridinterplayer==1
    l_daprime=l_daprime-1;
end
a_gridvals=CreateGridvals(n_a,a_grid,1);
% Switch to z_gridvals
if simoptions.alreadygridvals==0
    [z_gridvals, ~, simoptions]=ExogShockSetup(n_z,z_grid,[],Parameters,simoptions,1);
elseif simoptions.alreadygridvals==1
    z_gridvals=z_grid;
end

CorrTransProbs=struct();

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

%% I want to do some things now, so that they can be used in setting up conditional restrictions
StationaryDist=reshape(StationaryDist,[N_a*N_z,1]);

% Make sure things are on the gpu (they should already be)
StationaryDist=gpuArray(StationaryDist);
Policy=gpuArray(Policy);

% Switch to PolicyValues, and permute
PolicyValues=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid,a_grid,simoptions);
PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]

%% Create big transition matrix P
N_semiz=0; % NOT YET IMPLEMENTED
N_e=0; % NOT YET IMPLEMENTED
pi_semiz=[];
pi_e=[];
P=CreatePTransitionMatrix(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z,pi_e,Parameters,simoptions);

%%
for ff=1:length(FnsToEvalNames)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
    Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
    Values=reshape(Values,[N_a*N_z,1]);

    %% Calculate the correlation
    % Correlation(x,y)=Cov(x,u)/(stddev(x)*stddev(y))
    % So first calculate the covariance and the two standard deviations
    meanV=sum(StationaryDist.*Values);
    stddevV=sqrt(sum(StationaryDist.*(Values-meanV).^2));
    % Calculate covariance between this period and next period values
    Covar=(StationaryDist.*Values)'*P*Values - meanV*meanV;
    % Calculate the correlation
    Corr=Covar/(stddevV*stddevV);

    CorrTransProbs.(FnsToEvalNames{ff}).Mean=meanV;
    CorrTransProbs.(FnsToEvalNames{ff}).StdDeviation=stddevV;
    CorrTransProbs.(FnsToEvalNames{ff}).AutoCovariance=Covar;
    CorrTransProbs.(FnsToEvalNames{ff}).AutoCorrelation=Corr;
    
    %% Calculate transition probabilties
    if simoptions.transprobs(ff)==1
        if isempty(simoptions.transprobquantiles)
            [vv,~,indexes]=unique(Values);
            n_fvals=length(vv); % number of unique values of the FnsToEvaluate{ff}
        else
            quantilesindicator=zeros(size(Values));
            % Convert from values into quantile indexes
            [SortedValues,sortindex]=sort(Values);
            SortedDist=StationaryDist(sortindex);
            CumSortedDist=cumsum(SortedDist);
            quantilecutoffs=nan(simoptions.transprobquantiles-1,1);
            for qq=1:simoptions.transprobquantiles-1
                [~,qqind]=max(CumSortedDist>qq*1/simoptions.transprobquantiles);
                quantilecutoffs(qq)=SortedValues(qqind);
            end
            % qq=1
            quantilesindicator(Values<=quantilecutoffs(1))=1;
            for qq=2:simoptions.transprobquantiles-1
                quantilesindicator(logical((Values>quantilecutoffs(qq-1)).*(Values<=quantilecutoffs(qq))))=qq;
            end
            quantilesindicator(Values>quantilecutoffs(end))=simoptions.transprobquantiles;
            % Call it indexes so that it is used in this same manner
            indexes=quantilesindicator;
            n_fvals=simoptions.transprobquantiles;
        end
        % Pintermediate: sum transition probabilities for next period based accumulating the unique values
        Pintermediate=zeros(N_a*N_z,n_fvals);
        for ii=1:N_a*N_z
            Pintermediate(ii,:)=accumarray(indexes,full(P(ii,:)));
        end
        % Final: weighted sum of rows based on this period weights
        P_v=zeros(n_fvals,n_fvals); % transition probabilities for the values
        Pintermediate=StationaryDist.*Pintermediate;
        for kk=1:n_fvals
            P_v(:,kk)=accumarray(indexes,Pintermediate(:,kk))./accumarray(indexes,StationaryDist);
        end
        
        CorrTransProbs.(FnsToEvalNames{ff}).TransitionProbs=P_v;
    end

    if ~isempty(simoptions.timehorizons)
        for t_c=1:length(simoptions.timehorizons)
            tt=simoptions.timehorizons(t_c);
            Ptt=P^tt;

            % Following is largely just a copy-paste of the tt=1 version, with Ptt in place of P
            % Calculate covariance between this period and next period values
            Covar=(StationaryDist.*Values)'*Ptt*Values - meanV*meanV;
            % Calculate the correlation
            Corr=Covar/(stddevV*stddevV);

            CorrTransProbs.(FnsToEvalNames{ff}).(['tperiods',num2str(tt)]).AutoCovariance=Covar;
            CorrTransProbs.(FnsToEvalNames{ff}).(['tperiods',num2str(tt)]).AutoCorrelation=Corr;

            % Calculate transition probabilties
            if simoptions.transprobs(ff)==1
                % n_fvals & indexes were already created when we did the 1-period transition probabilites, so can just reuse them [because still on same ff]

                % Pintermediate: sum transition probabilities for next period based accumulating the unique values
                Pintermediate=zeros(N_a*N_z,n_fvals);
                for ii=1:N_a*N_z
                    Pintermediate(ii,:)=accumarray(indexes,full(Ptt(ii,:)));
                end
                % Final: weighted sum of rows based on this period weights
                P_v=zeros(n_fvals,n_fvals); % transition probabilities for the values
                Pintermediate=StationaryDist.*Pintermediate;
                for kk=1:n_fvals
                    P_v(:,kk)=accumarray(indexes,Pintermediate(:,kk))./accumarray(indexes,StationaryDist);
                end

                CorrTransProbs.(FnsToEvalNames{ff}).(['tperiods',num2str(tt)]).TransitionProbs=P_v;
            end

        end
    end
end



end
