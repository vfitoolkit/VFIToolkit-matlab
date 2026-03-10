function AllStatsPath=EvalFnOnAggShocksPath_AllStats_InfHorz(GeneralizedTransitionFn,T,n_d,n_a,n_z,n_S,d_grid,a_grid,z_grid,FnsToEvaluate,Parameters,simoptions)

if ~exist('simoptions','var')
    simoptions.npoints=100;
    simoptions.nquantiles=20;
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
else
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100;
    end
    if ~isfield(simoptions,'nquantiles')
        simoptions.nquantiles=20;
    end
    if ~isfield(simoptions,'whichstats')
        simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
    end
end

%% Setup
PricePathNames=fieldnames(GeneralizedTransitionFn.PricePath);
AggShockNames=fieldnames(GeneralizedTransitionFn.AggShocksPath);

FnsToEvalNames=fieldnames(FnsToEvaluate);
for ff=1:length(FnsToEvalNames)
    AllStatsPath.(FnsToEvalNames{ff}).Mean=zeros(1,T);
    AllStatsPath.(FnsToEvalNames{ff}).Median=zeros(1,T);
    AllStatsPath.(FnsToEvalNames{ff}).RatioMeanToMedian=zeros(1,T);
    AllStatsPath.(FnsToEvalNames{ff}).Variance=zeros(1,T);
    AllStatsPath.(FnsToEvalNames{ff}).StdDeviation=zeros(1,T);
    AllStatsPath.(FnsToEvalNames{ff}).LorenzCurve=zeros(simoptions.npoints,T);
    AllStatsPath.(FnsToEvalNames{ff}).Gini=zeros(1,T);
    AllStatsPath.(FnsToEvalNames{ff}).QuantileCutoffs=zeros(simoptions.nquantiles+1,T);  % Includes the min and max values
    AllStatsPath.(FnsToEvalNames{ff}).QuantileMeans=zeros(simoptions.nquantiles,T);
end


%% Loop over tt=1:T and get AllStats
% We already have GeneralizedTransitionFn.AgentDistPath so just calculate from this.
for tt=1:T
    for pp=1:length(PricePathNames)
        temp=GeneralizedTransitionFn.PricePath.(PricePathNames{pp});
        Parameters.(PricePathNames{pp})=temp(tt);
    end
    for SS_c=1:length(n_S)
        temp=GeneralizedTransitionFn.AggShocksPath.(AggShockNames{SS_c});
        Parameters.(AggShockNames{SS_c})=temp(tt);
    end
    
    AgentDist=GeneralizedTransitionFn.AgentDistPath(:,:,tt);
    Policy=GeneralizedTransitionFn.PolicyPath(:,:,:,tt);

    % would be slightly faster if I pass FnsToEvaluateCell and FnsToEvaluateParamNames
    AllStats_tt=EvalFnOnAgentDist_AllStats_Case1(AgentDist,Policy,FnsToEvaluate,Parameters,[],n_d,n_a,n_z,d_grid,a_grid,z_grid,simoptions);
    
    for ff=1:length(FnsToEvalNames)
        temp=AllStatsPath.(FnsToEvalNames{ff}).Mean;
        temp(tt)=AllStats_tt.(FnsToEvalNames{ff}).Mean;
        AllStatsPath.(FnsToEvalNames{ff}).Mean=temp;

        temp=AllStatsPath.(FnsToEvalNames{ff}).Median;
        temp(tt)=AllStats_tt.(FnsToEvalNames{ff}).Median;
        AllStatsPath.(FnsToEvalNames{ff}).Median=temp;

        temp=AllStatsPath.(FnsToEvalNames{ff}).RatioMeanToMedian;
        temp(tt)=AllStats_tt.(FnsToEvalNames{ff}).RatioMeanToMedian;
        AllStatsPath.(FnsToEvalNames{ff}).RatioMeanToMedian=temp;

        temp=AllStatsPath.(FnsToEvalNames{ff}).Variance;
        temp(tt)=AllStats_tt.(FnsToEvalNames{ff}).Variance;
        AllStatsPath.(FnsToEvalNames{ff}).Variance=temp;

        temp=AllStatsPath.(FnsToEvalNames{ff}).StdDeviation;
        temp(tt)=AllStats_tt.(FnsToEvalNames{ff}).StdDeviation;
        AllStatsPath.(FnsToEvalNames{ff}).StdDeviation=temp;

        temp=AllStatsPath.(FnsToEvalNames{ff}).LorenzCurve;
        temp(:,tt)=AllStats_tt.(FnsToEvalNames{ff}).LorenzCurve;
        AllStatsPath.(FnsToEvalNames{ff}).LorenzCurve=temp;

        temp=AllStatsPath.(FnsToEvalNames{ff}).Gini;
        temp(tt)=AllStats_tt.(FnsToEvalNames{ff}).Gini;
        AllStatsPath.(FnsToEvalNames{ff}).Gini=temp;

        temp=AllStatsPath.(FnsToEvalNames{ff}).QuantileCutoffs;
        temp(:,tt)=AllStats_tt.(FnsToEvalNames{ff}).QuantileCutoffs;
        AllStatsPath.(FnsToEvalNames{ff}).QuantileCutoffs=temp;

        temp=AllStatsPath.(FnsToEvalNames{ff}).QuantileMeans;
        temp(:,tt)=AllStats_tt.(FnsToEvalNames{ff}).QuantileMeans;
        AllStatsPath.(FnsToEvalNames{ff}).QuantileMeans=temp;
    end

end









end