function  [MarketClearanceVec]=MarketClearance_Case1(SSvalues_AggVars,p, MarketClearanceEqns, Parameters, MarketClearanceParamNames)
%For models with more than one market MultiMarketCriterion determines which method to use to combine them.

MarketClearanceVec=ones(1,length(MarketClearanceEqns),'gpuArray')*Inf;
for i=1:length(MarketClearanceEqns)
    if isempty(MarketClearanceParamNames(i).Names)  % check for 'MarketClearanceParamNames(i).Names={}'
        MarketClearanceParamsVec=gpuArray([]);
    else
        MarketClearanceParamsVec=gpuArray(CreateVectorFromParams(Parameters,MarketClearanceParamNames(i).Names));
    end
    
    MarketClearanceVec(i)=MarketClearanceEqns{i}(SSvalues_AggVars, p, MarketClearanceParamsVec);
end


end
