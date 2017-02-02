function  [MarketClearanceVec]=MarketClearance_Case1(SSvalues_AggVars,p, MarketClearanceEqns, Parameters, MarketClearanceParamNames)
%For models with more than one market MultiMarketCriterion determines which method to use to combine them.

% Includes check for cases in which no parameters are actually required
if (isempty(MarketClearanceParamNames) || strcmp(MarketClearanceParamNames(1),'')) % check for 'SSvalueParamNames={}'
    MarketPriceParamsVec=gpuArray([]);
else
    MarketPriceParamsVec=gpuArray(CreateVectorFromParams(Parameters,MarketClearanceParamNames));
end

MarketClearanceVec=ones(1,length(MarketClearanceEqns),'gpuArray')*Inf;
for i=1:length(MarketClearanceEqns)
    MarketClearanceVec(i)=MarketClearanceEqns{i}(SSvalues_AggVars, p, MarketPriceParamsVec);
%     MarketClearanceVec(i)=p(i)-MarketPriceEqns{i}(SSvalues_AggVars, p, MarketPriceParamsVec);
end


end
