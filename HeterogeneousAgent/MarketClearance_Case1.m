function  [MarketClearanceVec]=MarketClearance_Case1(SSvalues_AggVars,p, MarketPriceEqns, Parameters, MarketPriceParamNames)
%For models with more than one market MultiMarketCriterion determines which method to use to combine them.

% Includes check for cases in which no parameters are actually required
if (isempty(MarketPriceParamNames) || strcmp(MarketPriceParamNames(1),'')) % check for 'SSvalueParamNames={}'
    MarketPriceParamsVec=gpuArray([]);
else
    MarketPriceParamsVec=gpuArray(CreateVectorFromParams(Parameters,MarketPriceParamNames));
end

MarketClearanceVec=ones(1,length(MarketPriceEqns),'gpuArray')*Inf;
for i=1:length(MarketPriceEqns)
    MarketClearanceVec(i)=p(i)-MarketPriceEqns{i}(SSvalues_AggVars, p, MarketPriceParamsVec);
end


end
