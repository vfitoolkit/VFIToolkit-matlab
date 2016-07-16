function  [MarketClearanceVec]=MarketClearance_Case2(SSvalues_AggVars,p, MarketPriceEqns, Parameters, MarketPriceParamNames)
% This code is actually identical to MarketClearance_Case1 anyway

MarketPriceParamsVec=gpuArray(CreateVectorFromParams(Parameters,MarketPriceParamNames));

MarketClearanceVec=ones(1,length(MarketPriceEqns),'gpuArray')*Inf;
for i=1:length(MarketPriceEqns)
    MarketClearanceVec(i)=p(i)-MarketPriceEqns{i}(SSvalues_AggVars, p, MarketPriceParamsVec);
end

end
