function  [MarketClearanceVec]=MarketClearance_Case2(SSvalues_AggVars,p, MarketClearanceEqns, Parameters, MarketClearanceParamNames)
% This code is actually identical to MarketClearance_Case1 anyway

MarketClearanceVec=ones(1,length(MarketClearanceEqns),'gpuArray')*Inf;
for i=1:length(MarketClearanceEqns)
    if isempty(MarketClearanceParamNames(i).Names)  % check for 'MarketClearanceParamNames(i).Names={}'
        MarketClearanceParamsVec=gpuArray([]);
        MarketClearanceVec(i)=MarketClearanceEqns{i}(SSvalues_AggVars, p, MarketClearanceParamsVec);
    else
        MarketClearanceParamsVec=gpuArray(CreateVectorFromParams(Parameters,MarketClearanceParamNames(i).Names));
        MarketClearanceParamsCell=cell(length(MarketClearanceParamsVec),1);
        for jj=1:length(MarketClearanceParamsVec)
            MarketClearanceParamsCell(jj,1)={MarketClearanceParamsVec(jj)};
        end

        MarketClearanceVec(i)=MarketClearanceEqns{i}(SSvalues_AggVars, p, MarketClearanceParamsCell{:});
    end
    
end

end
