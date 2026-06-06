function [ReturnFn_temp, DiscountFactorParamNames_temp]=PType_setup_ReturnFnDiscountFactor(iistr,ReturnFn,DiscountFactorParamNames)


DiscountFactorParamNames_temp=DiscountFactorParamNames;
if isstruct(DiscountFactorParamNames)
    names=fieldnames(DiscountFactorParamNames);
    for jj=1:length(names)
        if strcmp(names{jj},iistr)
            DiscountFactorParamNames_temp=DiscountFactorParamNames.(names{jj});
        end
    end
end

if isstruct(ReturnFn)
    ReturnFn_temp=ReturnFn.(iistr);
else
    ReturnFn_temp=ReturnFn;
end