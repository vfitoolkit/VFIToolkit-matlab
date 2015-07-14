function  [MarketClearanceVec]=MarketClearance_Case2(SSvalues_AggVars,p_c,n_p,p_grid, MarketPriceEqns, MarketPriceParams)
%For models with more than one market MultiMarketCriterion determines which method to use to combine them.


MarketClearanceVec=ones(1,length(MarketPriceEqns))*Inf;
p_sub=ind2sub_homemade(n_p,p_c);
num_p_vars=length(n_p);
p=zeros(num_p_vars,1);
for i=1:num_p_vars
    if i==1
        p(i)=p_grid(p_sub(1));
    else
        p(i)=p_grid(sum(n_p(1:i-1))+p_sub(i));
    end
end
for i=1:length(MarketPriceEqns)
    MarketClearanceVec(i)=p(i)-MarketPriceEqns{i}(SSvalues_AggVars, p, MarketPriceParams);
end



end
