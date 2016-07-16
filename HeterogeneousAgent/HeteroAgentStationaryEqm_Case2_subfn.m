function MarketClearance=HeteroAgentStationaryEqm_Case2_subfn(p, V0Kron, n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid, Phi_aprimeKron, Case2_Type, ReturnFn, SSvaluesFn, MarketPriceEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, SSvalueParamNames, MarketPriceParamNames, PriceParamNames,heteroagentoptions, simoptions, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_s=prod(n_s);
N_p=prod(n_p);

l_p=length(n_p);

%% 
for ii=1:l_p
    Parameters.(PriceParamNames{ii})=p(ii);
end

%     ReturnFnParams(IndexesForPricesInReturnFnParams)=p;
[~,Policy]=ValueFnIter_Case2(V0Kron, n_d,n_a,n_s,d_grid,a_grid,s_grid, pi_s, Phi_aprimeKron, Case2_Type, DiscountFactorParamNames, ReturnFn,vfoptions,Parameters,ReturnFnParamNames);

%Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
StationaryDistKron=StationaryDist_Case2(Policy,Phi_aprimeKron,Case2_Type,n_d,n_a,n_s,pi_s,simoptions);
SSvalues_AggVars=SSvalues_AggVars_Case2(StationaryDistKron, Policy, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s,p,2); % The 2 is for Parallel (use GPU) % The 2 is for Parallel (use GPU)

% The following line is often a useful double-check if something is going wrong.
%    SSvalues_AggVars

% use of real() is a hack that could disguise errors, but I couldn't
% find why matlab was treating output as complex
MarketClearanceVec=real(MarketClearance_Case2(SSvalues_AggVars,p, MarketPriceEqns, Parameters,MarketPriceParamNames));

if heteroagentoptions.multimarketcriterion==0 %only used when there is only one price 
    MarketClearance=MarketClearanceVec;
elseif heteroagentoptions.multimarketcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    MarketClearance=sum(MarketClearanceVec.^2);                                                                                                         
end

MarketClearance=gather(MarketClearance);

if heteroagentoptions.verbose==1
    [p,MarketClearance]
end

end