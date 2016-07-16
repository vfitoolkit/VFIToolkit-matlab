function [p_eqm,p_eqm_index,MarketClearance]=HeteroAgentStationaryEqm_Case1_pgrid(V0Kron, n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid, ReturnFn, SSvaluesFn, MarketPriceEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, SSvalueParamNames, MarketPriceParamNames, PriceParamNames,heteroagentoptions, simoptions, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_s=prod(n_s);
N_p=prod(n_p);

l_p=length(n_p);

p_grid=heteroagentoptions.pgrid;

%% 

if vfoptions.parallel==2
    MarketClearanceKron=ones(N_p,l_p,'gpuArray');
else
    MarketClearanceKron=ones(N_p,l_p);
end
% V0Kron=reshape(V0Kron,[N_a,N_s]);

for p_c=1:N_p
    if heteroagentoptions.verbose==1
        p_c
    end
    
    V0Kron(~isfinite(V0Kron))=0; %Since we loop through with V0Kron from previous p_c this is necessary to avoid contamination by -Inf's
    
    %Step 1: Solve the value fn iteration problem (given this price, indexed by p_c)
    %Calculate the price vector associated with p_c
    p_index=ind2sub_homemade(n_p,p_c);
    p=zeros(l_p,1);
    for ii=1:l_p
        if ii==1
            p(ii)=p_grid(p_index(1));
        else
            p(ii)=p_grid(sum(n_p(1:ii-1))+p_index(ii));
        end
        Parameters.(PriceParamNames{ii})=p(ii);
    end
    
    %     ReturnFnParams(IndexesForPricesInReturnFnParams)=p;
    [~,Policy]=ValueFnIter_Case1(V0Kron, n_d,n_a,n_s,d_grid,a_grid,s_grid, pi_s, DiscountFactorParamNames, ReturnFn,vfoptions,Parameters,ReturnFnParamNames);

    %Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
    StationaryDistKron=StationaryDist_Case1(Policy,n_d,n_a,n_s,pi_s,simoptions);
    SSvalues_AggVars=SSvalues_AggVars_Case1(StationaryDistKron, Policy, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s,p,2); % The 2 is for Parallel (use GPU)
    
    % The following line is often a useful double-check if something is going wrong.
%    SSvalues_AggVars
    
    % use of real() is a hack that could disguise errors, but I couldn't
    % find why matlab was treating output as complex
%     MarketClearanceKron(p_c,:)=real(MarketClearance_Case1_pgrid(SSvalues_AggVars,p_c,n_p,p_grid, MarketPriceEqns, Parameters,MarketPriceParamNames));
    MarketClearanceKron(p_c,:)=real(MarketClearance_Case1(SSvalues_AggVars,p, MarketPriceEqns, Parameters,MarketPriceParamNames));
end

if heteroagentoptions.multimarketcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(MarketClearanceKron.^2,2));                                                                                                         
end

%p_eqm_index=zeros(num_p,1);
p_eqm_index=ind2sub_homemade_gpu(n_p,p_eqm_indexKron);
MarketClearance=reshape(MarketClearanceKron,[n_p,1]);

%Calculate the price associated with p_eqm_index
l_p=length(n_p);
p_eqm=zeros(l_p,1);
for ii=1:l_p
    if ii==1
        p_eqm(ii)=p_grid(p_eqm_index(1));
    else
        p_eqm(ii)=p_grid(sum(n_p(1:ii-1))+p_eqm_index(ii));
    end
end

% Move results from gpu to cpu before returning them
p_eqm=gather(p_eqm);
p_eqm_index=gather(p_eqm_index);
MarketClearance=gather(MarketClearance);

end