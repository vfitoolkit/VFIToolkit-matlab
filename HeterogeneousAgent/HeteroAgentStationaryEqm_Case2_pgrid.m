function [p_eqm,p_eqm_index,MarketClearance]=HeteroAgentStationaryEqm_Case2_pgrid(V0Kron, n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid, Phi_aprimeKron, Case2_Type, ReturnFn, SSvaluesFn, MarketPriceEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, SSvalueParamNames, MarketPriceParamNames, PriceParamNames,heteroagentoptions, simoptions, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_s=prod(n_s);
N_p=prod(n_p);

l_p=length(n_p);

MarketClearanceKron=ones(N_p,l_p);
%V0Kron=reshape(V0,[N_a,N_s]);

disp('Running through possible prices for general eqm')

for p_c=1:N_p
    if heteroagentoptions.verbose==1
        p_c
    end
    
    V0Kron(~isfinite(V0Kron))=0; %Since we loop through with V0Kron from previous p_c this is necessary to avoid contamination by -Inf's

    %Calculate the price vector associated with p_c
    p_index=ind2sub_homemade(n_p,p_c);
    p=cell(l_p,1);
    for i=1:l_p
        if i==1
            p{i}=p_grid(p_index(1));
        else
            p{i}=p_grid(sum(n_p(1:i-1))+p_index(i));
        end
    end
    
        %     ReturnFnParams(IndexesForPricesInReturnFnParams)=p;
%     [~,Policy]=ValueFnIter_Case1(V0Kron, n_d,n_a,n_s,d_grid,a_grid,s_grid, pi_s, DiscountFactorParamNames, ReturnFn,vfoptions,Parameters,ReturnFnParamNames);
    [~, Policy]=ValueFnIter_Case2(V0, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s, Phi_aprimeKron, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    
%     if vfoptions.returnmatrix==1
%         ReturnMatrix=ReturnFnHeteroAgent(p{:});
%     else
%         ReturnFn=@(d,a,s) ReturnFnHeteroAgent(d,a,s,p{:});
%         ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a, n_s, d_grid, a_grid, s_grid, vfoptions.parallel);
%     end
%     
%     if vfoptions.parallel==0
%         [VKron, PolicyIndexesKron]=ValueFnIter_Case2_raw(V0Kron, n_d,n_a,n_s, pi_s, beta, ReturnMatrix,Phi_aprimeKron,Case2_Type,vfoptions.howards,vfoptions.verbose); 
%     elseif vfoptions.parallel==1
%         [VKron, PolicyIndexesKron]=ValueFnIter_Case2_Par1_raw(V0Kron, n_d,n_a,n_s, pi_s, beta, ReturnMatrix,Phi_aprimeKron,Case2_Type,vfoptions.howards,vfoptions.verbose);
%     end
    
%    [V0Kron,PolicyIndexesKron]=ValueFnIter_Case2_raw(Tolerance, V0Kron, N_d, N_a, N_s, pi_s, Phi_aprimeKron, Case2_Type, beta, FmatrixKron, Howards);

    %Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
    StationaryDistKron=StationaryDist_Case2(Policy,Phi_aprimeKron,Case2_Type,n_d,n_a,n_s,pi_s,simoptions);
    SSvalues_AggVars=SSvalues_AggVars_Case2(StationaryDistKron, Policy, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s,p,2); % The 2 is for Parallel (use GPU)
    
    % The following line is often a useful double-check if something is going wrong.
%    SSvalues_AggVars
    
    % use of real() is a hack that could disguise errors, but I couldn't
    % find why matlab was treating output as complex
%     MarketClearanceKron(p_c,:)=MarketClearance_Case2(SSvalues_AggVars,p_c,n_p,p_grid, MarketPriceEqns, MarketPriceParams);
    MarketClearanceKron(p_c,:)=real(MarketClearance_Case2(SSvalues_AggVars,p, MarketPriceEqns, Parameters,MarketPriceParamNames));
    
% %Step 2: Calculate the Steady-state distn (given this price)
%     %and use it to assess market clearance
%     StationaryDistKron=StationaryDist_Case2_Simulation_raw(PolicyIndexesKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_s,pi_s, simoptions);
%     if simoptions.iterate==1
%         StationaryDistKron=StationaryDist_Case2_Iteration_raw(StationaryDistKron, PolicyIndexesKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_s,pi_s, simoptions);
%     end
%     SSvalues_AggVars=SSvalues_AggVars_Case2_raw(StationaryDistKron, PolicyIndexesKron, SSvaluesFn, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s,p);
%     MarketClearanceKron(p_c,:)=MarketClearance_Case2(SSvalues_AggVars,p_c,n_p,p_grid, MarketPriceEqns, MarketPriceParams);
    
    %toc
end


if heteroagentoptions.multimarketcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(MarketClearanceKron.^2,2));                                                                                                         
end

%p_eqm_index=zeros(num_p,1);
p_eqm_index=ind2sub_homemade(n_p,p_eqm_indexKron);
MarketClearance=reshape(MarketClearanceKron,[n_p,1]);


%Calculate the price associated with p_eqm_index
l_p=length(n_p);
p_eqm=zeros(l_p,1);
for i=1:l_p
    if i==1
        p_eqm(i)=p_grid(p_eqm_index(1));
    else
        p_eqm(i)=p_grid(sum(n_p(1:i-1))+p_eqm_index(i));
    end
end

% Move results from gpu to cpu before returning them
p_eqm=gather(p_eqm);
p_eqm_index=gather(p_eqm_index);
MarketClearance=gather(MarketClearance);

end