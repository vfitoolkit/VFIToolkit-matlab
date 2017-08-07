function [p_eqm,p_eqm_index,MarketClearance]=HeteroAgentStationaryEqm_Case2_FHorz_pgrid(jequaloneDist,AgeWeights,n_d, n_a, n_s, N_j, n_p, pi_s, d_grid, a_grid, s_grid, Phi_aprimeFn, Case2_Type, ReturnFn, SSvaluesFn, MarketClearanceEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, SSvalueParamNames, MarketClearanceParamNames, PriceParamNames,heteroagentoptions, simoptions, vfoptions)

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
%V0Kron=reshape(V0,[N_a,N_s]);

for p_c=1:N_p
    if heteroagentoptions.verbose==1
        sprintf('Evaluating price vector %i of %i',p_c,N_p)
    end
    
    %Step 1: Solve the value fn iteration problem (given this price, indexed by p_c)
    %Calculate the price vector associated with p_c
    p_index=ind2sub_homemade(n_p,p_c);
    p=nan(l_p,1);
    for ii=1:l_p
        if ii==1
            p(ii)=p_grid(p_index(1));
        else
            p(ii)=p_grid(sum(n_p(1:ii-1))+p_index(ii));
        end
        Parameters.(PriceParamNames{ii})=gather(p(ii));
    end
    
    [~, Policy]=ValueFnIter_Case2_FHorz(n_d,n_a,n_s,N_j,d_grid, a_grid, s_grid, pi_s, Phi_aprimeFn, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions);
    
    %Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
    StationaryDist=StationaryDist_FHorz_Case2(jequaloneDist,AgeWeights,Policy,n_d,n_a,n_s,N_j,d_grid, a_grid, s_grid,pi_s,Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames,simoptions);
%     StationaryDistKron=StationaryDist_FHorz_Case2(jequaloneDist,Policy,n_d,n_a,n_s,N_j,d_grid, a_grid, s_grid,pi_s,Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames,simoptions);
    %SSvalues_AggVars=SSvalues_AggVars_Case2(StationaryDistKron, Policy, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s,p,2); % The 2 is for Parallel (use GPU)
    SSvalues_AggVars=SSvalues_AggVars_FHorz_Case2(StationaryDist, Policy, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_s,N_j, d_grid, a_grid, s_grid, 2); % The 2 is for Parallel (use GPU)

    % The following line is often a useful double-check if something is going wrong.
%    SSvalues_AggVars
    
    % use of real() is a hack that could disguise errors, but I couldn't
    % find why matlab was treating output as complex
    MarketClearanceKron(p_c,:)=real(MarketClearance_Case2(SSvalues_AggVars,p, MarketClearanceEqns, Parameters,MarketClearanceParamNames));
end


if heteroagentoptions.multimarketcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(MarketClearanceKron.^2,2));                                                                                                         
end

p_eqm_index=ind2sub_homemade_gpu(n_p,p_eqm_indexKron);
if l_p>1
    MarketClearance=nan(N_p,1+l_p,'gpuArray');
    if heteroagentoptions.multimarketcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
        MarketClearance(:,1)=sum(MarketClearanceKron.^2,2);
    end
    MarketClearance(:,2:end)=MarketClearanceKron;
    MarketClearance=reshape(MarketClearance,[n_p,1+l_p]);
else
    MarketClearance=reshape(MarketClearanceKron,[n_p,1]);
end

%Calculate the price associated with p_eqm_index
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