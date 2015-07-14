function [p_eqm,p_eqm_index,MarketClearance]=HeteroAgentStationaryEqm_Case2_Par2(V0Kron, n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid, p_grid,Phi_aprimeKron, Case2_Type, beta, ReturnFn, SSvaluesFn, MarketPriceEqns, MarketPriceParams, MultiMarketCriterion, vfoptions, simoptions,ReturnFnParams, IndexesForPricesInReturnFnParams)
                                                                               % (V0Kron, n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid, p_grid, beta, ReturnFn, SSvaluesFn, SSvalueParams, MarketPriceEqns, MarketPriceParams, MultiMarketCriterion, simoptions, vfoptions,ReturnFnParams, IndexesForPricesInReturnFnParams)

%Things you may want to try adjusting for speed
% In 'SteadyState_Case2_Simulation_raw' you might want to adjust; nsims, periods, burnin

N_d=prod(n_d);
N_a=prod(n_a);
N_s=prod(n_s);
N_p=prod(n_p);
l_p=length(n_p);

%% Check which simoptions and vfoptions have been used, set all others to defaults 
if nargin<21
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_s/2)];
    simoptions.simperiods=10^4;
    simoptions.burnin=10^3;
    simoptions.parallel=2;
    simoptions.verbose=0;
    simoptions.ncores=1;
    simoptions.ssfull=0;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;simoptions.seedpoint;','fieldexists=0;')
    if fieldexists==0
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_s/2)];
    end
    eval('fieldexists=1;simoptions.simperiods;','fieldexists=0;')
    if fieldexists==0
        simoptions.simperiods=10^4;
    end
    eval('fieldexists=1;simoptions.burnin;','fieldexists=0;')
    if fieldexists==0
        simoptions.burnin=10^3;
    end
    eval('fieldexists=1;simoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        simoptions.parallel=2;
    end
    eval('fieldexists=1;simoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        simoptions.verbose=0;
    end
    eval('fieldexists=1;simoptions.ncores;','fieldexists=0;')
    if fieldexists==0
        simoptions.ncores=1;
    end
    eval('fieldexists=1;simoptions.ssfull;','fieldexists=0;')
    if fieldexists==0
        simoptions.ssfull=0;
    end
end

if nargin<20
    %If vfoptions is not given, just use all the defaults
    vfoptions.lowmemory=0;
    vfoptions.polindorval=1;
    vfoptions.howards=80;
    vfoptions.parallel=2;
    vfoptions.verbose=0;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;vfoptions.lowmemory;','fieldexists=0;')
    if fieldexists==0
        vfoptions.lowmemory=0;
    end
    eval('fieldexists=1;vfoptions.polindorval;','fieldexists=0;')
    if fieldexists==0
        vfoptions.polindorval=1;
    end
    eval('fieldexists=1;vfoptions.howards;','fieldexists=0;')
    if fieldexists==0
        vfoptions.howards=80;
    end
    eval('fieldexists=1;vfoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        vfoptions.parallel=0;
    end
    eval('fieldexists=1;vfoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        vfoptions.verbose=0;
    end
end

%%

zerosinphi_aprimekron=sum(sum(sum(sum(Phi_aprimeKron==0))));
fprintf('If this number is not zero there is an problem with Phi_aprimeKron: %.0f \n', zerosinphi_aprimekron)

MarketClearanceKron=ones(N_p,l_p,'gpuArray');
V0Kron=reshape(V0Kron,[N_a,N_s]);

disp('Running through possible prices for general eqm')

for p_c=1:N_p
    p_c
    
    V0Kron(~isfinite(V0Kron))=0; %Since we loop through with V0Kron from previous p_c this is necessary to avoid contamination by -Inf's
%     %Calculate the price vector associated with p_c
%     p_index=ind2sub_homemade(n_p,p_c);
%     p=cell(l_p,1);
%     for i=1:l_p
%         if i==1
%             p{i}=p_grid(p_index(1));
%         else
%             p{i}=p_grid(sum(n_p(1:i-1))+p_index(i));
%         end
%     end
%     %p_c
    %Step 1: Solve the value fn iteration problem (given this price, indexed by p_c)
    %Calculate the price vector associated with p_c
    %p_index=zeros(num_p,1);
    p_index=ind2sub_homemade(n_p,p_c);
    p=zeros(l_p,1);
    for i=1:l_p
        if i==1
            p(i)=p_grid(p_index(1));
        else
            p(i)=p_grid(sum(n_p(1:i-1))+p_index(i));
        end
    end

    %tic;
    
%     if vfoptions.returnmatrix==1
%         ReturnMatrix=ReturnFnHeteroAgent(p{:});
%     else
%         ReturnFn=@(d,a,s) ReturnFnHeteroAgent(d,a,s,p{:});
%         ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a, n_s, d_grid, a_grid, s_grid, vfoptions.parallel);
%     end
    ReturnFnParams(IndexesForPricesInReturnFnParams)=p;
    [~,Policy]=ValueFnIter_Case2(V0Kron, n_d,n_a,n_s,d_grid,a_grid,s_grid, pi_s, beta, ReturnFn,Phi_aprimeKron, Case2_Type, vfoptions,ReturnFnParams);
    
%     if vfoptions.parallel==0
%         [VKron, PolicyIndexesKron]=ValueFnIter_Case2_raw(Tolerance, V0Kron, n_d,n_a,n_s, pi_s, beta, ReturnMatrix,Phi_aprimeKron,Case2_Type,vfoptions.howards,vfoptions.verbose); 
%     elseif vfoptions.parallel==1
%         [VKron, PolicyIndexesKron]=ValueFnIter_Case2_Par1_raw(Tolerance, V0Kron, n_d,n_a,n_s, pi_s, beta, ReturnMatrix,Phi_aprimeKron,Case2_Type,vfoptions.howards,vfoptions.verbose);
%     end
    
%    [V0Kron,PolicyIndexesKron]=ValueFnIter_Case2_raw(Tolerance, V0Kron, N_d, N_a, N_s, pi_s, Phi_aprimeKron, Case2_Type, beta, FmatrixKron, Howards);
   
    %Step 2: Calculate the Steady-state distn (given this price)
    %and use it to assess market clearance
%     SteadyStateDistKron=SteadyState_Case2_Simulation_raw(PolicyIndexesKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_s,pi_s, simoptions);
%    SteadyStateDistKron=SteadyState_Case2_Simulation_raw(SimPeriods, ceil(SimPeriods/100), NumOfSeedPoint, PolicyIndexesKron, Phi_aprimeKron,Case2_Type,N_d,N_a,N_s,pi_s);
    SteadyStateDist=SteadyState_Case2_Simulation(Policy,Phi_aprimeKron,Case2_Type,n_d,n_a,n_s,pi_s, simoptions);
    if simoptions.ssfull==1
%         SteadyStateDistKron=SteadyState_Case2_raw(SteadyStateDistKron,Tolerance,PolicyIndexesKron,Phi_aprimeKron, Case2_Type, N_d,N_a,N_s,pi_s,0);
        SteadyStateDist=SteadyState_Case2(SteadyStateDist,Policy,Phi_aprimeKron,Case2_Type,n_d,n_a,n_s,pi_s,simoptions);
    end
    SSvalues_AggVars=SSvalues_AggVars_Case2(SteadyStateDist, Policy, SSvaluesFn, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s,p);
%     SSvalues_AggVars=SSvalues_AggVars_Case2_raw(SteadyStateDistKron, PolicyIndexesKron, SSvaluesFn, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s,p);

    MarketClearanceKron(p_c,:)=MarketClearance_Case2(SSvalues_AggVars,p_c,n_p,p_grid, MarketPriceEqns, MarketPriceParams);
    
    %toc
end


if MultiMarketCriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [~,p_eqm_indexKron]=min(sum(MarketClearanceKron.^2,2)); 
end

%[trash,p_eqm_indexKron]=min(MarketClearanceKron); 
%p_eqm_index=zeros(num_p_vars,1);
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

end