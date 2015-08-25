function [p_eqm,p_eqm_index,MarketClearance]=HeteroAgentStationaryEqm_Case1(V0, n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid, p_grid, beta, ReturnFn_p, SSvaluesFn, MarketPriceEqns, MarketPriceParams, MultiMarketCriterion, simoptions, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_s=prod(n_s);
N_p=prod(n_p);

l_p=length(n_p);

%% Check which vfoptions and simoptions have been used, set all others to defaults 
if nargin<18
    %If vfoptions is not given, just use all the defaults
    vfoptions.lowmemory=0;
    vfoptions.polindorval=1;
    vfoptions.howards=80;
    vfoptions.parallel=2;
    vfoptions.verbose=0;
    vfoptions.returnmatrix=2;
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
        vfoptions.parallel=2;
    end
    eval('fieldexists=1;vfoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        vfoptions.verbose=0;
    end
    eval('fieldexists=1;vfoptions.returnmatrix;','fieldexists=0;')
    if fieldexists==0
        vfoptions.returnmatrix=2;
    end
end

if nargin<17
    simoptions.iterate=1;
    simoptions.nagents=0;
    simoptions.maxit=5*10^4; %In my experience, after a simulation, if you need more that 5*10^4 iterations to reach the steady-state it is because something has gone wrong
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_s/2)];
    simoptions.simperiods=10^4;
    simoptions.burnin=10^3;
    simoptions.parallel=2;
    simoptions.verbose=0;
    try
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
else
    eval('fieldexists=1;simoptions.iterate;','fieldexists=0;')
    if fieldexists==0
        simoptions.iterate=1;
    end
    eval('fieldexists=1;simoptions.nagents;','fieldexists=0;')
    if fieldexists==0
        simoptions.nagents=0;
    end
    eval('fieldexists=1;simoptions.maxit;','fieldexists=0;')
    if fieldexists==0
        simoptions.maxit=5*10^4;
    end
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
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
end

%% 

MarketClearanceKron=ones(N_p,l_p);
V0Kron=reshape(V0,[N_a,N_s]);

tic;
for p_c=1:N_p
    p_c
    
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
    tic()
    if n_d(1)==0
        ReturnFn=@(aprime_val, a_val, s_val) ReturnFn_p(p, aprime_val, a_val, s_val);
    else
        ReturnFn=@(d_val, aprime_val, a_val, s_val) ReturnFn_p(p,d_val, aprime_val, a_val, s_val);
    end
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_s, d_grid, a_grid, s_grid, vfoptions.parallel);
    time_returnfn=toc();
    %Since the next choice of prices (p_index) is likely to be very close
    %to the current one, and since the methods are in any case universally
    %convergent we update V0Kron.

    
    %Notice that we return V0Kron from the value fn, so the starting point
    %for the value fn next p_c is this times value fn.
    tic()
    if n_d(1)==0
        if vfoptions.parallel==0
            [V0Kron,Policy]=ValueFnIter_Case1_NoD_raw(Tolerance, V0Kron, n_a, n_s, pi_s, beta, ReturnMatrix, vfoptions.howards, vfoptions.verbose);
        elseif vfoptions.parallel==1
            [V0Kron,Policy]=ValueFnIter_Case1_NoD_Par1_raw(Tolerance, V0Kron, n_a, n_s, pi_s, beta, ReturnMatrix, vfoptions.howards, vfoptions.verbose);
        end
    else
        if vfoptions.parallel==0
            [V0Kron, Policy]=ValueFnIter_Case1_raw(Tolerance, V0Kron, n_d,n_a,n_s, pi_s, beta, ReturnMatrix,vfoptions.howards,vfoptions.verbose);
        elseif vfoptions.parallel==1
            [V0Kron, Policy]=ValueFnIter_Case1_Par1_raw(Tolerance, V0Kron, n_d,n_a,n_s, pi_s, beta, ReturnMatrix,vfoptions.howards,vfoptions.verbose);
        end
    end
    time_valuefn=toc();
    
    %Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
    StationaryDistKron=StationaryDist_Case1_Simulation_raw(Policy,N_d,N_a,N_s,pi_s,simoptions);
    if simoptions.iterate==1
        StationaryDistKron=StationaryDist_Case1_Iteration_raw(StationaryDistKron,Policy,N_d,N_a,N_s,pi_s,simoptions);
    end
    tic()
    SSvalues_AggVars=SSvalues_AggVars_Case1_raw(StationaryDistKron, Policy, SSvaluesFn, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s,p);
    time_aggvars=toc();
    tic()
    MarketClearanceKron(p_c,:)=MarketClearance_Case1(SSvalues_AggVars,p_c,n_p,p_grid, MarketPriceEqns, MarketPriceParams);
    time_marketclearance=toc();
    
    disp('Time required')
    [time_returnfn; time_valuefn; time_steadystatesim; time_steadystatessfull; time_aggvars; time_marketclearance]
end

if MultiMarketCriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    [trash,p_eqm_indexKron]=min(sum(MarketClearanceKron.^2,2));                                                                                                         
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



end