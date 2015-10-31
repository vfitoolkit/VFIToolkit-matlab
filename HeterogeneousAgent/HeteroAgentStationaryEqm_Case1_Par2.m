function [p_eqm,p_eqm_index,MarketClearance]=HeteroAgentStationaryEqm_Case1_Par2(V0Kron, n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid, p_grid, beta, ReturnFn, SSvaluesFn, SSvalueParamNames, MarketPriceEqns, MarketPriceParamNames, MultiMarketCriterion, simoptions, vfoptions,Parameters, ReturnFnParamNames, PriceParamNames)

N_d=prod(n_d);
N_a=prod(n_a);
N_s=prod(n_s);
N_p=prod(n_p);

l_p=length(n_p);

%% Check which vfoptions and simoptions have been used, set all others to defaults 
tic()
if nargin<19
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

if nargin<18
    simoptions.iterate=1;
    simoptions.nagents=0;
    simoptions.maxit=5*10^4; %In my experience, after a simulation, if you need more that 5*10^4 iterations to reach the steady-state it is because something has gone wrong
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_s/2)];
    simoptions.simperiods=10^3;
    simoptions.burnin=10^2;
    simoptions.parallel=2;
    simoptions.verbose=0;
    simoptions.ncores=1;
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
        simoptions.simperiods=10^3;
    end
    eval('fieldexists=1;simoptions.burnin;','fieldexists=0;')
    if fieldexists==0
        simoptions.burnin=10^2;
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
end

%% 

MarketClearanceKron=ones(N_p,l_p,'gpuArray');
V0Kron=reshape(V0Kron,[N_a,N_s]);

for p_c=1:N_p
    p_c
    
    %Step 1: Solve the value fn iteration problem (given this price, indexed by p_c)
    %Calculate the price vector associated with p_c
    %p_index=zeros(num_p,1);
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
    [~,Policy]=ValueFnIter_Case1(V0Kron, n_d,n_a,n_s,d_grid,a_grid,s_grid, pi_s, beta, ReturnFn,vfoptions,Parameters,ReturnFnParamNames);

    %Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
    StationaryDistKron=StationaryDist_Case1(Policy,n_d,n_a,n_s,pi_s,simoptions);
    SSvalues_AggVars=SSvalues_AggVars_Case1(StationaryDistKron, Policy, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s,p,2); % The 2 is for Parallel (use GPU)

    % use of real() is a hack that could disguise errors, but I couldn't
    % find why matlab was treating output as complex
    MarketClearanceKron(p_c,:)=real(MarketClearance_Case1(SSvalues_AggVars,p_c,n_p,p_grid, MarketPriceEqns, Parameters,MarketPriceParamNames));
end

if MultiMarketCriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
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