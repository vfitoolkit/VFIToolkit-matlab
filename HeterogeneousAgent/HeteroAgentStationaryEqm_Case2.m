function [p_eqm,p_eqm_index,MarketClearance]=HeteroAgentStationaryEqm_Case2(V0, n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid,Phi_aprimeKron, Case2_Type, ReturnFn, SSvaluesFn, MarketPriceEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, SSvalueParamNames, MarketPriceParamNames, PriceParamNames,heteroagentoptions, simoptions, vfoptions)
% If n_p=0 then will use fminsearch to find the general equilibrium (find
% price vector that corresponds to MarketClearance=0). By setting n_p to
% nonzero it is assumend you want to use a grid on prices, which must then
% be passed in heteroagentoptions.p_grid

N_d=prod(n_d);
N_a=prod(n_a);
N_s=prod(n_s);
N_p=prod(n_p);

l_p=length(n_p);

%% Check which simoptions and vfoptions have been used, set all others to defaults 
if nargin<23
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
        vfoptions.parallel=2;
    end
    eval('fieldexists=1;vfoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        vfoptions.verbose=0;
    end
end

if nargin<22
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_s/2)];
    simoptions.simperiods=10^4;
    simoptions.burnin=10^3;
    simoptions.parallel=2;
    simoptions.verbose=0;
    simoptions.iterate=1;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
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
    eval('fieldexists=1;simoptions.iterate;','fieldexists=0;')
    if fieldexists==0
        simoptions.iterate=1;
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

if nargin<21
    heteroagentoptions.multimarketcritereon=1;
    heteroagentoptions.verbose=0;
else
    eval('fieldexists=1;heteroagentoptions.multimarketcriterion;','fieldexists=0;')
    if fieldexists==0
        heteroagentoptions.multimarketcriterion=1;
    end
    if N_p~=0
        eval('fieldexists=1;heteroagentoptions.pgrid;','fieldexists=0;')
        if fieldexists==0
            disp('VFI Toolkit ERROR: you have set n_p to a non-zero value, but not declared heteroagentoptions.pgrid')
        end
    end
    eval('fieldexists=1;heteroagentoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        heteroagentoptions.verbose=0;
    end
    eval('fieldexists=1;heteroagentoptions.fminalgo;','fieldexists=0;')
    if fieldexists==0
        heteroagentoptions.fminalgo=1; % use fminsearch
    end
end

%%

zerosinphi_aprimekron=sum(sum(sum(sum(Phi_aprimeKron==0))));
disp(sprintf('If this number is not zero there is an problem with Phi_aprimeKron: %.0f', zerosinphi_aprimekron))

%%
V0Kron=reshape(V0,[N_a,N_s]);


if N_p~=0
    [p_eqm,p_eqm_index,MarketClearance]=HeteroAgentStationaryEqm_Case2_pgrid(V0Kron, n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid, Phi_aprimeKron, Case2_Type, ReturnFn, SSvaluesFn, MarketPriceEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, SSvalueParamNames, MarketPriceParamNames, PriceParamNames,heteroagentoptions, simoptions, vfoptions);
    return
end

%%  Otherwise, use fminsearch to find the general equilibrium
% I SHOULD IMPLEMENT A BETTER V0Kron HERE
MarketClearanceFn=@(p) HeteroAgentStationaryEqm_Case2_subfn(p, V0Kron, n_d, n_a, n_s, n_p, pi_s, d_grid, a_grid, s_grid, Phi_aprimeKron, Case2_Type, ReturnFn, SSvaluesFn, MarketPriceEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, SSvalueParamNames, MarketPriceParamNames, PriceParamNames,heteroagentoptions, simoptions, vfoptions)

p0=nan(length(PriceParamNames),1);
for ii=1:l_p
    p0(ii)=Parameters.(PriceParamNames{ii});
end

if heteroagentoptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
    heteroagentoptions.multimarketcriterion=0;
    [p_eqm,MarketClearance]=fzero(MarketClearanceFn,p0);    
elseif heteroagentoptions.fminalgo==1
    [p_eqm,MarketClearance]=fminsearch(MarketClearanceFn,p0);
else
    [p_eqm,MarketClearance]=fminsearch(MarketClearanceFn,p0);
end

p_eqm_index=nan; % If not using p_grid then this is irrelevant/useless


end