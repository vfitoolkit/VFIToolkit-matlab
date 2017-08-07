function [p_eqm,p_eqm_index,MarketClearance]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist, n_d, n_a, n_s, N_j, n_p, pi_s, d_grid, a_grid, s_grid, ReturnFn, SSvaluesFn, MarketClearanceEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, SSvalueParamNames, MarketClearanceParamNames, PriceParamNames,heteroagentoptions, simoptions, vfoptions)
% If n_p=0 then will use fminsearch to find the general equilibrium (find
% price vector that corresponds to MarketClearance=0). By setting n_p to
% nonzero it is assumed you want to use a grid on prices, which must then
% be passed in using heteroagentoptions.p_grid


N_d=prod(n_d);
N_a=prod(n_a);
N_s=prod(n_s);
N_p=prod(n_p);

l_p=length(n_p);

p_eqm=nan; p_eqm_index=nan; MarketClearance=nan;

%% Check which options have been used, set all others to defaults 
if nargin<22
    vfoptions.parallel=2;
    %If vfoptions is not given, just use all the defaults
    %Note that the defaults will be set when we call 'ValueFnIter...'
    %commands and the like, so no need to set them here except for a few.
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    eval('fieldexists=1;vfoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        vfoptions.parallel=2;
    end
end

if nargin<21
    simoptions.fakeoption=0; % create a 'placeholder' simoptions that can be passed to subcodes
end

if nargin<20
    heteroagentoptions.multimarketcriterion=1;
    heteroagentoptions.fminalgo=1;
    heteroagentoptions.verbose=0;
    heteroagentoptions.maxiter=1000;
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
    eval('fieldexists=1;heteroagentoptions.maxiter;','fieldexists=0;')
    if fieldexists==0
        heteroagentoptions.maxiter=1000; % use fminsearch
    end
end

%%

if N_p~=0
    [p_eqm,p_eqm_index,MarketClearance]=HeteroAgentStationaryEqm_Case1_FHorz_pgrid(jequaloneDist, n_d, n_a, n_s, N_j, n_p, pi_s, d_grid, a_grid, s_grid, ReturnFn, SSvaluesFn, MarketClearanceEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, SSvalueParamNames, MarketClearanceParamNames, PriceParamNames, heteroagentoptions, simoptions, vfoptions);
    return
end

%% Otherwise, use fminsearch to find the general equilibrium

MarketClearanceFn=@(p) HeteroAgentStationaryEqm_Case1_FHorz_subfn(p, jequaloneDist, n_d, n_a, n_s, N_j, n_p, pi_s, d_grid, a_grid, s_grid, ReturnFn, SSvaluesFn, MarketClearanceEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, SSvalueParamNames, MarketClearanceParamNames, PriceParamNames, heteroagentoptions, simoptions, vfoptions)

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