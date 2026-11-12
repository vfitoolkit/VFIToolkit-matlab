function [PricePath,GEcondnPath]=TransitionPath_MixHorz_PType_Anderson(PricePath0, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, FullFnsToEvaluate, GeneralEqmEqns, PricePathSizeVec, ParamPathSizeVec, PricePathSizeVec_ii, ParamPathSizeVec_ii, GEeqnNames,nGeneralEqmEqns,nGeneralEqmEqns_acrossptypes,GeneralEqmEqnsCell,GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, transpathoptions, PTypeStructure)
% Solve the mixed-horizon PType transition path using Anderson Acceleration
% (transpathoptions.GEnewprice=2).
%
% Everything Anderson-specific lives in AndersonAcceleration(), which is shared
% with the stationary general eqm solver (heteroagentoptions.fminalgo=9) and
% with TransitionPath_FHorz_Anderson / TransitionPath_InfHorz_Anderson.

N_i=PTypeStructure.N_i;
nPrices=size(PricePath0,2);

%% Pre-allocate Step0_setup once across all PTypes so single-pass calls reuse buffers
for ii=1:N_i
    iistr=PTypeStructure.Names_i{ii};
    if isfinite(PTypeStructure.(iistr).N_j)
        [PTypeStructure.(iistr).PolicyIndexesPath,PTypeStructure.(iistr).N_probs,PTypeStructure.(iistr).II1,PTypeStructure.(iistr).II2,PTypeStructure.(iistr).exceptlastj,PTypeStructure.(iistr).exceptfirstj,PTypeStructure.(iistr).justfirstj]=TransitionPath_FHorz_substeps_Step0_setup(PTypeStructure.(iistr).l_d,PTypeStructure.(iistr).l_aprime,PTypeStructure.(iistr).N_a,0,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,PTypeStructure.(iistr).N_j,T,transpathoptions,PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).simoptions);
    else
        [PTypeStructure.(iistr).PolicyIndexesPath,PTypeStructure.(iistr).N_probs,PTypeStructure.(iistr).II1,PTypeStructure.(iistr).II2]=TransitionPath_InfHorz_substeps_Step0_setup(PTypeStructure.(iistr).l_d,PTypeStructure.(iistr).l_aprime,PTypeStructure.(iistr).N_a,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,T,transpathoptions,PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).simoptions);
    end
end

%% Single-pass options for TransitionPath_MixHorz_PType_shooting
% setupGEnewprice3_shooting wrote the parsed howtoupdate struct into GEnewprice2;
% copy it to GEnewprice3 so _shooting and updatePricePath work unchanged.
transpathoptions_single=transpathoptions;
transpathoptions_single.GEnewprice=3;
transpathoptions_single.GEnewprice3=transpathoptions.GEnewprice2;
if ~isfield(transpathoptions_single,'oldpathweight')
    transpathoptions_single.oldpathweight=zeros(size(transpathoptions_single.GEnewprice3.factor));
end
transpathoptions_single.maxiter=1;
transpathoptions_single.toleranceGEprices=Inf;
transpathoptions_single.toleranceGEcondns=Inf;
transpathoptions_single.verbose=0;
transpathoptions_single.graphpricepath=0;
transpathoptions_single.graphaggvarspath=0;
transpathoptions_single.graphGEcondns=0;
transpathoptions_single.historyofpricepath=0;

%% Cache of the most recent single-pass evaluation so ShootingMapFn does not re-solve the model
last_p=[];
last_PricePathNew=[];

    function GEc_vec=eval_single_pass(p_vec)
        PricePathTrial=reshape(p_vec,[T,nPrices]);
        [~,GEc_mat]=TransitionPath_MixHorz_PType_shooting( ...
            PricePathTrial, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, ...
            FullFnsToEvaluate, GeneralEqmEqns, PricePathSizeVec, ParamPathSizeVec, PricePathSizeVec_ii, ParamPathSizeVec_ii, ...
            GEeqnNames, nGeneralEqmEqns, nGeneralEqmEqns_acrossptypes, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, ...
            use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, ...
            tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, ...
            tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, ...
            transpathoptions_single, PTypeStructure);
        GEc_mat(T,:)=0; % Period T is pinned; zero its residual so Anderson never moves it
        GEc_vec=GEc_mat(:);
    end

    function g_vec=eval_shooting_map(p_vec,GEc_vec)
        PricePathCurr=reshape(p_vec,[T,nPrices]);
        G_mat=TransitionPath_FromGEcondns_shooting(PricePathCurr,reshape(GEc_vec,[T,nGeneralEqmEqns_acrossptypes]),PricePathSizeVec,transpathoptions_single,T);
        G_mat(T,:)=PricePath0(T,:); % Keep row T pinned to final steady state
        g_vec=G_mat(:);
    end

GEcondnsFn=@(p) eval_single_pass(p);
ShootingMapFn=@(p,GEc) eval_shooting_map(p,GEc);

if transpathoptions.multiGEcriterion==0
    DistanceFn=@(GEc) max(sum(abs(transpathoptions.multiGEweights.*reshape(GEc(1:(T-1)*nGeneralEqmEqns_acrossptypes),[T-1,nGeneralEqmEqns_acrossptypes])),2));
elseif transpathoptions.multiGEcriterion==1
    DistanceFn=@(GEc) max(sqrt(sum(transpathoptions.multiGEweights.*(reshape(GEc(1:(T-1)*nGeneralEqmEqns_acrossptypes),[T-1,nGeneralEqmEqns_acrossptypes]).^2),2)));
end

%% Run Anderson Acceleration
p0=PricePath0(:);
[p_star,GEc_star]=AndersonAcceleration(GEcondnsFn,ShootingMapFn,DistanceFn,p0,transpathoptions.toleranceGEcondns,transpathoptions.anderson);

PricePath=reshape(p_star,[T,nPrices]);
GEcondnPath=reshape(GEc_star,[T,nGeneralEqmEqns_acrossptypes]);


end
