function [PricePathOld,GEcondnPath]=TransitionPath_FHorz_PType_Anderson(PricePathOld, PricePathNames, PricePathSizeVec, PricePathSizeVec_ii, ParamPath, ParamPathNames, ParamPathSizeVec, T, FnNames, GEeqnNames, nGeneralEqmEqns_acrossptypes, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, transpathoptions, PTypeStructure)
% Anderson Acceleration of the shooting map on the whole price path, for the transition path with
% permanent types. Selected by transpathoptions.GEnewprice=2. The same as TransitionPath_FHorz_Anderson,
% with TransitionPath_FHorz_PType_singlepathiter as the single path iteration.
%
% The shooting algorithm (GEnewprice=3) is the fixed-point iteration p <-- G(p), where G() applies
% the howtoupdate rule period by period. Anderson mixing combines the last m iterates of that same
% map to take a quasi-Newton-like step, without ever building a Jacobian. So it costs the same one
% path solve per iteration as shooting (plus one more per Anderson step when the safeguard is on).
%
% The iterate is the whole price path flattened to a column, T*nPrices long, where a price that
% depends on ptype has N_i columns. Period T is included even though it never moves:
% updatePricePathNew_TPath_T copies the terminal row through unchanged, so those coordinates are a
% fixed point of G() and contribute nothing to the Anderson history.
% A general eqm condition that depends on ptype (transpathoptions.GEptype) is N_i conditions, so
% there are nGeneralEqmEqns_acrossptypes of them, and setupGEnewprice3_shooting has already expanded
% howtoupdate to match.
%
% Everything Anderson-specific lives in AndersonAcceleration(), which is shared with the stationary
% general eqm solver (heteroagentoptions.fminalgo=9) and with the other transition path Anderson commands.

nPrices=size(PricePathOld,2);
nGEcondns=nGeneralEqmEqns_acrossptypes;

% Anderson builds its step out of the history of iterates, which only means anything if G() is the
% same map at every iteration. That rules out the additional-factor ramp (setupGEnewprice3_shooting
% errors if it is set for GEnewprice=2), and it is why itercounter is held at 1.
itercounter=1;

%% The three things Anderson Acceleration needs: the residual, the map, and the distance
% GEcondnsFn is the expensive one, a full solve of the path (value fn, agent dist, aggregates, for
% every ptype, then the general eqm conditions). ShootingMapFn is the ordinary shooting update, which
% is cheap because it is handed the general eqm conditions rather than recomputing them. DistanceFn is
% the convergence criterion, and is also what the safeguard judges trial points on.
GEcondnsFn=@(p) reshape(TransitionPath_FHorz_PType_singlepathiter(reshape(p,T,nPrices), PricePathNames, PricePathSizeVec, PricePathSizeVec_ii, ParamPath, ParamPathNames, ParamPathSizeVec, T, FnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, transpathoptions, PTypeStructure),[],1);
ShootingMapFn=@(p,GEc) reshape(updatePricePathNew_TPath_T(reshape(GEc,T-1,nGEcondns),reshape(p,T,nPrices),T,itercounter,transpathoptions),[],1);
if transpathoptions.multiGEcriterion==0
    DistanceFn=@(GEc) max(sum(abs(transpathoptions.multiGEweights.*reshape(GEc,T-1,nGEcondns)),2));
elseif transpathoptions.multiGEcriterion==1
    DistanceFn=@(GEc) max(sqrt(sum(transpathoptions.multiGEweights.*(reshape(GEc,T-1,nGEcondns).^2),2)));
end

andersonoptions=transpathoptions.anderson;
andersonoptions.verbose=transpathoptions.verbose;
if ~isfield(andersonoptions,'maxiter')
    andersonoptions.maxiter=transpathoptions.maxiter; % so that transpathoptions.maxiter means what it does for every other transition path algorithm
end

%% Solve
[p,GEcondns,output]=AndersonAcceleration(GEcondnsFn,ShootingMapFn,DistanceFn,reshape(PricePathOld,[],1),transpathoptions.toleranceGEcondns,andersonoptions);

PricePathOld=reshape(p,T,nPrices);
GEcondnPath=reshape(GEcondns,T-1,nGEcondns);

if transpathoptions.verbose==1
    fprintf('Number of iterations on transition path (Anderson): %i \n',output.iterations)
    fprintf('Current distance of the general eqm conditions from zero: %8.6f \n', DistanceFn(GEcondns))
    fprintf('Ratio of current distance to the convergence tolerance: %.2f (convergence when reaches 1) \n',DistanceFn(GEcondns)/transpathoptions.toleranceGEcondns)
    fprintf('Number of Anderson steps rejected by the safeguard along the way: %i \n',output.nrejectedsteps)
end

%% Create plots of the transition path
% Unlike the other algorithms these are only drawn once, at the end. The Anderson iteration lives
% inside AndersonAcceleration(), which has no way to return the aggregate variables of the path it
% is currently on, so drawing them each iteration would mean solving the path a second time.
if transpathoptions.graphpricepath==1 || transpathoptions.graphaggvarspath==1 || transpathoptions.graphGEcondns==1
    AggVarsPooledPath=zeros(T-1,length(FnNames),'gpuArray'); % only the aggregate variables graph needs these, and getting them means solving the path once more
    if transpathoptions.graphaggvarspath==1
        [~,AggVarsPooledPath]=TransitionPath_FHorz_PType_singlepathiter(PricePathOld, PricePathNames, PricePathSizeVec, PricePathSizeVec_ii, ParamPath, ParamPathNames, ParamPathSizeVec, T, FnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, transpathoptions, PTypeStructure);
    end
    createTPathFeedbackPlots(PricePathNames,FnNames,GEeqnNames,PricePathOld,AggVarsPooledPath,GEcondnPath,transpathoptions);
end

end
