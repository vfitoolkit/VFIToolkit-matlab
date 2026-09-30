function [PricePathOld,GEcondnPath]=TransitionPath_FHorz_Anderson(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions)
% Anderson Acceleration of the shooting map on the whole price path. Selected by
% transpathoptions.GEnewprice=2.
%
% The shooting algorithm (GEnewprice=3) is the fixed-point iteration p <-- G(p), where G() applies
% the howtoupdate rule period by period. Anderson mixing combines the last m iterates of that same
% map to take a quasi-Newton-like step, without ever building a Jacobian. So it costs the same one
% path solve per iteration as shooting (plus one more per Anderson step when the safeguard is on),
% unlike GEnewprice=1 whose Jacobian costs many path solves.
%
% The iterate is the whole price path flattened to a column, T*nPrices long. Period T is included
% even though it never moves: updatePricePathNew_TPath_T copies the terminal row through unchanged,
% so those coordinates are a fixed point of G() and contribute nothing to the Anderson history.
%
% Everything Anderson-specific lives in AndersonAcceleration(), which is shared with the stationary
% general eqm solver (heteroagentoptions.fminalgo=9) and with TransitionPath_InfHorz_Anderson.

%% Setup, the shapes of various of these objects vary depending on the setting
[PolicyIndexesPath,N_probs,II1,II2,exceptlastj,exceptfirstj,justfirstj]=TransitionPath_FHorz_substeps_Step0_setup(l_d,l_aprime,N_a,N_semiz,N_z,N_e,N_j,T,transpathoptions,vfoptions,simoptions);

% Semi-exogenous state: the form of pi_e_J the SemiExo value fn wants
pi_e_J_vf=pi_e_J;
if N_semiz>0
    if transpathoptions.fastOLG==1 && N_e>0 && N_z==0
        pi_e_J_vf=reshape(pi_e_J,[N_a*N_j,1,N_e]); % SemiExo value fn keeps the bothz dim, so needs (a,j)-by-1-by-e even when N_z==0
    end
end

nPrices=size(PricePathOld,2);
nGEcondns=length(GeneralEqmEqnsCell);

% Anderson builds its step out of the history of iterates, which only means anything if G() is the
% same map at every iteration. That rules out the additional-factor ramp (setupGEnewprice3_shooting
% errors if it is set for GEnewprice=2), and it is why itercounter is held at 1 rather than counting
% up: the only thing the path substeps use it for is that ramp.
itercounter=1;

%% The three things Anderson Acceleration needs: the residual, the map, and the distance
% GEcondnsFn is the expensive one, a full solve of the path (value fn, agent dist, aggregates,
% general eqm conditions). ShootingMapFn is the ordinary shooting update, which is cheap because it
% is handed the general eqm conditions rather than recomputing them. DistanceFn is the convergence
% criterion, and is also what the safeguard judges trial points on.
GEcondnsFn=@(p) reshape(TransitionPath_FHorz_singlepathiter(reshape(p,T,nPrices), PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj),[],1);
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
    AggVarsPath=zeros(T-1,length(AggVarNames),'gpuArray'); % only the aggregate variables graph needs these, and getting them means solving the path once more
    if transpathoptions.graphaggvarspath==1
        [~,AggVarsPath]=TransitionPath_FHorz_singlepathiter(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj);
    end
    createTPathFeedbackPlots(PricePathNames,AggVarNames,GEeqnNames,PricePathOld,AggVarsPath,GEcondnPath,transpathoptions);
end

end
