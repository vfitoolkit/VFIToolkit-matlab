function [PricePathOld,GEcondnPath]=TransitionPath_FHorz_shooting(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_semiz,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions)
% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

if transpathoptions.verbose==1
    % Set up some things to be used later
    pathnametitles=strjoin(PricePathNames,' ');
    wpathnametitle=10*length(PricePathNames); % roughly the space that will use to print the prices themselves
    % fprintf('%-*s || %-*s \n',wpathnametitle,'Old',wpathnametitle,'New')
    % fprintf('%-*s || %-*s \n',wpathnametitle,pathnametitles,wpathnametitle,pathnametitles)
end

%%
% Setup, the shapes of various of these objects vary depending on the setting
[PolicyIndexesPath,N_probs,II1,II2,exceptlastj,exceptfirstj,justfirstj]=TransitionPath_FHorz_substeps_Step0_setup(l_d,l_aprime,N_a,N_semiz,N_z,N_e,N_j,T,transpathoptions,vfoptions,simoptions);
% Note: some of these outputs are empty, depending on the setting
% PolicyIndexesPath is just pre-allocating matrices of the appropriate size, contains zeros

%% Semi-exogenous state: prepare the value-function-form grids/transitions (the substeps below dispatch to SemiExo variants)
pi_e_J_vf=pi_e_J;
if N_semiz>0
    if transpathoptions.fastOLG==1 && N_e>0 && N_z==0
        pi_e_J_vf=reshape(pi_e_J,[N_a*N_j,1,N_e]); % SemiExo value fn keeps the bothz dim, so needs (a,j)-by-1-by-e even when N_z==0
    end
end

%%
PricePathDist=Inf;
GEcondnPathDist=Inf;
itercounter=1;

%% Local search, vfoptions.localsearch=1
% The scheme is TransitionPath_InfHorz_shooting's, unchanged; see there for the reasoning. Iteration 1
% is a STANDARD sweep that also seeds the reference (vfoptions.localsearchseed); later iterations
% restrict aprime to a window around the previous iteration's answer at the same (j,t); when those
% converge, ONE standard sweep re-evaluates the SAME price path, and only that can end the solve. If it
% moves us, its answer reseeds the reference and local search resumes.
% THE VERIFICATION SWEEP CERTIFIES THE GE CONDITIONS, NOT THE POLICY. A local-optimum trap at a few
% states, or at states with little mass, can survive it; with multiple equilibria the restriction can
% select a different one. Neither occurs with an objective single-peaked in aprime.
% localsearch is defaulted here, above its first read, because the PType and MixHorz solvers reach
% this file with a vfoptions that has never heard of it.
if ~isfield(vfoptions,'localsearch')
    vfoptions.localsearch=0;
end
uselocalsearch=vfoptions.localsearch;
vfoptions.localsearch=0; % iteration 1 is standard
vfoptions.localsearchseed=uselocalsearch;
aprimeReferencePath=[];  % Step1 allocates and fills it on that first sweep
nverify=0;
lsmovemax=0; lsnedge=0; lsntotal=0; lsnedgedstar=0; lsntotaldstar=0;
% nlocalsearch is a STARTING window. It resets to the input value on entering local search (after
% iteration 1 and after every verification sweep) and then ratchets: up by nlocalsearchup whenever the
% answer sat on a window edge anywhere, down by nlocalsearchdown otherwise, floor 1, cap where the
% window is the whole grid. ONE window for every (j,t): a per-age window would be more state for no
% measured need.
if uselocalsearch==1
    nlocalsearchstart=vfoptions.nlocalsearch;
    nlocalsearchcap=floor((N_a-1)/2); % at the cap the window is the whole grid
    lsnmin=Inf; lsnmax=0; lsnup=0; lsndown=0; lsnpinned=0;
end
converged=0;
while itercounter<=transpathoptions.maxiter % convergence is tested further down, at the point where the distances are known, so that the loop stops on the path it just evaluated

    %% One iteration of the path: value fn backwards, agent dist forwards, general eqm conditions
    [GEcondnPath,AggVarsPath,PolicyIndexesPath,PricePathNew,aprimeReferencePath,lsdiag]=TransitionPath_FHorz_singlepathiter(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, aprimeReferencePath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj);


    %% Now update prices, give verbose feedback, and check for convergence
    if transpathoptions.updatepert==0
        % Every general eqm condition, for every period, is now known, so the update can use them together
        PricePathNew=updatePricePathNew_TPath_T(GEcondnPath,PricePathOld,T,itercounter,transpathoptions);
    end

    % See how far apart the price paths are
    % A price path can reach somewhere the model cannot be solved, and the general eqm conditions then
    % come back non-finite. Stop rather than carry on: every later update just propagates it, and
    % there is no good path left to fall back to. Note that a NaN passes silently through any > or <=
    % test, so without this the iteration would either run out its full maxiter or, worse, look
    % converged and return the NaN path as the answer.
    if any(~isfinite(GEcondnPath),'all')
        error(['TransitionPath_FHorz_shooting: the general eqm conditions are NaN/Inf at iteration %i. ' ...
            'The price path has reached somewhere the model cannot be solved. Try a smaller factor ' ...
            'in GEnewprice3.howtoupdate, a less aggressive GEnewprice3.additionalfactor, or a ' ...
            'starting price path closer to the solution.'],itercounter)
    end

    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    % Notice that the distance is always calculated ignoring the time t=T periods, as these needn't ever converges
    % And how far the general eqm conditions are from zero. Scalarize across the general eqm eqns in each
    % time period the same way the stationary general eqm does, then take the L-Infinity norm over time
    % (the same norm as is used for the prices). GEcondnPath is the raw conditions, before the permute
    % and before updateaccuracycutoff is applied.
    if transpathoptions.multiGEcriterion==0
        GEcondnPathDist=max(sum(abs(transpathoptions.multiGEweights.*GEcondnPath),2));
    elseif transpathoptions.multiGEcriterion==1
        GEcondnPathDist=max(sqrt(sum(transpathoptions.multiGEweights.*(GEcondnPath.^2),2)));
    end

    if transpathoptions.verbose==1
        fprintf(' \n')
        fprintf('%-*s || %-*s \n',wpathnametitle,'Old',wpathnametitle,'New')
        fprintf('%-*s || %-*s \n',wpathnametitle,pathnametitles,wpathnametitle,pathnametitles)

        % Would be nice to have a way to get the iteration count without having the whole printout of path values (I think that would be useful?)
        [PricePathOld,PricePathNew]
    end

    % Create plots of the transition path (before we update pricepath)
    createTPathFeedbackPlots(PricePathNames,AggVarNames,GEeqnNames,PricePathOld,AggVarsPath,GEcondnPath,transpathoptions);

    % Update PricePathOld

    TransPathConvergence=max(PricePathDist/transpathoptions.toleranceGEprices,GEcondnPathDist/transpathoptions.toleranceGEcondns); % So when this gets to 1 we have convergence, we require convergence in both
    if transpathoptions.verbose==1
        fprintf('Number of iterations on transition path: %i \n',itercounter)
        if isfinite(transpathoptions.toleranceGEprices)
            fprintf('Current distance between old and new price path (in L-Infinity norm): %8.6f \n', PricePathDist)
        end
        fprintf('Current distance of the general eqm conditions from zero: %8.6f \n', GEcondnPathDist)
        fprintf('Ratio of current distance to the convergence tolerance: %.2f (convergence when reaches 1) \n',TransPathConvergence)
    end

    if transpathoptions.historyofpricepath==1
        % Store the whole history of the price path and save it every ten iterations
        PricePathHistory{itercounter,1}=PricePathDist;
        PricePathHistory{itercounter,2}=PricePathOld;
        if rem(itercounter,10)==1
            save ./SavedOutput/TransPath_Internal.mat PricePathHistory
        end
    end


    % Convergence. Tested here, after the distances are known but before the price path is updated,
    % so that what gets returned is the path whose general eqm conditions were actually evaluated.
    % Testing it at the top of the loop instead would leave the loop having applied one more update
    % than it checked, and so return a path one step past the GEcondnPath returned alongside it.
    % The ratchet on nlocalsearch, driven by the edge count of the restricted sweep just done
    if ~isempty(lsdiag)
        lsmovemax=max(lsmovemax,lsdiag.movemax);
        lsnedge=lsnedge+lsdiag.nedge;
        lsntotal=lsntotal+lsdiag.ntotal;
        lsnedgedstar=lsnedgedstar+lsdiag.nedgedstar;
        lsntotaldstar=lsntotaldstar+lsdiag.ntotaldstar;
        lsnmin=min(lsnmin,vfoptions.nlocalsearch);
        lsnmax=max(lsnmax,vfoptions.nlocalsearch);
        if lsdiag.nedge>0
            vfoptions.nlocalsearch=min(vfoptions.nlocalsearch+vfoptions.nlocalsearchup,nlocalsearchcap);
            lsnup=lsnup+1;
        else
            vfoptions.nlocalsearch=max(vfoptions.nlocalsearch-vfoptions.nlocalsearchdown,1);
            lsndown=lsndown+1;
        end
        if vfoptions.nlocalsearch>=nlocalsearchcap
            lsnpinned=lsnpinned+1;
        end
    end

    if PricePathDist<=transpathoptions.toleranceGEprices && GEcondnPathDist<=transpathoptions.toleranceGEcondns
        if vfoptions.localsearch==0
            converged=1;
            break % converged under a standard sweep, so the answer is certified
        else
            % Converged under local search. Re-evaluate THIS SAME path with a standard sweep: the
            % prices are deliberately NOT updated, or the check would be of a different path.
            vfoptions.localsearch=0;
            vfoptions.localsearchseed=1; % the verification sweep reseeds the reference, per d
            nverify=nverify+1;
            if transpathoptions.verbose==1
                fprintf('Local search converged at iteration %i, checking it with a standard sweep \n',itercounter)
            end
            itercounter=itercounter+1;
            continue
        end
    end

    if uselocalsearch==1 && vfoptions.localsearch==0
        vfoptions.localsearch=1; % a reference exists now, so the restricted search can start
        vfoptions.localsearchseed=0;
        vfoptions.nlocalsearch=nlocalsearchstart; % the input value is the window used immediately after every standard sweep
    end

    PricePathOld=updatePricePath(PricePathOld,PricePathNew,transpathoptions,T);

    itercounter=itercounter+1;


end

%% Local search report
% Printed whenever local search was used, not only when verbose is on: it is a few lines, it reports
% an opt-in feature, and the edge rate is what a user needs in order to choose nlocalsearch at all.
% nverify is the number of standard verification sweeps; one means the first check passed.
if uselocalsearch==1
    fprintf('Local search: %i iterations, of which %i standard verification sweep(s) \n',itercounter,nverify)
    if lsnup+lsndown>0
        fprintf('Local search: nlocalsearch started at %i, ranged %i to %i, ended at %i, with %i ratchet(s) up and %i down \n',nlocalsearchstart,lsnmin,lsnmax,vfoptions.nlocalsearch,lsnup,lsndown)
        % movemax is bounded by the window; the edge share is the signal the ratchet responds to
        fprintf('Local search: the answer sat on a window edge, excluding the grid ends, at %.2f%% of state-age-periods; the largest move from a reference was %i grid points, which the window bounds \n',100*lsnedge/max(lsntotal,1),lsmovemax)
        % With a d the ratchet is driven by the ALL-d rate; the chosen-d rate answers whether the
        % restriction bound the policy actually realised. See TransitionPath_InfHorz_shooting.
        if lsntotaldstar>0
            fprintf('Local search: that edge rate is over every d; at the CHOSEN d only it is %.2f%% of state-age-periods \n',100*lsnedgedstar/max(lsntotaldstar,1))
        end
    end
    if lsnpinned>0
        fprintf('Local search: WARNING nlocalsearch reached its cap of %i on %i iteration(s). At the cap the window covers the whole grid AND pays the index overhead, so local search is strictly worse than not using it on this model. \n',nlocalsearchcap,lsnpinned)
    end
end

if converged==0
    warning(['TransitionPath_FHorz_shooting: reached maxiter (%i) without convergence; the general eqm ' ...
        'conditions are %8.6f from zero, against toleranceGEcondns=%g. Consider increasing ' ...
        'transpathoptions.maxiter, or adjusting the GEnewprice3.howtoupdate factors.'], ...
        transpathoptions.maxiter,GEcondnPathDist,transpathoptions.toleranceGEcondns)
end


end
