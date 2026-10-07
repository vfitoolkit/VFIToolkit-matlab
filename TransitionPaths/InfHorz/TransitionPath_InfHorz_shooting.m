function [PricePathOld,GEcondnPath]=TransitionPath_InfHorz_shooting(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d,n_a,n_z,n_e, N_d,N_a,N_z,N_e, l_d,l_aprime,l_a,l_z,l_e, d_gridvals,aprime_gridvals,a_gridvals,a_grid,z_gridvals,e_gridvals,ze_gridvals,pi_z,pi_z_sparse,pi_e, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions,transpathoptions)
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
[PolicyIndexesPath,N_probs,II1,II2]=TransitionPath_InfHorz_substeps_Step0_setup(l_d,l_aprime,N_a,N_z,N_e,T,transpathoptions,vfoptions,simoptions);

%%
PricePathDist=Inf;
GEcondnPathDist=Inf;
itercounter=1;

%% Local search, vfoptions.localsearch=1
% The scheme: iteration 1 is a STANDARD sweep, whatever divideandconquer and gridinterplayer are set
% to, and its Policy becomes the reference. Later iterations restrict the aprime search to a window
% around the previous iteration's answer, refreshing the reference as they go. When those converge,
% ONE standard sweep re-evaluates the SAME price path: if it is still converged we are done and the
% answer is certified by an exact solve, and if it moves us then its Policy becomes the new reference
% and local search resumes. So the restriction can never change the answer, only how fast it is
% reached, and the certification needs no assumption about the shape of the problem.
% vfoptions.localsearch is toggled per iteration, which is all Step1 needs to know; uselocalsearch
% remembers what the caller actually asked for.
% The alternation is bounded only by maxiter. Termination is not provable: each verification that
% moves us lands somewhere else with a fresh reference, so it should progress, but if it thrashes the
% existing non-convergence warning is what fires. nverify is reported for exactly that reason.
uselocalsearch=vfoptions.localsearch;
vfoptions.localsearch=0; % iteration 1 is standard
aprimeReferencePath=[];  % Step1 allocates and fills it on that first sweep
nverify=0;
lsmovemax=0; lsnbind=0; lsntotal=0;
converged=0;
while itercounter<=transpathoptions.maxiter % convergence is tested further down, at the point where the distances are known, so that the loop stops on the path it just evaluated

    %% One iteration of the path: value fn backwards, agent dist forwards, general eqm conditions
    [GEcondnPath,AggVarsPath,PolicyIndexesPath,PricePathNew,aprimeReferencePath,lsdiag]=TransitionPath_InfHorz_singlepathiter(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d, n_a, n_z, n_e, N_a, N_z, N_e, l_d, l_aprime, d_gridvals, aprime_gridvals, a_gridvals, a_grid, z_gridvals, e_gridvals, ze_gridvals, pi_z, pi_z_sparse, pi_e, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, aprimeReferencePath, N_probs, II1, II2);
    

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
        error(['TransitionPath_InfHorz_shooting: the general eqm conditions are NaN/Inf at iteration %i. ' ...
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
    % How far the policy moved from the reference it was given, accumulated over the local search
    % iterations. At an interior reference a move below nlocalsearch means the window did not bind, so
    % the restricted answer was the unrestricted one; the share that reach nlocalsearch is therefore
    % both the statistic that says what window a model needs AND the one that says whether this could
    % be made exact per iteration, which is what Anderson would require and shooting does not.
    if ~isempty(lsdiag)
        lsmovemax=max(lsmovemax,lsdiag.movemax);
        lsnbind=lsnbind+lsdiag.nbind;
        lsntotal=lsntotal+lsdiag.ntotal;
    end

    if PricePathDist<=transpathoptions.toleranceGEprices && GEcondnPathDist<=transpathoptions.toleranceGEcondns
        if vfoptions.localsearch==0
            converged=1;
            break % converged under a standard sweep, so the answer is certified
        else
            % Converged under local search. Re-evaluate THIS SAME path with a standard sweep: the
            % prices are deliberately NOT updated, or the check would be of a different path.
            vfoptions.localsearch=0;
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
    end

    PricePathOld=updatePricePath(PricePathOld,PricePathNew,transpathoptions,T);

    itercounter=itercounter+1;


end

%% Local search report
% nverify is the number of times the local search declared convergence and a standard sweep was run
% to check it. One means the first check passed, so the restricted iterations landed where an exact
% solve agrees. More than one means a check moved the path and local search had to resume, which is
% the scheme working but costing more than it should.
% The policy movement is what says whether nlocalsearch is set sensibly: the largest move any state
% made away from its reference is the nlocalsearch at or above which the window would never have
% bound. The share that reached nlocalsearch is the share of state-periods where the window DID bind,
% and so is also the share at which this scheme could not be certified exact iteration by iteration
% -- which is what the Anderson algorithm would need, and this one does not.
% Printed whenever local search was used, not only when verbose is on: it is one line, it reports an
% opt-in feature, and the bind rate is the thing a user needs in order to choose nlocalsearch at all.
if uselocalsearch==1
    fprintf('Local search: %i iterations, of which %i standard verification sweep(s); the largest policy move from its reference was %i grid points, so nlocalsearch=%i would never have bound; the window bound at %.2f%% of state-periods \n',itercounter,nverify,lsmovemax,lsmovemax,100*lsnbind/max(lsntotal,1))
end


if converged==0
    warning(['TransitionPath_InfHorz_shooting: reached maxiter (%i) without convergence; the general eqm ' ...
        'conditions are %8.6f from zero, against toleranceGEcondns=%g. Consider increasing ' ...
        'transpathoptions.maxiter, or adjusting the GEnewprice3.howtoupdate factors.'], ...
        transpathoptions.maxiter,GEcondnPathDist,transpathoptions.toleranceGEcondns)
end



end
