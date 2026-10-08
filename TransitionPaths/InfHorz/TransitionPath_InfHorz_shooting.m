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
lsmovemax=0; lsnedge=0; lsntotal=0;
% nlocalsearch is a STARTING window, not the window. It resets to the input value on entering local
% search -- which happens after iteration 1 and after every verification sweep -- and then ratchets:
% up by vfoptions.nlocalsearchup whenever the answer sat on a window edge anywhere, down by
% nlocalsearchdown otherwise. Only restricted sweeps ratchet; a standard one has no window and so
% gives no signal. The trajectory of nlocalsearch is the diagnostic worth reading: it is what window
% THIS application needed, found by the solver rather than guessed in advance.
if uselocalsearch==1
    nlocalsearchstart=vfoptions.nlocalsearch;
    nlocalsearchcap=floor((N_a-1)/2); % at the cap the window is the whole grid
    lsnmin=Inf; lsnmax=0; lsnup=0; lsndown=0; lsnpinned=0;
end
% Two measurements for the Anderson question, both gated on transpathoptions.localsearchdiagnostic.
% (1) CUMULATIVE movement: how far the policy ends up from the reference the FIRST standard sweep
%     produced. A frozen reference -- which is what Anderson could have without forking the shared
%     accelerator, since its closure can capture one by value -- has to span this, not the
%     per-iteration movement that lsmovemax reports.
% (2) RESIDUAL CONTAMINATION: at each iteration, the general eqm residual from the restricted sweep
%     against the residual an EXACT sweep gives at the SAME prices. That difference is what Anderson
%     would be differencing into DeltaF, and what matters is its size relative to the genuine change
%     in residual between iterations, so both are accumulated and the ratio reported. Measuring this
%     costs a second full iteration each time, which is why it is off by default.
aprimeReferenceSeed=[];
lscumulmax=0;
lscontam=0; lsdF=0; GEcondnPathPrev=[];
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
        lsnedge=lsnedge+lsdiag.nedge;
        lsntotal=lsntotal+lsdiag.ntotal;
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

    if uselocalsearch==1 && transpathoptions.localsearchdiagnostic==1
        if isempty(aprimeReferenceSeed)
            % Iteration 1 is the standard sweep, so this is the reference a frozen scheme would use
            aprimeReferenceSeed=aprimeReferencePath;
        else
            lscumulmax=max(lscumulmax,gather(max(abs(aprimeReferencePath-aprimeReferenceSeed),[],'all')));
        end
        if vfoptions.localsearch==1
            % The same prices, solved exactly. vfoptionsX is this iteration's options with the
            % restriction off; the reference goes in as [] because a standard sweep does not read one.
            vfoptionsX=vfoptions;
            vfoptionsX.localsearch=0;
            GEcondnPathX=TransitionPath_InfHorz_singlepathiter(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d, n_a, n_z, n_e, N_a, N_z, N_e, l_d, l_aprime, d_gridvals, aprime_gridvals, a_gridvals, a_grid, z_gridvals, e_gridvals, ze_gridvals, pi_z, pi_z_sparse, pi_e, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptionsX, simoptions, transpathoptions, itercounter, PolicyIndexesPath, [], N_probs, II1, II2);
            lscontam=max(lscontam,gather(max(abs(GEcondnPath-GEcondnPathX),[],'all')));
        end
        if ~isempty(GEcondnPathPrev)
            lsdF=max(lsdF,gather(max(abs(GEcondnPath-GEcondnPathPrev),[],'all')));
        end
        GEcondnPathPrev=GEcondnPath;
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
        vfoptions.nlocalsearch=nlocalsearchstart; % the input value is the window used immediately after every standard sweep
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
    fprintf('Local search: %i iterations, of which %i standard verification sweep(s) \n',itercounter,nverify)
    if lsnup+lsndown>0
        fprintf('Local search: nlocalsearch started at %i, ranged %i to %i, ended at %i, with %i ratchet(s) up and %i down \n',nlocalsearchstart,lsnmin,lsnmax,vfoptions.nlocalsearch,lsnup,lsndown)
        % movemax is bounded by the window, so it says what the window allowed and not how far the
        % policy would have moved. The edge share is the signal the ratchet actually responds to.
        fprintf('Local search: the answer sat on a window edge, excluding the grid ends, at %.2f%% of state-periods; the largest move from a reference was %i grid points, which the window bounds \n',100*lsnedge/max(lsntotal,1),lsmovemax)
    end
    if lsnpinned>0
        fprintf('Local search: WARNING nlocalsearch reached its cap of %i on %i iteration(s). At the cap the window covers the whole grid AND pays the index overhead, so local search is strictly worse than not using it on this model. \n',nlocalsearchcap,lsnpinned)
    end
    if transpathoptions.localsearchdiagnostic==1
        % lscumulmax is what a FROZEN reference would have to span, against lsmovemax for a refreshed
        % one. lscontam over lsdF is the signal-to-noise Anderson would be working with: the error in
        % a residual against the genuine change in residual between iterations, which is the quantity
        % its DeltaF history actually holds.
        fprintf('Local search diagnostic: policy ended %i grid points from the FIRST reference (against %i per iteration), so a frozen reference needs nlocalsearch=%i \n',lscumulmax,lsmovemax,lscumulmax)
        fprintf('Local search diagnostic: residual contamination %.3e against a between-iteration residual change of %.3e, a ratio of %.3e \n',lscontam,lsdF,lscontam/max(lsdF,realmin))
    end
end


if converged==0
    warning(['TransitionPath_InfHorz_shooting: reached maxiter (%i) without convergence; the general eqm ' ...
        'conditions are %8.6f from zero, against toleranceGEcondns=%g. Consider increasing ' ...
        'transpathoptions.maxiter, or adjusting the GEnewprice3.howtoupdate factors.'], ...
        transpathoptions.maxiter,GEcondnPathDist,transpathoptions.toleranceGEcondns)
end



end
