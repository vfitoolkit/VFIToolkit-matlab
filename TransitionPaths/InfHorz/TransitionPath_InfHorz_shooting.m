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

% Constraints on the prices: TransitionPath_InfHorz sets useconstraints and the bounds, but default them here too,
% above their read, as this command reads them itself
if ~isfield(transpathoptions,'useconstraints')
    transpathoptions.useconstraints=0;
end
nconstraintcuts=0; % how many (period,price) elements the constraints cut back on the most recent update
nconstraintiters=0; % how many updates the constraints cut back at all

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
% localsearchseed tells the single-step dispatcher that a standard sweep must ALSO leave behind a
% reference for the next restricted sweep. It is set on exactly the sweeps where localsearch is off
% but in use for the solve -- iteration 1 and every verification sweep -- and is what makes the
% scheme work with a d, where the reference has to exist for every d and not only the one the sweep
% chose. With no d it also unifies the reference: both kinds of sweep now return the coarse argmax
% of their own search, where the standard sweep used to have its converted from Policy.
vfoptions.localsearchseed=uselocalsearch;
aprimeReferencePath=[];  % Step1 allocates and fills it on that first sweep
nverify=0;
lsmovemax=0; lsnedge=0; lsntotal=0; lsnedgedstar=0; lsntotaldstar=0;
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
%     would be differencing into DeltaF. What matters is its size relative to the GENUINE change in
%     residual between iterations, and that ratio is taken PER ITERATION, never as one maximum
%     divided by another: the genuine change is largest early and the contamination matters most
%     late, when the residual has nearly stopped moving, so dividing the two maxima would report a
%     flattering number drawn from two different iterations. The per-iteration ratio's maximum, and
%     its value on the last iteration, are the two numbers that bear on Anderson.
%     Each ratio is taken between the two arrays' max-abs within one iteration, not elementwise: an
%     individual condition can be momentarily stationary, giving a denominator near zero and a ratio
%     that says nothing.
%     Measuring this costs a second full iteration each time, which is why it is off by default.
aprimeReferenceSeed=[];
lscumulmax=0;
lscontam=0; lsdF=0; GEcondnPathPrev=[];
lsratiomax=0; lsratiolast=NaN; lscontamlast=NaN; lsdFlast=NaN; lsratioiter=0;
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

    if uselocalsearch==1 && transpathoptions.localsearchdiagnostic==1
        if isempty(aprimeReferenceSeed)
            % Iteration 1 is the standard sweep, so this is the reference a frozen scheme would use
            aprimeReferenceSeed=aprimeReferencePath;
        else
            lscumulmax=max(lscumulmax,gather(max(abs(aprimeReferencePath-aprimeReferenceSeed),[],'all')));
        end
        contamnow=NaN;
        if vfoptions.localsearch==1
            % The same prices, solved exactly. vfoptionsX is this iteration's options with the
            % restriction off; the reference goes in as [] because a standard sweep does not read one.
            % PolicyIndexesPath goes in unchanged because Step1 treats it as scratch and overwrites it,
            % so handing over the restricted sweep's copy cannot contaminate this exact evaluation.
            vfoptionsX=vfoptions;
            vfoptionsX.localsearch=0;
            GEcondnPathX=TransitionPath_InfHorz_singlepathiter(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d, n_a, n_z, n_e, N_a, N_z, N_e, l_d, l_aprime, d_gridvals, aprime_gridvals, a_gridvals, a_grid, z_gridvals, e_gridvals, ze_gridvals, pi_z, pi_z_sparse, pi_e, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptionsX, simoptions, transpathoptions, itercounter, PolicyIndexesPath, [], N_probs, II1, II2);
            contamnow=gather(max(abs(GEcondnPath-GEcondnPathX),[],'all'));
            lscontam=max(lscontam,contamnow);
        end
        dFnow=NaN;
        if ~isempty(GEcondnPathPrev)
            dFnow=gather(max(abs(GEcondnPath-GEcondnPathPrev),[],'all'));
            lsdF=max(lsdF,dFnow);
        end
        % Both defined means this is a restricted sweep with a predecessor, which is exactly an
        % iteration whose residual difference Anderson would have put in DeltaF.
        if ~isnan(contamnow) && ~isnan(dFnow)
            rationow=contamnow/max(dFnow,realmin);
            lsratiomax=max(lsratiomax,rationow);
            lsratiolast=rationow; lscontamlast=contamnow; lsdFlast=dFnow;
            lsratioiter=lsratioiter+1;
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

    PricePathProposed=updatePricePath(PricePathOld,PricePathNew,transpathoptions,T);
    if transpathoptions.useconstraints==1
        % Fraction-to-the-boundary: any (period,price) that the update would take to or past its bound moves only
        % constraintstepfraction of the way from its current value to that bound instead. Everything else is the
        % update exactly as proposed. PricePathOld is strictly inside the bounds (the initial guess is checked, and
        % this rule keeps it there), so the result is too. Row T is never updated.
        % Done by logical indexing, never by multiplying by an indicator: an unconstrained column has bounds -Inf and
        % Inf, and 0*Inf is NaN.
        Theta=repmat(transpathoptions.constraintstepfractionvec,T-1,1); % the step fraction can differ by price
        LB=repmat(transpathoptions.pricepathlowerbound,T-1,1);
        UB=repmat(transpathoptions.pricepathupperbound,T-1,1);
        Pprop=PricePathProposed(1:T-1,:);
        Pold=PricePathOld(1:T-1,:);
        belowlb=(Pprop<=LB);
        aboveub=(Pprop>=UB);
        Pprop(belowlb)=Pold(belowlb)+Theta(belowlb).*(LB(belowlb)-Pold(belowlb));
        Pprop(aboveub)=Pold(aboveub)+Theta(aboveub).*(UB(aboveub)-Pold(aboveub));
        PricePathProposed(1:T-1,:)=Pprop;
        nconstraintcuts=gather(sum(belowlb(:))+sum(aboveub(:)));
        if nconstraintcuts>0
            nconstraintiters=nconstraintiters+1;
        end
        if transpathoptions.verbose==1
            fprintf('Constraints on the prices cut back %i (period,price) elements of this update \n',nconstraintcuts)
        end
    end
    PricePathOld=PricePathProposed;

    itercounter=itercounter+1;


end

%% Constraints on the prices report
% One line whenever constraints were set, as for local search: they are opt-in, and whether they ever bound is what
% the user needs to know. If they cut back the last update, the path was still pressing against a bound, and a
% solution right at a bound cannot be reached exactly (the bounds are open), so that gets a warning.
if transpathoptions.useconstraints==1
    fprintf('Constraints on the prices: cut back %i of the %i updates \n',nconstraintiters,itercounter-1)
    if nconstraintcuts>0
        warning('TransitionPath_InfHorz: the constraints on the prices cut back the last update (%i elements), so the price path may be pressing against a bound',nconstraintcuts)
    end
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
        % WITH A d THERE ARE TWO READINGS, and the ratchet is driven by the ALL-d one. That is a
        % choice about robustness, not about this model: a d whose window truncated it might have won
        % on a wider window, and that loss is invisible without computing it, so the count has to
        % include the d values the period rejected.
        % DO NOT SWITCH IT TO THE CHOSEN-d RATE ON THE STRENGTH OF THESE TWO NUMBERS. On a model where
        % the optimal aprime varies smoothly and monotonically in d, every per-d reference is well
        % centred, so the all-d rate comes out LOWER than the chosen-d rate -- 0.16% against 0.64% on
        % the doPart(6) labour-supply model -- and looks like the weaker signal. Where d has no such
        % ordering, a rejected d's optimum can sit far from its reference and bind, and that is
        % precisely the d the chosen-d count cannot see.
        % The chosen-d rate is printed because it answers a different question: whether the
        % restriction bound the policy that was actually realised.
        if lsntotaldstar>0
            fprintf('Local search: that edge rate is over every d; at the CHOSEN d only it is %.2f%% of state-periods \n',100*lsnedgedstar/max(lsntotaldstar,1))
        end
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
        fprintf('Local search diagnostic: largest residual contamination %.3e, largest between-iteration residual change %.3e, over %i measured iteration(s) \n',lscontam,lsdF,lsratioiter)
        fprintf('Local search diagnostic: the PER-ITERATION ratio of contamination to residual change peaked at %.3e, and on the last measured iteration was %.3e (contamination %.3e against a change of %.3e) \n',lsratiomax,lsratiolast,lscontamlast,lsdFlast)
        % Below about 1e-2 the restricted residuals are signal and Anderson's DeltaF history is sound.
        % Approaching or above 1 the differences Anderson would extrapolate from are mostly the error
        % the restriction introduces, so a per-iteration reference cannot be handed to it and the
        % choice is between a frozen reference and a fork that re-solves exactly before each update.
        if lsratiomax>=1
            fprintf('Local search diagnostic: WARNING the contamination reached the size of the residual change itself, so a per-iteration reference would give Anderson a DeltaF history made mostly of restriction error \n')
        end
    end
end


if converged==0
    warning(['TransitionPath_InfHorz_shooting: reached maxiter (%i) without convergence; the general eqm ' ...
        'conditions are %8.6f from zero, against toleranceGEcondns=%g. Consider increasing ' ...
        'transpathoptions.maxiter, or adjusting the GEnewprice3.howtoupdate factors.'], ...
        transpathoptions.maxiter,GEcondnPathDist,transpathoptions.toleranceGEcondns)
end



end
