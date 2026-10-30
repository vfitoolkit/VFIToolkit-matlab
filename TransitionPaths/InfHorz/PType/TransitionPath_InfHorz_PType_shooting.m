function [PricePathOld,GEcondnPath]=TransitionPath_InfHorz_PType_shooting(PricePathOld, PricePathNames, PricePathSizeVec, PricePathSizeVec_ii, ParamPath, ParamPathNames, ParamPathSizeVec, T, FnNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, transpathoptions, PTypeStructure)
% The shooting algorithm (transpathoptions.GEnewprice=3) for the transition path with permanent types.
% The same as TransitionPath_InfHorz_shooting, with TransitionPath_InfHorz_PType_singlepathiter as the
% single path iteration. Things like z, e and semiz are handled per ptype inside it (to permit
% that they differ across ptype).
%
% PricePathOld is matrix of size T-by-'number of prices' (a price that depends on ptype has N_i columns)
% ParamPath is matrix of size T-by-'number of parameters that change over path'

if transpathoptions.verbose==1
    % Set up some things to be used later
    pathnametitles=strjoin(PricePathNames,' ');
    wpathnametitle=10*length(PricePathNames); % roughly the space that will use to print the prices themselves
end

%% Titles for the graph of the price path: a price that depends on ptype has N_i (blocks of) columns, drawn as
% lines in its subplot in Names_i order, so the title lists them
PricePathNames_plot=PricePathNames;
for pp=1:length(PricePathNames)
    if PTypeStructure.PricePath_Idependsonptype(pp)==1
        PricePathNames_plot{pp}=[PricePathNames{pp},' (',strjoin(PTypeStructure.Names_i,', '),')'];
    end
end
%% Names for the graph of the general eqm conditions: a condition that depends on ptype (transpathoptions.GEptype)
% has N_i columns in GEcondnPath, one per ptype, titled <eqn>_<ptype name> (as the per-ptype aggregates are named in Parameters)
GEeqnNames_plot={};
for gg=1:length(GEeqnNames)
    if transpathoptions.GEptype(gg)==0
        GEeqnNames_plot{end+1}=GEeqnNames{gg};
    else
        for ii=1:PTypeStructure.N_i
            GEeqnNames_plot{end+1}=[GEeqnNames{gg},'_',PTypeStructure.Names_i{ii}];
        end
    end
end

%%
PricePathDist=Inf;
GEcondnPathDist=Inf;
itercounter=1;
converged=0;

% Constraints on the prices: TransitionPath_InfHorz_PType sets useconstraints and the bounds, but default them here too,
% above their read, as this command reads them itself
if ~isfield(transpathoptions,'useconstraints')
    transpathoptions.useconstraints=0;
end
nconstraintcuts=0; % how many (period,price) elements the constraints cut back on the most recent update
nconstraintiters=0; % how many updates the constraints cut back at all
while itercounter<=transpathoptions.maxiter % convergence is tested further down, at the point where the distances are known, so that the loop stops on the path it just evaluated

    %% One iteration of the path: each ptype, value fn backwards and agent dist forwards, then the general eqm conditions on the pooled aggregates
    [GEcondnPath,AggVarsPooledPath]=TransitionPath_InfHorz_PType_singlepathiter(PricePathOld, PricePathNames, PricePathSizeVec, PricePathSizeVec_ii, ParamPath, ParamPathNames, ParamPathSizeVec, T, FnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, transpathoptions, PTypeStructure);

    %% Now update prices, give verbose feedback, and check for convergence
    % Every general eqm condition, for every period, is now known, so the update can use them together
    PricePathNew=updatePricePathNew_TPath_T(GEcondnPath,PricePathOld,T,itercounter,transpathoptions);

    % See how far apart the price paths are
    % A price path can reach somewhere the model cannot be solved, and the general eqm conditions then
    % come back non-finite. Stop rather than carry on: every later update just propagates it, and
    % there is no good path left to fall back to. Note that a NaN passes silently through any > or <=
    % test, so without this the iteration would either run out its full maxiter or, worse, look
    % converged and return the NaN path as the answer.
    if any(~isfinite(GEcondnPath),'all')
        error(['TransitionPath_InfHorz_PType_shooting: the general eqm conditions are NaN/Inf at iteration %i. ' ...
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
    createTPathFeedbackPlots(PricePathNames_plot,FnNames,GEeqnNames_plot,PricePathOld,AggVarsPooledPath,GEcondnPath,transpathoptions,PricePathSizeVec);

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
    if PricePathDist<=transpathoptions.toleranceGEprices && GEcondnPathDist<=transpathoptions.toleranceGEcondns
        converged=1;
        break
    end

    % Update PricePathOld
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
        warning('TransitionPath_InfHorz_PType: the constraints on the prices cut back the last update (%i elements), so the price path may be pressing against a bound',nconstraintcuts)
    end
end

if converged==0
    warning(['TransitionPath_InfHorz_PType_shooting: reached maxiter (%i) without convergence; the general eqm ' ...
        'conditions are %8.6f from zero, against toleranceGEcondns=%g. Consider increasing ' ...
        'transpathoptions.maxiter, or adjusting the GEnewprice3.howtoupdate factors.'], ...
        transpathoptions.maxiter,GEcondnPathDist,transpathoptions.toleranceGEcondns)
end

end
