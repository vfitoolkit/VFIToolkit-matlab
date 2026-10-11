function [p,GEcondns,output,lsstate]=AndersonAccelerationTPath(GEcondnsFn,ShootingMapFn,DistanceFn,VerifyFn,lsstate0,p0,tolerance,andersonoptions)
% Anderson Acceleration of a fixed-point map
%     p  <--  G(p)
% where G() is a 'shooting' style update built from the general eqm conditions.
% Anderson mixing combines the last m iterates to take a quasi-Newton-like
% step without any derivatives. A safeguard rejects any Anderson step that
% increases the residual, or cannot be evaluated at all (the conditions come back
% NaN/Inf, or the model itself throws), falling back to a plain
% shooting step, so worst-case behaviour is that of the shooting algorithm.
%
% A FORK OF AndersonAcceleration FOR TRANSITION PATHS WITH vfoptions.localsearch. Identical
% algorithm; what differs is that the expensive evaluation carries STATE -- the reference policy the
% restricted value fn search centres its window on -- and that convergence is CERTIFIED by an
% unrestricted re-evaluation before it is believed.
%
% WHY A FORK AND NOT AN OPTION ON THE SHARED FILE. AndersonAcceleration takes a PURE residual
% function and is shared with the stationary general eqm code (fminalgo=9). Threading state through
% it would change that contract for a caller that has no state, so the two are kept apart. Any fix to
% the algorithm itself has to be made in both files; the algorithm is not expected to change often,
% and the alternative was a stateful contract imposed on the stationary solver.
%
% THE TWO CHANGES, both of which are forced rather than chosen:
%   (1) GEcondnsFn and VerifyFn take and return lsstate. The restricted search needs a reference at
%       the prices being evaluated, and produces the next one as a by-product. State is COMMITTED ONLY
%       WHEN THE POINT IT WAS BUILT AT IS KEPT -- see the safeguard below, which evaluates the
%       conditions at a trial point it may then reject.
%   (2) Convergence triggers VerifyFn, an UNRESTRICTED evaluation at the converged prices, and is only
%       believed if that is still inside the tolerance. If it is not, the reference is reseeded from
%       the exact sweep and THE HISTORY IS RESTARTED.
%       The verification CANNOT be an ordinary iteration. DeltaX and DeltaF hold successive
%       DIFFERENCES, so re-evaluating at the same prices gives a zero DeltaX against a non-zero
%       DeltaF: a degenerate secant pair, which is precisely what the restart guards exist to throw
%       away. So it is a side evaluation that never enters the history, and a failure restarts.
%       The restart path is the one the safeguard already uses when it rejects a step.
%
% WHAT THE RESTRICTION DOES TO DeltaF, measured rather than assumed. The restricted residual differs
% from the exact one at the same prices, and that error is differenced into DeltaF along with the
% signal. CoreInfHorzTPathTests doPart(6) measures the ratio PER ITERATION: it peaked at 1.93e-05 and
% did not grow as the solve converged (contamination 1.99e-09 against a residual change of 1.03e-04 on
% the last iteration), because the error shrinks in step with the signal. What is NOT measured is how
% much the least-squares solve amplifies it: sensitivity of gamma to a perturbation in DeltaF scales
% with the conditioning of DeltaF'*DeltaF, and these systems are ill-conditioned (rcond of the TPath
% Jacobian measured 2.5e-3 at epsprice=1e-3). The Tikhonov term and the safeguard both damp it, and
% the verification sweep is what makes a bad gamma recoverable rather than fatal.
%
% A FROZEN reference -- one built once and never updated, which needs no fork because a pure closure
% can capture it -- was measured and rejected: the policy ended 16 grid points from the first
% reference against 8 per iteration, and at nlocalsearch=16 the restricted search is 193 fine points
% against 214 for the unrestricted one, so there is no speed-up left to have.
%
% Inputs:
%   GEcondnsFn:    handle, p (column vector) -> general eqm conditions (vector)
%                  This is the expensive step (value fn, agent dist, aggregates).
%   ShootingMapFn: handle, (p,GEcondns) -> G(p), the plain shooting step. Takes
%                  the GE conditions as an input so that it does not have to
%                  re-evaluate them.
%   DistanceFn:    handle, GEcondns -> scalar distance. This is the convergence
%                  criterion, and is also what the safeguard compares.
%   VerifyFn:      handle, p -> (GEcondns,lsstate). An UNRESTRICTED evaluation at p, which also
%                  returns a fresh reference built from that exact sweep. Called only on convergence.
%   lsstate0:      the initial state. Pass [] and the first evaluation builds it.
%   p0:            initial guess (column vector, same space as GEcondnsFn takes)
%   tolerance:     convergence criterion, converged when DistanceFn()<tolerance
%   andersonoptions: relevant fields (defaults in brackets)
%      .memory      [5]    m, the Anderson memory/depth. Set by user.
%                          No automatic capping is done; if m is too large the
%                          least-squares problem becomes ill-conditioned
%                          (regularization keeps it solvable, and the safeguard
%                          keeps iterates safe, but frequent safeguard
%                          rejections are a sign m should be reduced).
%      .maxiter     [1000]
%      .warmup      [2]    number of initial plain (shooting) steps before
%                          Anderson steps begin
%      .regularization [1e-10] Tikhonov parameter in least-squares, RELATIVE to
%                          norm(DeltaF,'fro')^2 (so it is scale-free in the size
%                          of the fixed-point residual)
%      .safeguard   [1]    1: evaluate GE conditions at the Anderson trial point
%                          and reject the step if the distance increases, or the
%                          conditions are NaN/Inf, or the model itself throws
%                          (costs one extra evaluation of the GE conditions per
%                          Anderson step); 0: accept all
%                          Anderson steps (faster per iteration, no fallback
%                          protection)
%      .type        ['II'] Only 'II' is implemented here. 'I' is refused by name below: it was
%                          not forked, because its unbounded step backtracks toward the previous
%                          iterate and a backtracked trial is not cleanly separable from an accepted
%                          one, so the commit-on-accept rule for the reference has nowhere to live.
%                          Its options (powell_theta, restart_tau, alpha, safeguard_D, safeguard_eps,
%                          maxbacktrack) are therefore not read here either.
%      .verbose     [0]
%
% Outputs:
%   p:        the parameter vector at the (approximate) fixed point
%   GEcondns: the general eqm conditions evaluated at p
%   output:   struct with fields .iterations, .converged,
%             .residualpath, .nrejectedsteps, and .nverify / .nverifyfailed
%   lsstate:  the final state, so the caller can see the reference the solve ended on

%% Defaults
if ~isfield(andersonoptions,'memory'); andersonoptions.memory=5; end
if ~isfield(andersonoptions,'maxiter'); andersonoptions.maxiter=1000; end
if ~isfield(andersonoptions,'warmup'); andersonoptions.warmup=2; end
if ~isfield(andersonoptions,'regularization'); andersonoptions.regularization=1e-10; end
if ~isfield(andersonoptions,'safeguard'); andersonoptions.safeguard=1; end
if ~isfield(andersonoptions,'type'); andersonoptions.type='II'; end
if ~isfield(andersonoptions,'verbose'); andersonoptions.verbose=0; end

p=p0(:);
nP=length(p);

% Type-I is not forked. Its step is unbounded and backtracks toward the previous iterate, so a
% rejected trial and an accepted one are not cleanly separable and the commit-on-accept rule for the
% reference has no single place to live. Refused by name rather than left to run with a reference
% that silently tracks discarded trial points.
if strcmp(andersonoptions.type,'I')
    error('anderson.type=''I'' is not implemented with vfoptions.localsearch on a transition path; use the default type ''II'', or turn local search off')
end

% The Type-I implementation is NOT carried in this fork, only refused above. It used the pure
% GEcondnsFn(p) contract, and leaving ~110 lines of it behind a refusal is a trap: lift the refusal
% and the reference silently tracks backtracked trial points. See AndersonAcceleration for Type-I.

% History matrices: columns hold successive differences of iterates (DeltaX)
% and of fixed-point residuals (DeltaF). At most 'memory' columns are kept.
DeltaX=zeros(nP,0);
DeltaF=zeros(nP,0);
p_prev=[];
f_prev=[];

residualpath=nan(andersonoptions.maxiter,1);
nrejectedsteps=0;
nverify=0; nverifyfailed=0;
lsstate=lsstate0;
converged=0;
haveGEcondns=0; % =1 when GEcondns already holds the conditions at p (a safeguarded Anderson step that was accepted)

%% Main iteration
for iter=1:andersonoptions.maxiter

    % Evaluate general eqm conditions at current point (the expensive step:
    % involves solving the value function, agent distribution, aggregates)
    % When the safeguard accepted the Anderson step, it already evaluated the conditions at this p,
    % so reuse them rather than solving the model a second time at the same point. GEcondnsFn is a
    % function of p alone, so this changes nothing but the cost.
    if haveGEcondns==0
        % p is the current iterate and is being kept, so the state this builds is committed.
        [GEcondns,lsstate]=GEcondnsFn(p,lsstate);
        GEcondns=GEcondns(:);
    end
    haveGEcondns=0;

    if any(~isfinite(GEcondns))
        error(['AndersonAcceleration: GE conditions evaluated to NaN/Inf at ' ...
            'the current iterate (iteration %i). Try a different initial ' ...
            'guess, or smaller factors in howtoupdate.'],iter)
    end

    currentresid=DistanceFn(GEcondns);
    residualpath(iter)=currentresid;

    if andersonoptions.verbose==1
        fprintf('Anderson Acceleration: iteration %i, distance of GE condns=%8.6f \n',iter,currentresid)
    end

    % Check convergence
    if currentresid<tolerance
        % CERTIFY IT BEFORE BELIEVING IT. The residual that just passed came from a RESTRICTED
        % sweep, so re-solve unrestricted at these same prices. The prices are deliberately not
        % updated first, or the check would be of a different path.
        [GEcondns_v,lsstate_v]=VerifyFn(p);
        GEcondns_v=GEcondns_v(:);
        nverify=nverify+1;
        if all(isfinite(GEcondns_v)) && DistanceFn(GEcondns_v)<tolerance
            % Still converged under an exact sweep. Report the EXACT conditions rather than the
            % restricted ones, so the caller is handed what it would have got without local search.
            GEcondns=GEcondns_v;
            lsstate=lsstate_v;
            converged=1;
            break
        end
        % The exact sweep moved us back out of tolerance, so the restricted answer was not a
        % solution. Reseed the reference from the exact sweep and RESTART the history. The restart
        % is not optional: the verification evaluated at the SAME p, so keeping it in the history
        % would contribute a zero DeltaX against a non-zero DeltaF, which is the degenerate secant
        % pair the guards below exist to discard. Clearing the history is also exactly what the
        % safeguard does when it rejects a step, so this reuses that path rather than inventing one.
        % Execution then falls through: with an empty DeltaF the next step is a plain shooting step
        % from the exact residual, which is the right place to restart from.
        nverifyfailed=nverifyfailed+1;
        lsstate=lsstate_v;
        DeltaX=zeros(nP,0);
        DeltaF=zeros(nP,0);
        p_prev=[];
        f_prev=[];
        GEcondns=GEcondns_v;
        currentresid=DistanceFn(GEcondns);
        residualpath(iter)=currentresid; % overwrite the restricted value with the exact one
        if andersonoptions.verbose==1
            fprintf('   converged under local search at iteration %i, but an exact sweep gives %8.6f; reseeding the reference and restarting the history \n',iter,currentresid)
        end
    end

    % Plain shooting step g=G(p)
    g=ShootingMapFn(p,GEcondns);
    f=g-p;                 % f = G(p)-p, the fixed-point residual

    % Update the history of differences
    if ~isempty(p_prev)
        DeltaX=[DeltaX, p-p_prev]; %#ok<AGROW>
        DeltaF=[DeltaF, f-f_prev]; %#ok<AGROW>
        if size(DeltaF,2)>andersonoptions.memory
            DeltaX(:,1)=[]; % drop oldest column
            DeltaF(:,1)=[];
        end
    end
    p_prev=p;
    f_prev=f;

    %% Compute the next iterate
    tookandersonstep=0;
    if iter>andersonoptions.warmup && ~isempty(DeltaF)
        % Anderson (Type-II) step:
        % solve min_gamma || f - DeltaF*gamma ||_2, Tikhonov-regularized
        % THE TIKHONOV TERM IS RELATIVE TO THE SIZE OF DeltaF, not an absolute number. f is the
        % fixed-point residual of the shooting map, so f=factor*GEcondn and DeltaF is a difference of
        % those: with the small factors a stable shooting map needs, DeltaF'*DeltaF is routinely 1e-19
        % or smaller, so an absolute term of 1e-10 IS the normal equations and gamma comes back at
        % ~1e-6 of what it should be. That silently turns every Anderson step into the plain shooting
        % step it was supposed to accelerate - CoreStationaryGeneralEqm caught it as fminalgo=9 taking
        % the same number of iterations as fminalgo=5 while paying the safeguard's second model solve,
        % so it was strictly the slower of the two. Scaling by norm(DeltaF)^2 makes the option mean
        % what its name says, a conditioning safeguard, at any scale of f.
        mk=size(DeltaF,2);
        DeltaFscale=norm(DeltaF,'fro')^2;
        if DeltaFscale==0
            % Every stored difference is exactly zero, so f has stopped moving. DeltaF'*DeltaF is then
            % zero too and a zero Tikhonov term would make gamma NaN, where the absolute term used to
            % give gamma=0 and hence the plain shooting step. Keep that.
            DeltaFscale=1;
        end
        gamma=(DeltaF'*DeltaF + andersonoptions.regularization*DeltaFscale*eye(mk)) \ (DeltaF'*f);
        p_new=g-(DeltaX+DeltaF)*gamma;
        tookandersonstep=1;
    else
        p_new=g; % warmup: plain shooting step
    end

    %% Safeguard the Anderson step
    if andersonoptions.safeguard==1 && tookandersonstep==1
        % An extrapolated trial point can land where the model cannot be solved. That shows up two
        % ways: the conditions come back NaN/Inf, or the model THROWS - a ReturnFn evaluated outside
        % its domain errors rather than returning NaN (a negative base raised to a fractional power
        % is 'POWER: needs to return a complex result' inside gpuArray/arrayfun). Both mean the same
        % thing to the safeguard, that this step is no good, so the error is caught and the step
        % rejected below rather than being allowed to end the run.
        % THE STATE FROM A TRIAL POINT IS PROVISIONAL. This evaluation may be rejected below, and a
        % reference built at a point that is then discarded should not become the window centre for
        % the step actually taken: the reference wants to track the sequence of iterates the history
        % is built from. So it goes into lsstate_trial and is committed only where the step is
        % accepted. (Keeping it anyway would not be wrong -- it is still a nearby window centre, and
        % the window slides -- but it would make the solve depend on points it threw away.)
        try
            [GEcondns_trial,lsstate_trial]=GEcondnsFn(p_new,lsstate);
            GEcondns_trial=GEcondns_trial(:);
            stepfailed=any(~isfinite(GEcondns_trial));
        catch
            GEcondns_trial=inf(size(GEcondns)); % never read: the || below short-circuits on stepfailed
            lsstate_trial=lsstate;
            stepfailed=1;
        end
        if stepfailed || DistanceFn(GEcondns_trial)>currentresid
            % Reject the Anderson step: take the plain shooting step instead
            % and clear the history (restart)
            p_new=g;
            DeltaX=zeros(nP,0);
            DeltaF=zeros(nP,0);
            p_prev=[];
            f_prev=[];
            nrejectedsteps=nrejectedsteps+1;
            if andersonoptions.verbose==1
                if stepfailed
                    fprintf('   Anderson step rejected (GE conditions NaN/Inf, or the model could not be solved there); restarting from plain shooting step \n')
                else
                    fprintf('   Anderson step rejected (distance increased); restarting from plain shooting step \n')
                end
            end
        else
            % Accept the Anderson step, and keep the conditions just evaluated at it for the next iteration
            GEcondns=GEcondns_trial;
            lsstate=lsstate_trial; % committed: the trial point becomes the next iterate
            haveGEcondns=1;
        end
    end

    p=p_new;
end

%% Finish up
if converged==0
    % Loop ended by maxiter: re-evaluate GE conditions at the final iterate
    % (p was updated after the last evaluation inside the loop), unless the safeguard already did
    if haveGEcondns==0
        [GEcondns,lsstate]=GEcondnsFn(p,lsstate);
        GEcondns=GEcondns(:);
    end
    warning(['AndersonAcceleration: reached maxiter (%i) without convergence; ' ...
        'distance of GE condns=%8.6f. Consider increasing anderson.maxiter, ' ...
        'adjusting the howtoupdate factors, or reducing anderson.memory if ' ...
        'many steps were rejected (%i rejected).'], ...
        andersonoptions.maxiter,DistanceFn(GEcondns),nrejectedsteps)
end

output.iterations=iter;
output.converged=converged;
output.residualpath=residualpath(1:iter);
output.nrejectedsteps=nrejectedsteps;
output.nverify=nverify;             % unrestricted certification sweeps taken
output.nverifyfailed=nverifyfailed; % of those, how many moved us back out of tolerance

if andersonoptions.verbose==1 && converged==1
    fprintf('Anderson Acceleration: converged in %i iterations (%i Anderson steps rejected along the way) \n',iter,nrejectedsteps)
end

end
