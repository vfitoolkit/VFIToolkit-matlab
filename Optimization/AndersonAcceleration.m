function [p,GEcondns,output]=AndersonAcceleration(GEcondnsFn,ShootingMapFn,DistanceFn,p0,tolerance,andersonoptions)
% Anderson Acceleration of a fixed-point map
%     p  <--  G(p)
% where G() is a 'shooting' style update built from the general eqm conditions.
% Anderson mixing combines the last m iterates to take a quasi-Newton-like
% step without any derivatives. A safeguard rejects any Anderson step that
% increases the residual (or produces NaN/Inf), falling back to a plain
% shooting step, so worst-case behaviour is that of the shooting algorithm.
%
% This is the shared core. Everything model-specific enters through the three
% function handles, so the same code serves both the stationary general eqm
% (fminalgo=9) and the transition path (transpathoptions.GEnewprice=2).
%
% Inputs:
%   GEcondnsFn:    handle, p (column vector) -> general eqm conditions (vector)
%                  This is the expensive step (value fn, agent dist, aggregates).
%   ShootingMapFn: handle, (p,GEcondns) -> G(p), the plain shooting step. Takes
%                  the GE conditions as an input so that it does not have to
%                  re-evaluate them.
%   DistanceFn:    handle, GEcondns -> scalar distance. This is the convergence
%                  criterion, and is also what the safeguard compares.
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
%      .regularization [1e-10] Tikhonov parameter in least-squares
%      .safeguard   [1]    1: evaluate GE conditions at the Anderson trial point
%                          and reject the step if the distance increases or
%                          evaluation fails (costs one extra evaluation of the
%                          GE conditions per Anderson step); 0: accept all
%                          Anderson steps (faster per iteration, no fallback
%                          protection)
%      .type        ['II'] 'II' = classic Type-II (default); 'I' = the
%                          stabilized Type-I AA-I-S-m of Zhang, O'Donoghue &
%                          Boyd (2020), globally convergent (uses
%                          .powell_theta, .restart_tau, .alpha, .safeguard_D,
%                          .safeguard_eps)
%      .maxbacktrack [10]  'I' only: the Type-I step is unbounded, so it can land where the
%                          model cannot be solved. This is how many times it may be halved back
%                          toward the previous iterate before giving up.
%      .verbose     [0]
%
% Outputs:
%   p:        the parameter vector at the (approximate) fixed point
%   GEcondns: the general eqm conditions evaluated at p
%   output:   struct with fields .iterations, .converged,
%             .residualpath, .nrejectedsteps

%% Defaults
if ~isfield(andersonoptions,'memory'); andersonoptions.memory=5; end
if ~isfield(andersonoptions,'maxiter'); andersonoptions.maxiter=1000; end
if ~isfield(andersonoptions,'warmup'); andersonoptions.warmup=2; end
if ~isfield(andersonoptions,'regularization'); andersonoptions.regularization=1e-10; end
if ~isfield(andersonoptions,'safeguard'); andersonoptions.safeguard=1; end
if ~isfield(andersonoptions,'type'); andersonoptions.type='II'; end
if ~isfield(andersonoptions,'verbose'); andersonoptions.verbose=0; end
if ~isfield(andersonoptions,'powell_theta'); andersonoptions.powell_theta=0.01; end   % Powell reg threshold, in (0,1)
if ~isfield(andersonoptions,'restart_tau'); andersonoptions.restart_tau=0.001; end     % restart independence threshold, in (0,1)
if ~isfield(andersonoptions,'alpha'); andersonoptions.alpha=1; end                     % KM averaging for the safe step, in (0,1]
if ~isfield(andersonoptions,'safeguard_D'); andersonoptions.safeguard_D=1e6; end        % safeguard constant D>0
if ~isfield(andersonoptions,'safeguard_eps'); andersonoptions.safeguard_eps=1e-6; end   % safeguard exponent eps>0
if ~isfield(andersonoptions,'maxbacktrack'); andersonoptions.maxbacktrack=10; end     % Type-I only: how many times a step may be halved back toward the previous iterate when the GE conditions come back non-finite

p=p0(:);
nP=length(p);

%% Type-I: stabilized AA-I-S-m (Zhang, O'Donoghue & Boyd 2020, Algorithm 3.1)
if strcmp(andersonoptions.type,'I')
    thetabar=andersonoptions.powell_theta; tau=andersonoptions.restart_tau;
    alpha=andersonoptions.alpha; Dsafe=andersonoptions.safeguard_D;
    epssafe=andersonoptions.safeguard_eps; mmax=andersonoptions.memory;
    residualpath=nan(andersonoptions.maxiter,1); nKMsteps=0; converged=0;
    % Counts of the robustness guards below, so that a run that came through cleanly can be told
    % apart from one that only survived because a guard caught it
    nbacktracks=0; nHfallbacks=0; nzerosecants=0; ndenomguards=0;

    % Initialization (line 2): residual g0=x0-Phi(x0); x^1=f_alpha(x^0)
    GEcondns=GEcondnsFn(p); GEcondns=GEcondns(:);
    if any(~isfinite(GEcondns)); error('AndersonAcceleration (Type-I): GE conditions NaN/Inf at the initial point.'); end
    g0=p-ShootingMapFn(p,GEcondns);
    Ubar=norm(g0);
    x_prev=p; g_prev=g0; x_tilde=p-alpha*g0; x_cur=x_tilde;
    H=eye(nP); Shat=zeros(nP,0); mc=0; nAA=0;

    for iter=1:andersonoptions.maxiter
        % residual at the trial x_tilde (line 5 needs g(x_tilde))
        % x_tilde is x_prev-H*g from the previous iteration, and nothing bounds H*g, so it can land
        % on a point where the model cannot be solved. x_prev is always a point whose conditions did
        % evaluate, so the step can be halved back toward it until the conditions come back finite.
        % Whenever x_cur is the same point it is carried along, which keeps the reuse below working.
        xcurwastilde=isequal(x_cur,x_tilde);
        if any(~isfinite(x_tilde))
            % H itself has gone non-finite, so there is no direction left to shorten. Fall back to a
            % plain shooting step from the last good iterate and reinitialise the Jacobian estimate.
            x_tilde=x_prev-alpha*g_prev;
            if xcurwastilde; x_cur=x_tilde; end
            H=eye(nP); Shat=zeros(nP,0); mc=0;
            nHfallbacks=nHfallbacks+1;
        end
        GEc_t=GEcondnsFn(x_tilde); GEc_t=GEc_t(:);
        nbacktrack=0;
        while any(~isfinite(GEc_t)) && nbacktrack<andersonoptions.maxbacktrack
            x_tilde=x_prev+(x_tilde-x_prev)/2;
            if xcurwastilde; x_cur=x_tilde; end
            nbacktrack=nbacktrack+1; nbacktracks=nbacktracks+1;
            GEc_t=GEcondnsFn(x_tilde); GEc_t=GEc_t(:);
        end
        if any(~isfinite(GEc_t)); error('AndersonAcceleration (Type-I): at iteration %i the general eqm conditions are still NaN/Inf after halving the step back toward the previous iterate %i times. Raise anderson.maxbacktrack, or start from a point further inside the region where the model can be solved.',iter,andersonoptions.maxbacktrack); end
        g_tilde=x_tilde-ShootingMapFn(x_tilde,GEc_t);
        % residual at x_cur (reuse if x_cur==x_tilde, i.e. previous step was accepted)
        if isequal(x_cur,x_tilde)
            GEc_k=GEc_t; g_k=g_tilde;
        else
            GEc_k=GEcondnsFn(x_cur); GEc_k=GEc_k(:);
            if any(~isfinite(GEc_k)); error('AndersonAcceleration (Type-I): the general eqm conditions are NaN/Inf at the current iterate at iteration %i (the safeguard accepted a step that cannot be evaluated).',iter); end
            g_k=x_cur-ShootingMapFn(x_cur,GEc_k);
        end
        currentresid=DistanceFn(GEc_k); residualpath(iter)=currentresid;
        if andersonoptions.verbose==1
            fprintf('Anderson Acceleration (Type-I): iteration %i, distance of GE condns=%8.6f \n',iter,currentresid)
        end
        if currentresid<tolerance; converged=1; break; end

        mc=mc+1;
        s=x_tilde-x_prev; y=g_tilde-g_prev;                       % secant pair (line 5)
        if s'*s==0
            % The iterate did not move, so there is no secant pair to learn from and shat would be
            % zero. Reinitialise rather than divide by zero in the Powell regularisation below.
            H=eye(nP); Shat=zeros(nP,0); mc=1;
            nzerosecants=nzerosecants+1;
        else
            shat=s;                                                % Gram-Schmidt (line 6)
            for jj=1:size(Shat,2); sj=Shat(:,jj); shat=shat-((sj'*s)/(sj'*sj))*sj; end
            if mc==mmax+1 || norm(shat)<tau*norm(s)                % restart checking (lines 7-8)
                mc=1; shat=s; H=eye(nP); Shat=zeros(nP,0);
            end
            gammaP=(shat'*H*y)/(shat'*shat);                      % Powell reg (lines 9-10)
            if abs(gammaP)>=thetabar
                theta=1;
            else
                sgn=sign(gammaP); if sgn==0; sgn=1; end
                theta=(1-sgn*thetabar)/(1-gammaP);
            end
            ytil=theta*y-(1-theta)*g_prev;
            denom=shat'*H*ytil;
            if denom==0 || ~isfinite(denom)
                % Nothing guards this denominator in the published algorithm, and a zero here turns
                % every entry of H into NaN. Reinitialise instead, and skip this secant pair.
                H=eye(nP); Shat=zeros(nP,0); mc=1;
                ndenomguards=ndenomguards+1;
            else
                H=H+((s-H*ytil)*(shat'*H))/denom;                 % rank-one update (line 11)
                Shat=[Shat, shat]; %#ok<AGROW>
            end
        end
        x_tilde_next=x_cur-H*g_k;
        if norm(g_k)<=Dsafe*Ubar*(nAA+1)^(-(1+epssafe))          % safeguard (lines 12-14)
            x_next=x_tilde_next; nAA=nAA+1;
        else
            x_next=x_cur-alpha*g_k; nKMsteps=nKMsteps+1;
        end
        x_prev=x_cur; g_prev=g_k; x_tilde=x_tilde_next; x_cur=x_next;
    end

    p=x_cur;
    if converged==1
        GEcondns=GEc_k;
    else
        GEcondns=GEcondnsFn(x_cur); GEcondns=GEcondns(:);
        warning('AndersonAcceleration (Type-I): reached maxiter (%i) without convergence; distance of GE condns=%8.6f (%i safeguard/KM steps).',andersonoptions.maxiter,DistanceFn(GEcondns),nKMsteps)
    end
    output.iterations=iter; output.converged=converged;
    output.residualpath=residualpath(1:iter); output.nrejectedsteps=nKMsteps;
    output.nbacktracks=nbacktracks; output.nHfallbacks=nHfallbacks;
    output.nzerosecants=nzerosecants; output.ndenomguards=ndenomguards;
    if nbacktracks>0 || nHfallbacks>0 || nzerosecants>0 || ndenomguards>0
        % Reported at any verbosity: each of these means the plain algorithm would have failed here.
        fprintf(['Anderson Acceleration (Type-I): robustness guards fired during this solve: ' ...
            '%i step halvings (the step reached a point where the general eqm conditions are not finite), ' ...
            '%i fallbacks to a plain step (H itself was not finite), ' ...
            '%i restarts on a zero secant pair, ' ...
            '%i restarts on a zero denominator in the rank-one update \n'], ...
            nbacktracks,nHfallbacks,nzerosecants,ndenomguards)
    end
    if andersonoptions.verbose==1 && converged==1
        fprintf('Anderson Acceleration (Type-I): converged in %i iterations (%i safeguard/KM steps) \n',iter,nKMsteps)
    end
    return
end

% History matrices: columns hold successive differences of iterates (DeltaX)
% and of fixed-point residuals (DeltaF). At most 'memory' columns are kept.
DeltaX=zeros(nP,0);
DeltaF=zeros(nP,0);
p_prev=[];
f_prev=[];

residualpath=nan(andersonoptions.maxiter,1);
nrejectedsteps=0;
converged=0;

%% Main iteration
for iter=1:andersonoptions.maxiter

    % Evaluate general eqm conditions at current point (the expensive step:
    % involves solving the value function, agent distribution, aggregates)
    GEcondns=GEcondnsFn(p);
    GEcondns=GEcondns(:);

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
        converged=1;
        break
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
        mk=size(DeltaF,2);
        gamma=(DeltaF'*DeltaF + andersonoptions.regularization*eye(mk)) \ (DeltaF'*f);
        p_new=g-(DeltaX+DeltaF)*gamma;
        tookandersonstep=1;
    else
        p_new=g; % warmup: plain shooting step
    end

    %% Safeguard the Anderson step
    if andersonoptions.safeguard==1 && tookandersonstep==1
        GEcondns_trial=GEcondnsFn(p_new);
        GEcondns_trial=GEcondns_trial(:);
        stepfailed=any(~isfinite(GEcondns_trial)); % NaN/Inf: extrapolated somewhere model breaks down
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
                    fprintf('   Anderson step rejected (GE conditions returned NaN/Inf); restarting from plain shooting step \n')
                else
                    fprintf('   Anderson step rejected (distance increased); restarting from plain shooting step \n')
                end
            end
        end
    end

    p=p_new;
end

%% Finish up
if converged==0
    % Loop ended by maxiter: re-evaluate GE conditions at the final iterate
    % (p was updated after the last evaluation inside the loop)
    GEcondns=GEcondnsFn(p);
    GEcondns=GEcondns(:);
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

if andersonoptions.verbose==1 && converged==1
    fprintf('Anderson Acceleration: converged in %i iterations (%i Anderson steps rejected along the way) \n',iter,nrejectedsteps)
end

end
