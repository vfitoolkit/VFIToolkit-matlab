function [p_eqm_vec,GEcondns,output] = StationaryGeneralEqm_subcode_fminalgo9_AndersonAcceleration(GeneralEqmConditionsFnOpt,p0,GeneralEqmEqns,GEPriceParamNames,nGEParams,heteroagentoptions)
% Solves for stationary general eqm prices using Anderson Acceleration of
% the fixed-point map that underlies the shooting algorithm (fminalgo=5):
%     p  <--  G(p),  the shooting update built from heteroagentoptions.fminalgo9.howtoupdate
%                    (same {GEeqnName,PriceName,add,factor} format as fminalgo5.howtoupdate)
% Anderson mixing combines the last m iterates to take a quasi-Newton-like
% step without any derivatives. A safeguard rejects any Anderson step that
% increases the residual (or produces NaN/Inf), falling back to a plain
% shooting step, so worst-case behaviour is that of the shooting algorithm.
%
% Note: p0 (and all internal iterates) are the transformed (unconstrained)
% parameters, the same space as the inputs to GeneralEqmConditionsFnOpt. The
% shooting update itself is applied in the original (constrained) price space
% and then mapped back (exactly as fminalgo=5 does), so when
% heteroagentoptions.constrainpositive/constrain0to1 are used the constrained
% parameters can never leave their feasible region.
%
% Inputs:
%   GeneralEqmConditionsFnOpt: handle, p (column vector) -> GE conditions (vector)
%   p0: initial GE parameter vector
%   GeneralEqmEqns, GEPriceParamNames, nGEParams: as passed to the fminalgo=5
%      subcode; used to parse .fminalgo9.howtoupdate and to transform prices
%      between the unconstrained and original (constrained) spaces.
%   heteroagentoptions: relevant fields (defaults in brackets)
%      .fminalgo9.howtoupdate  REQUIRED. {GEeqnName,PriceName,add,factor} cell,
%                                   one row per GE eqn (same format/role as
%                                   .fminalgo5.howtoupdate). Defines the base
%                                   shooting map that Anderson accelerates.
%      .toleranceGEcondns    [1e-4] convergence criterion on max(abs(GEcondns))
%      .verbose              [0]
%      .anderson.memory      [5]    m, the Anderson memory/depth. Set by user.
%                                   No automatic capping is done; if m is too
%                                   large the least-squares problem becomes
%                                   ill-conditioned (regularization keeps it
%                                   solvable, and the safeguard keeps iterates
%                                   safe, but frequent safeguard rejections
%                                   are a sign m should be reduced).
%      .anderson.maxiter     [1000]
%      .anderson.warmup      [2]    number of initial plain (shooting) steps
%                                   before Anderson steps begin
%      .anderson.regularization [1e-10] Tikhonov parameter in least-squares
%      .anderson.safeguard   [1]    1: evaluate GE conditions at the Anderson
%                                   trial point and reject the step if the
%                                   residual increases or evaluation fails
%                                   (costs one extra evaluation of the GE
%                                   conditions per Anderson step); 0: accept
%                                   all Anderson steps (faster per iteration,
%                                   no fallback protection)
%
% Outputs:
%   p_eqm_vec: the GE parameter vector at the (approximate) equilibrium
%   GEcondns:  the general eqm conditions evaluated at p_eqm_vec
%   output:    struct with fields .iterations, .converged,
%              .residualpath, .nrejectedsteps

%% Defaults
if ~isfield(heteroagentoptions,'toleranceGEcondns')
    heteroagentoptions.toleranceGEcondns=1e-4;
end
if ~isfield(heteroagentoptions,'verbose')
    heteroagentoptions.verbose=0;
end
if ~isfield(heteroagentoptions,'anderson')
    heteroagentoptions.anderson=struct();
end
andersonoptions=heteroagentoptions.anderson;
if ~isfield(andersonoptions,'memory'); andersonoptions.memory=5; end
if ~isfield(andersonoptions,'maxiter'); andersonoptions.maxiter=1000; end
if ~isfield(andersonoptions,'warmup'); andersonoptions.warmup=2; end
if ~isfield(andersonoptions,'regularization'); andersonoptions.regularization=1e-10; end
if ~isfield(andersonoptions,'safeguard'); andersonoptions.safeguard=1; end

% The base fixed-point map is the shooting map (fminalgo=5). Parse
% heteroagentoptions.fminalgo9.howtoupdate with the same code fminalgo=5 uses;
% this fills heteroagentoptions.fminalgo9 with permute/add/factor/keepold and
% sets heteroagentoptions.updateaccuracycutoff.
if ~isfield(heteroagentoptions,'fminalgo9') || ~isfield(heteroagentoptions.fminalgo9,'howtoupdate')
    error('fminalgo=9 (Anderson acceleration) requires heteroagentoptions.fminalgo9.howtoupdate (same format as fminalgo5.howtoupdate)')
end
heteroagentoptions=setupGEnewprice3_shooting(heteroagentoptions,GeneralEqmEqns,GEPriceParamNames);
permute=heteroagentoptions.fminalgo9.permute(:);   % reorder GEcondns into GEPriceParamNames order
add=heteroagentoptions.fminalgo9.add(:);           % 1 -> add factor*condn, 0 -> subtract
factor=heteroagentoptions.fminalgo9.factor(:);     % step size per price
keepold=heteroagentoptions.fminalgo9.keepold(:);   % 0 only for factor=Inf (replace-old) rows
signedfactor=(2*add-1).*factor;                    % add.*factor - (1-add).*factor
updateaccuracycutoff=heteroagentoptions.updateaccuracycutoff;

p=p0(:);
nP=length(p);

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
    GEcondns=GeneralEqmConditionsFnOpt(p);
    GEcondns=GEcondns(:);

    if any(~isfinite(GEcondns))
        error(['HeteroAgentStationaryEqm_AndersonAcceleration: GE conditions ' ...
            'evaluated to NaN/Inf at the current iterate (iteration %i). ' ...
            'Try a different initial guess, smaller damping, or use ' ...
            'constrainpositive/constrain0to1 on parameters that must be constrained.'],iter)
    end

    currentresid=max(abs(GEcondns));
    residualpath(iter)=currentresid;

    if heteroagentoptions.verbose==1
        fprintf('Anderson Acceleration: iteration %i, max(abs(GE condns))=%8.6f \n',iter,currentresid)
    end

    % Check convergence
    if currentresid<heteroagentoptions.toleranceGEcondns
        converged=1;
        break
    end

    % Plain shooting step g=G(p): apply the howtoupdate rule in original
    % (constrained) price space, then map back to unconstrained space, so the
    % base map is identical to fminalgo=5's shooting update.
    [p_orig,~]=ParameterConstraints_TransformParamsToOriginal(p',0:1:nGEParams,GEPriceParamNames,heteroagentoptions);
    p_i=GEcondns(permute);                       % reorder GEcondns into price order
    p_i=(abs(p_i)>updateaccuracycutoff).*p_i;    % accuracy cutoff (per shooting)
    p_orig_new=keepold.*p_orig'+signedfactor.*p_i;
    g=ParameterConstraints_TransformParamsToUnconstrained(p_orig_new',0:1:nGEParams,GEPriceParamNames,heteroagentoptions,0)'; % g = G(p)
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
        GEcondns_trial=GeneralEqmConditionsFnOpt(p_new);
        GEcondns_trial=GEcondns_trial(:);
        stepfailed=any(~isfinite(GEcondns_trial)); % NaN/Inf: extrapolated somewhere model breaks down
        if stepfailed || max(abs(GEcondns_trial))>currentresid
            % Reject the Anderson step: take the plain shooting step instead
            % and clear the history (restart)
            p_new=g;
            DeltaX=zeros(nP,0);
            DeltaF=zeros(nP,0);
            p_prev=[];
            f_prev=[];
            nrejectedsteps=nrejectedsteps+1;
            if heteroagentoptions.verbose==1
                if stepfailed
                    fprintf('   Anderson step rejected (GE conditions returned NaN/Inf); restarting from plain shooting step \n')
                else
                    fprintf('   Anderson step rejected (residual increased); restarting from plain shooting step \n')
                end
            end
        end
    end

    p=p_new;
end

%% Finish up
p_eqm_vec=p;
if converged==0
    % Loop ended by maxiter: re-evaluate GE conditions at the final iterate
    % (p was updated after the last evaluation inside the loop)
    GEcondns=GeneralEqmConditionsFnOpt(p);
    GEcondns=GEcondns(:);
    warning(['HeteroAgentStationaryEqm_AndersonAcceleration: reached maxiter (%i) ' ...
        'without convergence; max(abs(GE condns))=%8.6f. Consider increasing ' ...
        'anderson.maxiter, adjusting the fminalgo9.howtoupdate factors, or reducing ' ...
        'anderson.memory if many steps were rejected (%i rejected).'], ...
        andersonoptions.maxiter,max(abs(GEcondns)),nrejectedsteps)
end

output.iterations=iter;
output.converged=converged;
output.residualpath=residualpath(1:iter);
output.nrejectedsteps=nrejectedsteps;

if heteroagentoptions.verbose==1 && converged==1
    fprintf('Anderson Acceleration: converged in %i iterations (%i Anderson steps rejected along the way) \n',iter,nrejectedsteps)
end

end
