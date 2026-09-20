function [p_eqm_vec,GEcondns,output] = StationaryGeneralEqm_subcode_fminalgo9_AndersonAcceleration_PType(GeneralEqmConditionsFnOpt,p0,GeneralEqmEqns,GEPriceParamNames,GEpriceindexesB,heteroagentoptions,N_i,GEpriceindexes)
% Solves for stationary general eqm prices (permanent types) using Anderson Acceleration of
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
%   GeneralEqmEqns, GEPriceParamNames, GEpriceindexesB, N_i, GEpriceindexes: as
%      passed to the fminalgo=5 PType subcode; used to parse .fminalgo9.howtoupdate
%      (PType form) and to transform prices between the unconstrained and
%      original (constrained) spaces.
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
%      .anderson.type        ['II'] 'II' = classic Type-II (default); 'I' = the
%                                   stabilized Type-I AA-I-S-m of Zhang,
%                                   O'Donoghue & Boyd (2020), globally convergent
%                                   (uses .powell_theta, .restart_tau, .alpha,
%                                   .safeguard_D, .safeguard_eps)
%
% Outputs:
%   p_eqm_vec: the GE parameter vector at the (approximate) equilibrium
%   GEcondns:  the general eqm conditions evaluated at p_eqm_vec
%   output:    struct with fields .iterations, .converged,
%              .residualpath, .nrejectedsteps
% This command just sets up the three function handles (the general eqm conditions, the
% shooting map, the distance) and then calls the Anderson Acceleration core,
% AndersonAcceleration(), which is shared with the non-PType version of this command and
% with the transition path solver. One copy of the algorithm, so they cannot drift apart.

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
andersonoptions.verbose=heteroagentoptions.verbose;

% The base fixed-point map is the shooting map (fminalgo=5). Parse
% heteroagentoptions.fminalgo9.howtoupdate with the same code fminalgo=5 uses;
% this fills heteroagentoptions.fminalgo9 with permute/add/factor/keepold and
% sets heteroagentoptions.updateaccuracycutoff.
if ~isfield(heteroagentoptions,'fminalgo9') || ~isfield(heteroagentoptions.fminalgo9,'howtoupdate')
    error('fminalgo=9 (Anderson acceleration) requires heteroagentoptions.fminalgo9.howtoupdate (same format as fminalgo5.howtoupdate)')
end
heteroagentoptions=setupGEnewprice3_shooting(heteroagentoptions,GeneralEqmEqns,GEPriceParamNames,N_i,GEpriceindexes');
permute=heteroagentoptions.fminalgo9.permute(:);   % reorder GEcondns into GEPriceParamNames order
add=heteroagentoptions.fminalgo9.add(:);           % 1 -> add factor*condn, 0 -> subtract
factor=heteroagentoptions.fminalgo9.factor(:);     % step size per price
keepold=heteroagentoptions.fminalgo9.keepold(:);   % 0 only for factor=Inf (replace-old) rows
signedfactor=(2*add-1).*factor;                    % add.*factor - (1-add).*factor
updateaccuracycutoff=heteroagentoptions.updateaccuracycutoff;
transformindex=GEpriceindexesB; % the PType price indexes, in place of the 0:1:nGEParams the non-PType version uses

%% The three handles that the Anderson Acceleration core needs
% The shooting map, and the distance (which is both the convergence criterion
% and what the safeguard compares trial points on).
ShootingMapFn=@(p,GEc) AAI_shootingstep(p,GEc,permute,signedfactor,keepold,updateaccuracycutoff,transformindex,GEPriceParamNames,heteroagentoptions);
DistanceFn=@(GEc) max(abs(GEc));

[p_eqm_vec,GEcondns,output]=AndersonAcceleration(GeneralEqmConditionsFnOpt,ShootingMapFn,DistanceFn,p0,heteroagentoptions.toleranceGEcondns,andersonoptions);

end

function Phi=AAI_shootingstep(p,GEcondns,permute,signedfactor,keepold,updateaccuracycutoff,transformindex,GEPriceParamNames,heteroagentoptions)
% Plain shooting fixed-point step Phi(p) in unconstrained space: apply the
% howtoupdate rule in original price space, map back.
[p_orig,~]=ParameterConstraints_TransformParamsToOriginal(p',transformindex,GEPriceParamNames,heteroagentoptions);
p_i=GEcondns(permute);
p_i=(abs(p_i)>updateaccuracycutoff).*p_i;
p_orig_new=keepold.*p_orig'+signedfactor.*p_i;
Phi=ParameterConstraints_TransformParamsToUnconstrained(p_orig_new',transformindex,GEPriceParamNames,heteroagentoptions,0)';
end
