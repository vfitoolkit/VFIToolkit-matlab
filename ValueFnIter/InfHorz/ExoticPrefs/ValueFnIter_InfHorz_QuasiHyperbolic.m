function varargout=ValueFnIter_InfHorz_QuasiHyperbolic(V0, n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames)
% (last two entries of) DiscountFactorParamNames contains the names for the two parameters relating to
% Quasi-hyperbolic preferences.
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the
% future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j: Vtilde= u_t+ beta0beta *E[V_{j+1}]
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the
% time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j: Vhat=u_t+beta0beta*E[Vundberbar_{j+1}]
% See documentation for a fuller explanation of this.
%
% Outputs are returned via varargout. Caller may request:
%   Naive:         [V, Policy, Valt, Policyalt]
%                  V         = Vtilde     (QH-discounted, used for the QH-optimal choice)
%                  Policy    = QH-optimal choice (argmax of Vtilde)
%                  Valt      = V_std      (std-discounted continuation, computed at Policyalt)
%                  Policyalt = std-optimal choice (argmax of V_std). Naive needs Policy AND
%                  Policyalt to reconstruct V/Valt from policy alone.
%   Sophisticated: [V, Policy, Valt]
%                  V       = Vhat       (QH-discounted from current self's perspective)
%                  Policy  = equilibrium choice
%                  Valt    = Vunderbar  (realised continuation under future selves' own QH choices)
%                  Policy alone suffices to reconstruct V and Valt.

N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);

if ~isfield(vfoptions,'quasi_hyperbolic')
    vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    % Sophisticated quasi-hyperbolic in infinite horizon does not converge tightly, and the reason
    % is not a grid too coarse: the policy CYCLES rather than settling. Measured in the test bank at
    % n_a=251, 53% of states change policy on every iteration at beta0=0.90, and 26% still do at
    % beta0=0.99, with 0 of 50 iterations settling. Howards cannot fix that (it accelerates policy
    % EVALUATION at a fixed policy), so the iteration stops on maxiter rather than on tolerance and
    % a smaller default saves time that buys nothing.
    if ~isfield(vfoptions,'maxiter')
        vfoptions.maxiter=1000;
    end
    % This is a DEFAULT, not an override. It used to be set unconditionally, which silently
    % discarded a maxiter the user had chosen -- running longer than the user asked is a worse
    % failure than running long.
    %
    % vfoptions.verbose is likewise no longer forced to 1 here. It was, with the stated intention
    % that you could watch whether currdist was cycling through numbers of the same magnitude; but
    % no quasi-hyperbolic raw read it, so that print never happened and the forcing only overrode a
    % user's choice. The raws now take and honour it (and print the policy-change count beside
    % currdist, which is what actually distinguishes cycling from slow convergence). Either way, a
    % run that stops on maxiter also warns at the end, whatever verbose is set to.
elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    % Check that one of the possible options have been used. If not then error.
    error('vfoptions.quasi_hyperbolic must be either Naive or Sophisticated (check spelling and capital letter)')
end

if ~isfield(vfoptions,'QHadditionaldiscount')
    error('You must declare vfoptions.QHadditionaldiscount when using quasi-hyperbolic discouting (you have vfoptions.exoticpreferences set to QuasiHyperbolic)')
elseif ~ischar(vfoptions.QHadditionaldiscount)
    error('vfoptions.QHadditionaldiscount must be the name of the additional discount parameter, given as a character vector such as ''beta0''')
end

if ~isscalar(n_a)
    error('InfHorz quasi-hyperbolic discounting is only implemented for a single endogenous state (scalar n_a). Two endogenous states (2A) are not supported on ANY quasi-hyperbolic path: the raws take prod(n_a) as N_a and have no a1/a2 split.')
end
% This guard used to sit inside the vfoptions.gridinterplayer==1 block, so the grid-interpolation
% case was refused cleanly while the plain and refined paths fell through to fail deep inside a raw
% with 'In repelem(A,N1,...,Nn), Nj must be a scalar or a vector of length size(A,j)'. Same outcome,
% unrecognisable message. It is an interface restriction, so it belongs with the interface checks.

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

% Most branches below return Kron-form V and Policy for the tail of this file to reshape and
% UnKron. The Naive grid-interpolation branch does not: it calls ValueFnIter_InfHorz_GridInterpLayer
% and ValueFnIter_Case1_FHorz, both of which return final-form output already. The flag keeps the
% tail from unkronning something twice.
alreadyfinalform=0;

%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
beta0=Parameters.(vfoptions.QHadditionaldiscount);
if ~isscalar(beta0)
    error('The quasi-hyperbolic additional discount factor (the parameter named by vfoptions.QHadditionaldiscount) must be a scalar; it cannot depend on age')
end

%%
if vfoptions.gridinterplayer==1
    % Default GI path (postGI, howardsgreedy=0, howardssparse=0). d-refinement is
    % preference-independent, so the refined return-matrix construction is identical
    % to the standard postGI raws; only the VFI/extra step uses QH discounting.
    % QH GI is always postGI — mirror the postGI defaults from ValueFnIter_InfHorz_GridInterpLayer.
    % This dispatcher bypasses that helper, so the defaults must be set here.
    if ~isfield(vfoptions,'multigridswitch')
        vfoptions.multigridswitch=10;
    end
    % postGI looks locally for the optimal grid interpolation point, in the area around the point that was optimal on the coarse grid.
    % By considering the nearest 'maxaprimediff' points on the original coarse grid the hope is that we set the 'local' sweep wide enough to ensure we catch the global optimum. By repeating this process 'postGIrepeat' times, we can hopefully take steps towards the global optimum of the fine grid problem.
    if ~isfield(vfoptions,'postGIrepeat')
        vfoptions.postGIrepeat=0;
        % In practice, the local optima appears to get stuck in a 'local basin', and so setting postGIrepeat>0 fails to achieve anything because each repeat remains stuck in the basin and does not get any closer to the global. This is not true for small values of maxaprimediff, but at the default value of maxaprimediff anything other than postGIrepeat=0 seems pointless. And setting a larger maxaprimediff gives much more robust behaviour than pairing a smaller maxaprimediff with postGIrepeat>0
        % We therefore set a default of postGIrepeat=0. This can be increased but in tests ran there was nothing to be gained from doing so: without a decision variable postGI was already exact (even for maxaprimediff as small as 3), and with a decision variable the conservatively large default maxaprimediff already reaches the global optimum.
    end
    if ~isNaive && ~isfield(vfoptions,'maxaprimediff')
        % NO DEFAULT FOR SOPHISTICATED. There is no value that is right across calibrations, and a
        % wrong one is silent rather than loud, so the user must choose.
        % Measured in CoreInfHorzTests/withQuasiHyperbolicDiscounting on 2026-10-21, at n_a=251,
        % as the width the window actually needs (taken on an unconstrained whole-grid search, so
        % not censored by the window's own clamp):
        %        beta0=0.90   beta0=0.80
        %   nod        53           71
        %   with d     61           74
        % A 0.10 fall in beta0 costs 34% and 21% more window at fixed n_a, because the width the
        % window must span is the quasi-hyperbolic-vs-exponential POLICY GAP and that widens as
        % beta0 falls. So no n_a-proportional rule can be right for more than one beta0.
        % Auto-detection was considered and rejected: a detector can only see the argmax sitting ON
        % the window boundary, which proves the window is too small, while an interior argmax
        % proves nothing unless the objective is single-peaked -- and with d refined out the return
        % matrix is a max over d of concave functions, which need not be concave. The detector
        % would return another lower bound dressed up as a guarantee.
        % Sophisticated also cannot do what Naive does (search the whole fine grid in one shot),
        % because its quasi-hyperbolic argmax sits INSIDE the iteration rather than being one extra
        % step after it.
        % This mirrors vfoptions.ngridinterp, which likewise has no default and errors when the
        % grid interpolation layer is on (ValueFnIter_InfHorz.m).
        % Note the format-args form of error(): a single-argument error() does NOT process escapes,
        % so the \n would print literally. That exact slip is live elsewhere in the toolkit.
        error('%s\n\n%s\n\n%s\n\n%s', ...
            'With vfoptions.quasi_hyperbolic=''Sophisticated'' and vfoptions.gridinterplayer=1 you must set vfoptions.maxaprimediff yourself. There is no default, because no single value is right across calibrations and a value that is too small fails silently: the quasi-hyperbolic choice is pinned at the edge of the search window and a wrong answer is returned with no error.', ...
            'A reasonable starting point is ceil(n_a/(beta0*4)). The window must span the gap between the quasi-hyperbolic and exponential policies, so it must grow as n_a grows, and it must grow as beta0 FALLS - a smaller beta0 moves the quasi-hyperbolic choice further from the exponential one.', ...
            'Then CHECK IT, and do not expect the check to be easy: the objective need not be single-peaked (with a decision variable the return matrix is a max over d of concave functions, which need not be concave), so the answer need not change monotonically as the window widens, and finding that one larger value gives the same answer does NOT prove the window is wide enough. Vary it over a range, not once.', ...
            'Note also that Sophisticated with the grid interpolation layer is separately limited at beta0<1 by the kinked-Vunderbar collapse: the continuation being interpolated is one criterion evaluated at another criterion''s argmax, so it is kinked, and linear interpolation of it collapses the fine argmax onto an interval endpoint however wide the window is. A wide enough window is necessary here, not sufficient.')
    end
    if ~isfield(vfoptions,'maxaprimediff')
        % Only Naive can reach here: Sophisticated either errored just above or set the field itself.
        % For Naive the window is now used ONLY by the step-1 exponential solve below, which is the
        % ordinary ValueFnIter_InfHorz_GridInterpLayer call. The quasi-hyperbolic step itself has no
        % window at all -- it goes through ValueFnIter_Case1_FHorz at N_j=1, and nothing on the FHorz
        % side reads vfoptions.maxaprimediff. So the default here is exactly the exponential
        % discounting default, mirrored from ValueFnIter_InfHorz_GridInterpLayer.
        % (Before the N_j=1 rewrite the quasi-hyperbolic step re-searched this same window, centred
        % on the exponential policy, so it needed a width of the order of the quasi-hyperbolic-vs-
        % exponential policy gap and this default had to be much wider. That requirement is gone.)
        if prod(n_d)==0 && length(n_a)==1
            % The small default is only safe when a1prime is the sole choice. As soon as something else
            % is chosen jointly with it (a decision variable d, or a second endogenous asset a2prime)
            % the a1prime objective can be flat or multimodal conditional on that other choice, and the
            % coarse argmax can then sit a long way from the fine one, which is what the window is
            % centred on.
            vfoptions.maxaprimediff=5; % only used for postGI (for vfoptions.preGI=0)
        else
            if n_a(1)<300
                vfoptions.maxaprimediff=ceil(n_a(1)/5);
            else
                vfoptions.maxaprimediff=ceil(n_a(1)/10);
            end
            if vfoptions.verbose_advice==1
                warning('vfoptions.postGI=1 is not guaranteed to converge globally. The default of vfoptions.maxaprimediff is set to a conservatively large value to hopefully achieve global convergence. You can try higher/lower values, and see if the solution is sensitive.')
            end
            % Based on testing models, the default vfoptions.maxaprimediff values set here were sufficient to always acheive global convergence. But this does not guarantee they are for all other models.
            % These defaults are also so conservative that the 'post' step takes more memory than the original coarse grid step. The vast majority of models will solve just fine for the global solution with lower values of vfoptions.maxaprimediff, and will be faster and use less memory.
        end
    end
    if N_d==0
        if isNaive
            % Step 1: the exponential solve, by the standard postGI path.
            [Valt, Policyalt]=ValueFnIter_InfHorz_GridInterpLayer(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, prod(DiscountFactorParamsVec), ReturnFnParamsVec, vfoptions);

            % Step 2: the quasi-hyperbolic step is EXACTLY a one-period finite-horizon problem,
            % max over aprime of u + beta0*beta*E[Valt]. The FHorz backward loop is
            % 'for reverse_j=1:N_j-1', so at N_j=1 it runs zero times and the whole solve is the
            % V_Jplus1 block: one step, which is what this is. That buys divide-and-conquer plus
            % the grid interpolation layer from code that is already implemented and tested, with
            % no search window and therefore no vfoptions.maxaprimediff -- whose required width is
            % the quasi-hyperbolic-vs-exponential policy gap, widens as beta0 falls (measured 53
            % rising to 71 for a 0.10 fall at n_a=251), and so has no safe default. A window too
            % narrow pins the choice at its edge and is silently wrong.
            vfoptions_QHstep=vfoptions;
            vfoptions_QHstep.exoticpreferences='None'; % else FHorz dispatches straight back into its own quasi-hyperbolic path
            vfoptions_QHstep.divideandconquer=1;
            vfoptions_QHstep.outputkron=0; % the InfHorz caller may have set this, and FHorz honours it
            vfoptions_QHstep.V_Jplus1=Valt;
            Parameters_QHstep=Parameters;
            Parameters_QHstep.QHstepdiscount=beta0*prod(DiscountFactorParamsVec);
            % z_gridvals is the [prod(n_z),length(n_z)] joint form, which is one of the shapes
            % ExogShockSetup_FHorz documents and accepts, so it needs no conversion here.
            [V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,1,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,Parameters_QHstep,{'QHstepdiscount'},ReturnFnParamNames,vfoptions_QHstep);
            V=reshape(V,[n_a,n_z]); % drop the trailing N_j=1 dimension
            Policy=reshape(Policy,[size(Policy,1),n_a,n_z]);
            alreadyfinalform=1; % Valt/Policyalt and V/Policy above are all already unkronned
        else
            [V, Policy, Valt, Policyalt]=ValueFnIter_InfHorz_QuasiHyperbolicS_postGI_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, beta0, ReturnFn, ReturnFnParamsVec, vfoptions);
        end
    else
        if isNaive
            % Step 1: the exponential solve, by the standard postGI path.
            [Valt, Policyalt]=ValueFnIter_InfHorz_GridInterpLayer(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, prod(DiscountFactorParamsVec), ReturnFnParamsVec, vfoptions);

            % Step 2: as in the N_d==0 arm above -- the quasi-hyperbolic step is exactly a
            % one-period finite-horizon problem, so solve it with the FHorz command at N_j=1 and
            % vfoptions.V_Jplus1=Valt.
            vfoptions_QHstep=vfoptions;
            vfoptions_QHstep.exoticpreferences='None'; % else FHorz dispatches straight back into its own quasi-hyperbolic path
            vfoptions_QHstep.divideandconquer=1;
            vfoptions_QHstep.outputkron=0; % the InfHorz caller may have set this, and FHorz honours it
            vfoptions_QHstep.V_Jplus1=Valt;
            Parameters_QHstep=Parameters;
            Parameters_QHstep.QHstepdiscount=beta0*prod(DiscountFactorParamsVec);
            % z_gridvals is the [prod(n_z),length(n_z)] joint form, which is one of the shapes
            % ExogShockSetup_FHorz documents and accepts, so it needs no conversion here.
            [V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,1,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,Parameters_QHstep,{'QHstepdiscount'},ReturnFnParamNames,vfoptions_QHstep);
            V=reshape(V,[n_a,n_z]); % drop the trailing N_j=1 dimension
            Policy=reshape(Policy,[size(Policy,1),n_a,n_z]);
            alreadyfinalform=1; % Valt/Policyalt and V/Policy above are all already unkronned
        else
            [V, Policy, Valt, Policyalt]=ValueFnIter_InfHorz_QuasiHyperbolicS_Refine_postGI_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, beta0, ReturnFn, ReturnFnParamsVec, vfoptions);
        end
    end
elseif N_d>0 && strcmp(vfoptions.solnmethod,'purediscretization_refinement')
    % Default GPU path for d-case: refine d out of the return function, then solve
    % the QH problem on the refined (nod-shaped) matrix. See subfn for the argument
    % that d-refinement is preference-independent.
    [V, Policy, Valt, Policyalt]=ValueFnIter_InfHorz_QuasiHyperbolic_Refine(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, ReturnFnParamsVec, DiscountFactorParamsVec, beta0, vfoptions, isNaive);
else
    % Use the same ReturnMatrix for both the continuation value, and the value fn
    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals,ReturnFnParamsVec,0);

    if isNaive % For Naive, just solve the standard value function problem, and then just one step following that.
        if N_d==0
            % First calculate the exponential discounting solution
            % prod(n_a), prod(n_z): unlike ValueFnIter_InfHorz_raw (which takes n_d,n_a,n_z and
            % reduces them itself), ValueFnIter_InfHorz_nod_raw takes N_a and N_z already reduced
            % and uses them directly in a colon expression. Passing n_a,n_z here was harmless only
            % because a single endogenous state and a single z make them scalars already; with a
            % two-dimensional z it reaches '0:1:N_z-1' as a vector and errors.
            if vfoptions.relativevfi==0
                [Valt,Policyalt]=ValueFnIter_InfHorz_nod_raw(V0, prod(n_a), prod(n_z), pi_z, prod(DiscountFactorParamsVec), ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            else % endogenous VFI (relative VFI when there is no z)
                [Valt,Policyalt]=ValueFnIter_InfHorz_EndogenousVFI_nod_raw(V0, prod(n_a), prod(n_z), pi_z, prod(DiscountFactorParamsVec), ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
            % Then the Naive quasi-hyperbolic from this
            [V,Policy]=ValueFnIter_InfHorz_QuasiHyperbolicN_nod_raw(Valt, n_a, n_z, pi_z, DiscountFactorParamsVec, beta0, ReturnMatrix);
            Policy=shiftdim(Policy,-1);
        else
            % First calculate the exponential discounting solution
            if vfoptions.relativevfi==0
                [Valt, Policyalt]=ValueFnIter_InfHorz_raw(V0, n_d,n_a,n_z, pi_z, prod(DiscountFactorParamsVec), ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
            else % endogenous VFI (relative VFI when there is no z)
                [Valt, Policyalt]=ValueFnIter_InfHorz_EndogenousVFI_raw(V0, n_d,n_a,n_z, pi_z, prod(DiscountFactorParamsVec), ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
            end
            % Then the Naive quasi-hyperbolic from this
            [V, Policy]=ValueFnIter_InfHorz_QuasiHyperbolicN_raw(Valt, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, beta0, ReturnMatrix);
        end
    else % Sophisticated
        if N_d==0
            [V,Policy,Valt]=ValueFnIter_InfHorz_QuasiHyperbolicS_nod_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, beta0, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter, vfoptions.verbose);
            Policy=shiftdim(Policy,-1);
        else
            [V, Policy,Valt]=ValueFnIter_InfHorz_QuasiHyperbolicS_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, beta0, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter, vfoptions.verbose);
        end
    end
end


%%
if alreadyfinalform==0
    V=reshape(V,[n_a,n_z]);
    Valt=reshape(Valt,[n_a,n_z]);
    if N_d==0
        Policy=UnKronPolicyIndexes1_z(Policy, n_a, n_a, n_z, vfoptions);
        if isNaive
            Policyalt=UnKronPolicyIndexes1_z(Policyalt, n_a, n_a, n_z, vfoptions);
        end
    else
        Policy=UnKronPolicyIndexes2_z(Policy, n_d, n_a, n_a, n_z, vfoptions);
        if isNaive
            Policyalt=UnKronPolicyIndexes2_z(Policyalt, n_d, n_a, n_a, n_z, vfoptions);
        end
    end
end

if isNaive
    varargout={V, Policy, Valt, Policyalt};
else
    varargout={V, Policy, Valt, []};
end

end
