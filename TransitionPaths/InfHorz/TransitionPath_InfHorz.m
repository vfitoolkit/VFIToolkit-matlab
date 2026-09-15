function varargout=TransitionPath_InfHorz(PricePath0, ParamPath, T, V_final, AgentDist_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions, EntryExitParamNames)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes)
%
% PricePath0 is a structure with fields names being the Prices and each field containing a T-by-1 path. It is the initial guess for the PricePath.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.
%
% transpathoptions is not a required input.

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

if all(size(d_grid)==[prod(n_z),prod(n_z)])
    error('Check input order: pi_z comes after z_grid') % Keep this error message until end of 2007, can remove after that
end

%% Check which transpathoptions have been used, set all others to defaults
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    % If transpathoptions is not given, just use all the defaults
    transpathoptions.t_updateJacobian=Inf; % GEnewprice=1 only: recompute the Jacobian from scratch every t_updateJacobian iterations, with Broyden rank-one updates in between. =1 is a full Newton method, =Inf computes it once and relies on Broyden thereafter
    transpathoptions.GEnewprice1.Jacobianmethod='LudwigPath'; % How the initial/reinitialised Jacobian is built for GEnewprice=1. 'LudwigPath' is the Omega kron I structure of Ludwig (2007) GSQN with Omega measured by perturbing the current price path in every period, costing nPrices path solves per rebuild. 'LudwigStationary' is the same structure with Omega taken instead at the final stationary eqm, as Ludwig sec 3.2 prescribes, costing nPrices+1 stationary solves once and never rebuilt. 'FullJacobian' finite-differences every (period,price) at (T-1)*nPrices path solves and assumes no structure, so it is the oracle the other two are checked against. 'LudwigSSJ' uses the sequence-space Jacobian as the initial matrix, keeping the cross-time structure that the two Ludwig methods collapse to the identity
    if ~isfield(transpathoptions,'GEnewprice1') || ~isfield(transpathoptions.GEnewprice1,'quasiNewton_reinitJacobian')
        if strcmp(transpathoptions.GEnewprice1.Jacobianmethod,'FullJacobian')
            transpathoptions.GEnewprice1.quasiNewton_reinitJacobian=10; % how many times the step may be halved before giving up on the direction and reinitialising the Jacobian. Rebuilding the full Jacobian costs (T-1)*nPrices path solves, so it is worth backtracking a long way first
        else
            transpathoptions.GEnewprice1.quasiNewton_reinitJacobian=3; % Ludwig's prescription: three line-search failures then reinitialise, which is cheap when the Jacobian is nPrices path solves
        end
    end
    transpathoptions.GEnewprice1.factor=0.1; % pnew=p+factor*dp, the factor on 'dp' when updating prices in the Newton (with Broyden) algorithm; a dampening hyperparameter, the smaller this number the smaller the update each iteration
    transpathoptions.epsprice=1e-3; % Finite-difference step used to build the Jacobian (used by GEnewprice=1, and by the fake-news Jacobian when that arrives). Do NOT make this small: with discretized choice the residual is a step function of the prices, so a tiny perturbation moves no policy and the difference is discretization noise rather than a derivative. Measured on the d_z model of CoreInfHorzTPathAlgoTests, rcond of the Jacobian was 2.5e-6 at 1e-5 and 2.5e-3 at 1e-3, a thousand times better for the same cost. Do not use sqrt(eps): with discretized choice the policy is a step function of prices, so a machine-precision bump moves no policy at all
    transpathoptions.GEnewprice1.BroydenRegularisation='minimumnorm'; % GEnewprice=1 only: how to regularise the Newton step, since the Jacobian of a price path is badly conditioned. 'minimumnorm' drops the directions the Jacobian does not resolve, 'TikhonovRegularisation' fades them out instead
    transpathoptions.GEnewprice1.Tikhonovlambda=1e-4; % GEnewprice=1 only, and only used by BroydenRegularisation='TikhonovRegularisation': the regularisation weight, taken relative to the largest singular value of the Jacobian so that it does not depend on the units of the prices
    transpathoptions.GEnewprice1.stepcap=0.1; % GEnewprice=1 only, and only used by BroydenRegularisation='stepcap': the most the Newton step may move the price path in one iteration, as a fraction of the length of the path itself
    transpathoptions.GEnewprice1.FullJacobianReuseVpath=0; % GEnewprice=1 with Jacobianmethod='FullJacobian' only: =0 builds the Jacobian one full path solve per column. =1 restarts the backward pass at the perturbed period instead, since the value fn at a later period cannot depend on an earlier price, which is exact and roughly halves the value fn work, at the cost of holding the whole path of V in memory
    transpathoptions.updatepert=0; % 0: build the new price path from all periods at once after the loop over t (updatePricePathNew_TPath_T), 1: build it period by period inside the loop (updatePricePathNew_TPath_tt, the original). Same answer either way for the shooting algorithm; =0 is what the Newton options need, as their Jacobian couples periods
    transpathoptions.anderson=struct(); % GEnewprice=2 only: the Anderson acceleration options, all documented in AndersonAcceleration(). Defaults are set there, except for safeguard just below
    transpathoptions.anderson.safeguard=0; % GEnewprice=2 only: 0: take every Anderson step. 1: evaluate the general eqm conditions at the trial point too, and fall back to a plain shooting step if the distance got worse, which costs a second path solve on every Anderson iteration
    transpathoptions.toleranceGEprices=Inf; % convergence criterion for GE prices, set =Inf to turn this off (it is off by default)
    transpathoptions.toleranceGEcondns=1e-4; % convergence criterion for GE condns
    transpathoptions.multiGEcriterion=1; % How to combine multiple GE condns (default is sum-of-squares)
    transpathoptions.multiGEweights=ones(1,length(fieldnames(GeneralEqmEqns)));
    transpathoptions.updateaccuracycutoff=10^(-9); % If the suggested update is less than this then don't bother; 10^(-9) is decent odds to be numerical error anyway (currently only works for transpathoptions.GEnewprice=3)
    transpathoptions.parallel=1+(gpuDeviceCount>0);
    % transpathoptions.GEnewprice must be set explicitly for now: =1 is Newton with Broyden updates, =3 is the shooting algorithm.
    % There is deliberately no default while the Newton options are being built, because 1 and 3 want different things
    % from the user (3 needs GEnewprice3.howtoupdate) and silently picking one would be the wrong kind of convenience.
    % RESTORE A DEFAULT LATER. The old line was:
    % transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period separately), 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up
    error('transpathoptions.GEnewprice must be set: =1 for quasi-Newton with Broyden updates, =2 for Anderson acceleration, =3 for the shooting algorithm')
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiter=1000;
    transpathoptions.verbose=0;
    transpathoptions.graphpricepath=0; % 1: creates a graph of the 'current' price path which updates each iteration.
    transpathoptions.graphaggvarspath=0; % 1: creates a graph of the 'current' aggregate variables which updates each iteration.
    transpathoptions.graphGEcondns=0;  % 1: creates a graph of the 'current' general eqm conditions which updates each iteration.
    transpathoptions.historyofpricepath=0;
    transpathoptions.stockvars={}; % 'stockvars' are prices where you write '_tminus1' and it should cumulate (to there will be a general eqm eqn that relates the _tminus1 to the t for a price in PricePath)
    transpathoptions.weightsforpath=ones(T,length(GeneralEqmEqns)); % Won't actually be used under the defaults, but am still setting it.
    transpathoptions.tanimprovement=1;
else
    % Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'t_updateJacobian')
        transpathoptions.t_updateJacobian=Inf; % GEnewprice=1 only: recompute the Jacobian from scratch every t_updateJacobian iterations, with Broyden rank-one updates in between. =1 is a full Newton method, =Inf computes it once and relies on Broyden thereafter
    end
    if ~isfield(transpathoptions,'GEnewprice1') || ~isfield(transpathoptions.GEnewprice1,'Jacobianmethod')
        transpathoptions.GEnewprice1.Jacobianmethod='LudwigPath'; % How the initial/reinitialised Jacobian is built for GEnewprice=1. 'LudwigPath' is the Omega kron I structure of Ludwig (2007) GSQN with Omega measured by perturbing the current price path in every period, costing nPrices path solves per rebuild. 'LudwigStationary' is the same structure with Omega taken instead at the final stationary eqm, as Ludwig sec 3.2 prescribes, costing nPrices+1 stationary solves once and never rebuilt. 'FullJacobian' finite-differences every (period,price) at (T-1)*nPrices path solves and assumes no structure, so it is the oracle the other two are checked against. 'LudwigSSJ' uses the sequence-space Jacobian as the initial matrix, keeping the cross-time structure that the two Ludwig methods collapse to the identity
    end
    % transpathoptions.GEnewprice1.LudwigGeneralEqmEqns has no default. It is only needed by
    % Jacobianmethod='LudwigStationary', and only when the GeneralEqmEqns refer to the previous or next period
    % (t-1 or t+1 prices, parameters or aggregate variables), which have no meaning in the stationary
    % general eqm that Ludwig's W is built from. It is then a copy of GeneralEqmEqns written with
    % same-period names only, same eqn names in the same order. TransitionPath_InfHorz_LudwigW errors
    % with an explanation if it is needed and absent.
    if ~isfield(transpathoptions,'GEnewprice1') || ~isfield(transpathoptions.GEnewprice1,'quasiNewton_reinitJacobian')
        if strcmp(transpathoptions.GEnewprice1.Jacobianmethod,'FullJacobian')
            transpathoptions.GEnewprice1.quasiNewton_reinitJacobian=10; % how many times the step may be halved before giving up on the direction and reinitialising the Jacobian. Rebuilding the full Jacobian costs (T-1)*nPrices path solves, so it is worth backtracking a long way first
        else
            transpathoptions.GEnewprice1.quasiNewton_reinitJacobian=3; % Ludwig's prescription: three line-search failures then reinitialise, which is cheap when the Jacobian is nPrices path solves
        end
    end
    if ~isfield(transpathoptions,'GEnewprice1') || ~isfield(transpathoptions.GEnewprice1,'factor')
        transpathoptions.GEnewprice1.factor=0.1; % pnew=p+factor*dp, the factor on 'dp' when updating prices in the Newton (with Broyden) algorithm; a dampening hyperparameter, the smaller this number the smaller the update each iteration
    end
    if ~isfield(transpathoptions,'epsprice')
        transpathoptions.epsprice=1e-3; % Finite-difference step used to build the Jacobian (used by GEnewprice=1, and by the fake-news Jacobian when that arrives). Do NOT make this small: with discretized choice the residual is a step function of the prices, so a tiny perturbation moves no policy and the difference is discretization noise rather than a derivative. Measured on the d_z model of CoreInfHorzTPathAlgoTests, rcond of the Jacobian was 2.5e-6 at 1e-5 and 2.5e-3 at 1e-3, a thousand times better for the same cost. Do not use sqrt(eps): with discretized choice the policy is a step function of prices, so a machine-precision bump moves no policy at all
    end
    if ~isfield(transpathoptions,'GEnewprice1') || ~isfield(transpathoptions.GEnewprice1,'BroydenRegularisation')
        transpathoptions.GEnewprice1.BroydenRegularisation='minimumnorm'; % GEnewprice=1 only: how to regularise the Newton step, since the Jacobian of a price path is badly conditioned. 'minimumnorm' drops the directions the Jacobian does not resolve, 'TikhonovRegularisation' fades them out instead
    end
    if ~isfield(transpathoptions,'GEnewprice1') || ~isfield(transpathoptions.GEnewprice1,'Tikhonovlambda')
        transpathoptions.GEnewprice1.Tikhonovlambda=1e-4; % GEnewprice=1 only, and only used by BroydenRegularisation='TikhonovRegularisation': the regularisation weight, taken relative to the largest singular value of the Jacobian so that it does not depend on the units of the prices
    end
    if ~isfield(transpathoptions,'GEnewprice1') || ~isfield(transpathoptions.GEnewprice1,'stepcap')
        transpathoptions.GEnewprice1.stepcap=0.1; % GEnewprice=1 only, and only used by BroydenRegularisation='stepcap': the most the Newton step may move the price path in one iteration, as a fraction of the length of the path itself
    end
    if ~isfield(transpathoptions,'GEnewprice1') || ~isfield(transpathoptions.GEnewprice1,'FullJacobianReuseVpath')
        transpathoptions.GEnewprice1.FullJacobianReuseVpath=0; % GEnewprice=1 with Jacobianmethod='FullJacobian' only: =0 builds the Jacobian one full path solve per column. =1 restarts the backward pass at the perturbed period instead, since the value fn at a later period cannot depend on an earlier price, which is exact and roughly halves the value fn work, at the cost of holding the whole path of V in memory
    end

    if ~isfield(transpathoptions,'updatepert')
        transpathoptions.updatepert=0; % 0: build the new price path from all periods at once after the loop over t (updatePricePathNew_TPath_T), 1: build it period by period inside the loop (updatePricePathNew_TPath_tt, the original). Same answer either way for the shooting algorithm; =0 is what the Newton options need, as their Jacobian couples periods
    end
    if ~isfield(transpathoptions,'anderson')
        transpathoptions.anderson=struct(); % GEnewprice=2 only: the Anderson acceleration options, all documented in AndersonAcceleration(). Defaults are set there, except for safeguard just below
    end
    if ~isfield(transpathoptions.anderson,'safeguard')
        transpathoptions.anderson.safeguard=0; % GEnewprice=2 only: 0: take every Anderson step. 1: evaluate the general eqm conditions at the trial point too, and fall back to a plain shooting step if the distance got worse, which costs a second path solve on every Anderson iteration
    end
    if isfield(transpathoptions,'tolerance')
        error('Old transpathoptions.tolerance, has now been renamed and you should use transpathoptions.toleranceGEcondns instead')
    end
    if ~isfield(transpathoptions,'toleranceGEprices')
        transpathoptions.toleranceGEprices=Inf; % convergence criterion for GE prices, set =Inf to turn this off (it is off by default)
    end
    if ~isfield(transpathoptions,'toleranceGEcondns')
        transpathoptions.toleranceGEcondns=1e-4; % convergence criterion for GE condns
    end
    if ~isfield(transpathoptions,'multiGEcriterion')
        transpathoptions.multiGEcriterion=1;
    end
    if ~isfield(transpathoptions,'multiGEweights')
        transpathoptions.multiGEweights=ones(1,length(fieldnames(GeneralEqmEqns)));
    end
    if ~isfield(transpathoptions,'updateaccuracycutoff')
        transpathoptions.updateaccuracycutoff=10^(-9);
    end
    if ~isfield(transpathoptions,'parallel')
        transpathoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(transpathoptions,'GEnewprice')
        % transpathoptions.GEnewprice must be set explicitly for now: =1 is Newton with Broyden updates, =3 is the shooting algorithm.
        % There is deliberately no default while the Newton options are being built, because 1 and 3 want different things
        % from the user (3 needs GEnewprice3.howtoupdate) and silently picking one would be the wrong kind of convenience.
        % RESTORE A DEFAULT LATER. The old line was:
        % transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period separately), 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up
        error('transpathoptions.GEnewprice must be set: =1 for quasi-Newton with Broyden updates, =2 for Anderson acceleration, =3 for the shooting algorithm')
    end
    if ~isfield(transpathoptions,'oldpathweight')
        if transpathoptions.GEnewprice==3
            transpathoptions.oldpathweight=0; % user has to specify them as part of setup
        else
            transpathoptions.oldpathweight=0.9;
        end
    end
    if ~isfield(transpathoptions,'weightscheme')
        transpathoptions.weightscheme=1;
    end
    if ~isfield(transpathoptions,'Ttheta')
        transpathoptions.Ttheta=1;
    end
    if ~isfield(transpathoptions,'maxiter')
        transpathoptions.maxiter=1000;
    end
    if ~isfield(transpathoptions,'verbose')
        transpathoptions.verbose=0;
    end
    if ~isfield(transpathoptions,'graphpricepath')
        transpathoptions.graphpricepath=0; % 1: creates a graph of the 'current' price path which updates each iteration.
    end
    if ~isfield(transpathoptions,'graphaggvarspath')
        transpathoptions.graphaggvarspath=0; % 1: creates a graph of the 'current' aggregate variables which updates each iteration.
    end
    if ~isfield(transpathoptions,'graphGEcondns')
        transpathoptions.graphGEcondns=0;  % 1: creates a graph of the 'current' general eqm conditions which updates each iteration.
    end
    if ~isfield(transpathoptions,'historyofpricepath')
        transpathoptions.historyofpricepath=0;
    end
    if ~isfield(transpathoptions,'stockvars')
        transpathoptions.stockvars={}; % 'stockvars' are prices where you write '_tminus1' and it should cumulate (to there will be a general eqm eqn that relates the _tminus1 to the t for a price in PricePath)
    end
    if ~isfield(transpathoptions,'weightsforpath')
        transpathoptions.weightsforpath=ones(T,length(GeneralEqmEqns));
    end
    if ~isfield(transpathoptions,'tanimprovement')
        transpathoptions.tanimprovement=1;
    end
end
if transpathoptions.parallel~=2
    error('Transition paths can only be solved if you have a GPU')
end


%% Check which vfoptions have been used, set all others to defaults
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    % Model setup:
    vfoptions.exoticpreferences='None';
    vfoptions.experienceasset=0;
    % Exogenous shocks
    vfoptions.n_semiz=0;
    vfoptions.n_e=0;
    % Algorithm to use:
    vfoptions.solnmethod='purediscretization'; % Currently this does nothing
    vfoptions.divideandconquer=0;
    vfoptions.gridinterplayer=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    % Model setup:
    if ~isfield(vfoptions,'exoticpreferences')
        vfoptions.exoticpreferences='None';
    end
    if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        if ~isfield(vfoptions,'quasi_hyperbolic')
            vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
        elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            error('When using Quasi-Hyperbolic discounting vfoptions.quasi_hyperbolic must be either Naive or Sophisticated \n')
        end
    end
    if ~isfield(vfoptions,'experienceasset')
            vfoptions.experienceasset=0;
    end
    % Exogenous shocks
    if ~isfield(vfoptions,'n_semiz')
        vfoptions.n_semiz=0;
    end
    if ~isfield(vfoptions,'n_e')
        vfoptions.n_e=0;
    end
    % Algorithm to use:
    if ~isfield(vfoptions,'solnmethod')
        vfoptions.solnmethod='purediscretization'; % Currently this does nothing
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0;
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    elseif vfoptions.gridinterplayer==1
        if ~isfield(vfoptions,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1 you must set vfoptions.ngridinterp')
        end
    end
end

if vfoptions.divideandconquer==1
    if ~isfield(vfoptions,'level1n')
        if isscalar(n_a)
            vfoptions.level1n=floor(sqrt(n_a(1)));
            if n_a(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
        elseif length(n_a)==2
            vfoptions.level1n=[floor(sqrt(n_a(1))),n_a(2)]; % default DC2A: level1n(2)==n_a(2) triggers DC2A branch
            if n_a(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
        end
        if vfoptions.verbose==1
            fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
        end
    end
end
vfoptions.parallel=2; % GPU, has to be or transpath will already have thrown an error

%% Check which simoptions have been used, set all others to defaults
if exist('simoptions','var')==0
    simoptions.verbose=0;
    simoptions.tolerance=10^(-9);
    % Model setup
    simoptions.experienceasset=0;
    % Algorithm to use
    simoptions.gridinterplayer=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-9);
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    % Model setup
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    % Algorithm to use
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    elseif simoptions.gridinterplayer==1
        if ~isfield(simoptions,'ngridinterp')
            error('When using simoptions.gridinterplayer=1 you must set simoptions.ngridinterp')
        end
    end
end
simoptions.parallel=2; % GPU, has to be or transpath will already have thrown an error

%% Check the sizes of some of the inputs
N_d=prod(n_d);
% N_a=prod(n_a);
N_z=prod(n_z);

if N_d>0
    if any(size(d_grid)~=[sum(n_d), 1]) && any(size(d_grid)~=[prod(n_d), length(n_d)]) % stacked-column-grid or joint-grid
        fprintf('d_grid is of size: %i by % i, while sum(n_d) is %i \n',size(d_grid,1),size(d_grid,2),sum(n_d))
        error('d_grid is not the correct shape [should be stacked-column of size sum(n_d)-by-1), or a joint-grid of size prod(n_z)-by-length(n_z) ] \n')
    end
end
if any(size(a_grid)~=[sum(n_a), 1])
    fprintf('a_grid is of size: %i by % i, while sum(n_a) is %i \n',size(a_grid,1),size(a_grid,2),sum(n_a))
    error('a_grid is not the correct shape (should be of size sum(n_a)-by-1) \n')
% check z_grid below when converting to z_gridvals
elseif any(size(pi_z)~=[N_z, N_z])
    fprintf('pi is of size: %i by % i, while N_z is %i \n',size(pi_z,1),size(pi_z,2),N_z)
    error('pi is not of size N_z-by-N_z \n')
end
if length(fieldnames(PricePath0))~=length(fieldnames(GeneralEqmEqns))
    fprintf('PricePath has %i prices and GeneralEqmEqns is % i eqns \n',length(fieldnames(PricePath0)), length(fieldnames(GeneralEqmEqns)))
    error('Initial PricePath contains less variables than GeneralEqmEqns (structure) \n')
end

%% Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath0,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath0,ParamPath,T);

PricePathStruct=struct();

%%
% If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);
PricePath0=gpuArray(PricePath0);
ParamPath=gpuArray(ParamPath);
V_final=gpuArray(V_final);
% Tan improvement means we want agent dist on cpu
AgentDist_initial=gather(AgentDist_initial);


%% Check the sizes of some of the inputs
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_e>0
    error('Have not yet implemented i.i.d., e, shocks')
end

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_aprime=l_a;
if vfoptions.experienceasset>=1
    l_aprime=l_aprime-1;
end
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end
if N_e==0
    l_e=0;
else
    l_e=length(vfoptions.n_e);
end

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

%% Set up exogenous shock processes
[z_gridvals, pi_z, pi_z_sparse, e_gridvals, pi_e, pi_e_sparse, ze_gridvals, transpathoptions, simoptions]=ExogShockSetup_InfHorz_TPath(n_z,z_grid,pi_z,Parameters,PricePathNames,ParamPathNames,T,transpathoptions,simoptions,4);
% Convert z and e to joint-grids and transition matrix
% output: z_gridvals, pi_z, e_gridvals, pi_e, transpathoptions,vfoptions,simoptions

% Sets up
% transpathoptions.zpathtrivial=1; % z_gridvals and pi_z are not varying over the path
%                              =0; % they vary over path, so z_gridvals_T and pi_z_T
% transpathoptions.epathtrivial=1; % e_gridvals and pi_e are not varying over the path
%                              =0; % they vary over path, so e_gridvals_T and pi_e_T
% and
% transpathoptions.gridsinGE=1; % grids depend on a GE parameter and so need to be recomputed every iteration
%                           =0; % grids are exogenous
%
% transpathoptions.zepathtrivial=0 when either of zpathtrival and epathtrivial both are zero

%% If using any non-standard endogenous states, setup for those
[vfoptions,simoptions]=SetupNonStandardEndoStates_InfHorz_TPath(n_d,n_a,d_grid,a_grid,vfoptions,simoptions);

%% Setup for V_final
% Note: I keep Policy as having a first dimension (even if it is just 1)
if N_e==0
    if N_z==0
        V_final=reshape(V_final,[N_a,1]);
    else
        V_final=reshape(V_final,[N_a,N_z]);
    end
else
    if N_z==0
        V_final=reshape(V_final,[N_a,N_e]);
    else
        V_final=reshape(V_final,[N_a,N_z,N_e]);
    end
end

%% Setup for AgentDist_initial
if N_e==0  % no z, no e
    if N_z==0
        AgentDist_initial=reshape(AgentDist_initial,[N_a,1]);
    else % z, no e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,1]);
    end
else
    if N_z==0 % no z, e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_e,1]);
    else % z & e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z*N_e,1]);
    end
end


%% Change to FnsToEvaluate as cell so that it is not being recomputed all the time
l_daprime=l_d+l_a;
if vfoptions.experienceasset>=1
    l_daprime=l_daprime-1;
end

AggVarNames=fieldnames(FnsToEvaluate);
FnsToEvaluateCell=cell(1,length(AggVarNames));
for ff=1:length(AggVarNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
    if length(temp)>(l_daprime+l_a+l_z+l_e)
        FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluateCell{ff}=FnsToEvaluate.(AggVarNames{ff});
end
% Change FnsToEvaluate out of structure form, but want to still create AggVars as a structure
simoptions.outputasstructure=1;


%% Set up Gridvals (used by FnsToEvaluate, among others)
a_gridvals=CreateGridvals(n_a,a_grid,1); % a_gridvals is [N_a,l_a]

if N_d>0
    % Gridvals: switch to joint-grids
    if all(size(d_grid)==[sum(n_d),1]) % if stacked-column grid
        d_gridvals=CreateGridvals(n_d,gpuArray(d_grid),1);
    elseif all(size(d_grid)==[prod(n_d),length(n_d)]) % if joint-grid
        d_gridvals=gpuArray(d_grid);
    end
else
    d_gridvals=[];
end

if vfoptions.gridinterplayer==0
    aprime_gridvals=a_gridvals;
elseif vfoptions.gridinterplayer==1
    % use fine grid for aprime_gridvals
    if isscalar(n_a)
        n_aprime=n_a+(n_a-1)*vfoptions.ngridinterp;
        aprime_grid=interp1(gpuArray(1:1:N_a)',a_grid,gpuArray(linspace(1,N_a,n_aprime))');
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
    else
        a1_grid=a_grid(1:n_a(1));
        n_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
        n_aprime=[n_a1prime,n_a(2:end)];
        a1prime_grid=interp1(gpuArray(1:1:n_a(1))',a1_grid,gpuArray(linspace(1,n_a(1),n_a1prime))');
        aprime_grid=[a1prime_grid; a_grid(n_a(1)+1:end)];
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
    end
    vfoptions.policyind2val_finegridinput=1; % aprime_gridvals contains the fine grid for the first asset (tells PolicyInd2Val_InfHorz_TPath)
end

%% GE eqns, switch from structure to cell setup
GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);

GeneralEqmEqnsCell=cell(1,nGeneralEqmEqns);
for gg=1:nGeneralEqmEqns
    temp=getAnonymousFnInputNames(GeneralEqmEqns.(GEeqnNames{gg}));
    GeneralEqmEqnParamNames(gg).Names=temp;
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns.(GEeqnNames{gg});
end
% Now:
%  GeneralEqmEqns is still the structure
%  GeneralEqmEqnsCell is cell
%  GeneralEqmEqnParamNames(ff).Names contains the names


%% If using intermediateEqns, switch from structure to cell setup
transpathoptions.useintermediateEqns=0;
if isfield(transpathoptions,'intermediateEqns')
    transpathoptions.useintermediateEqns=1;
    intEqnNames=fieldnames(transpathoptions.intermediateEqns);
    nIntEqns=length(intEqnNames);

    transpathoptions.intermediateEqnsCell=cell(1,nIntEqns);
    for gg=1:nIntEqns
        temp=getAnonymousFnInputNames(transpathoptions.intermediateEqns.(intEqnNames{gg}));
        transpathoptions.intermediateEqnParamNames(gg).Names=temp;
        transpathoptions.intermediateEqnsCell{gg}=transpathoptions.intermediateEqns.(intEqnNames{gg});
    end
    % Now:
    %  transpathoptions.intermediateEqns is still the structure
    %  transpathoptions.intermediateEqnsCell is cell
    %  transpathoptions.intermediateEqnParamNames(gg).Names contains the names
end


%% If using a shooting algorithm, set that up
if transpathoptions.GEnewprice==2 && (~isfield(transpathoptions,'GEnewprice2') || ~isfield(transpathoptions.GEnewprice2,'howtoupdate'))
    % Anderson acceleration accelerates the shooting map, so it needs the same instructions the shooting algorithm does
    error('transpathoptions.GEnewprice=2 (Anderson acceleration) requires transpathoptions.GEnewprice2.howtoupdate (same format as GEnewprice3.howtoupdate)')
end
transpathoptions=setupGEnewprice3_shooting(transpathoptions,GeneralEqmEqns,PricePathNames);


%% Check if using _tminus1 and/or _tplus1 variables.
[tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames,ParamPathNames,{},transpathoptions);

% Following lines remove transpathoptions.stockvars from tminus1priceNames, and update use_tminus1price if necessary
if ~isempty(transpathoptions.stockvars)
    use_stockvars=1;
    stockvarsNames=transpathoptions.stockvars;
    transpathoptions=rmfield(transpathoptions,'stockvars');


    % how to find stockvars in PricePathNames
    stockvarsInPricePathNames=zeros(length(stockvarsNames),1); %% the pp index in PricePathNames that corresponds to each stockvar
    for kk=1:length(stockvarsNames)
        % throw error if the stockvar is not in PriceParamNames
        if ~any(strcmp(stockvarsNames{kk},PricePathNames))
            fprintf('Following error relates to stockvar: %s \n', stockvarsNames{kk})
            error('Cannot use a transpathoptions.stockvar which is not in PricePath')
        end
        % otherwise, find the matching index
        for pp=1:length(PricePathNames)
            if strcmp(stockvarsNames{kk},PricePathNames{pp})
                stockvarsInPricePathNames(kk)=pp;
            end
        end
    end

    % remove from stockvars from tminus1priceNames [stockvars have _tminus1 in name, but they 'cumulate' so have to be treated separately]
    for pp=1:length(stockvarsNames)
        if ~any(strcmp(stockvarsNames{pp},tminus1priceNames))
            error('transpathoptions.stockvars must appear as prices that are used with _tminus1')
        else
            tminus1priceNames(strcmp(tminus1priceNames, stockvarsNames{pp})) = [];
        end
    end
    if isempty(tminus1priceNames)
        use_tminus1params=0;
    end
else
    use_stockvars=0;
    stockvarsNames=[];
    stockvarsInPricePathNames=[];
end

%%
if transpathoptions.verbose>=1
    transpathoptions
end

if transpathoptions.verbose==2
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end


%% If there is entry and exit, then send to relevant command
if isfield(simoptions,'agententryandexit')==1 % isfield(transpathoptions,'agententryandexit')==1
    error('Have not yet implemented transition path for models with entry/exit \n')
end

%%
if transpathoptions.GEnewprice==1 % Damped Newton on the whole price path, Jacobian by brute-force finite differences with Broyden updates
    [PricePath,GEcondnPathmatrix]=TransitionPath_InfHorz_quasiNewton(PricePath0, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d,n_a,n_z,vfoptions.n_e, N_d,N_a,N_z,N_e, l_d,l_aprime,l_a,l_z,l_e, d_gridvals,aprime_gridvals,a_gridvals,a_grid,z_gridvals,e_gridvals,ze_gridvals,pi_z,pi_z_sparse,pi_e, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, stockvarsNames, stockvarsInPricePathNames, vfoptions, simoptions,transpathoptions);
elseif transpathoptions.GEnewprice==2 % Anderson acceleration of the shooting update
    [PricePath,GEcondnPathmatrix]=TransitionPath_InfHorz_Anderson(PricePath0, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d,n_a,n_z,vfoptions.n_e, N_d,N_a,N_z,N_e, l_d,l_aprime,l_a,l_z,l_e, d_gridvals,aprime_gridvals,a_gridvals,a_grid,z_gridvals,e_gridvals,ze_gridvals,pi_z,pi_z_sparse,pi_e, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, stockvarsNames, stockvarsInPricePathNames, vfoptions, simoptions,transpathoptions);
else % the shooting algorithm
    [PricePath,GEcondnPathmatrix]=TransitionPath_InfHorz_shooting(PricePath0, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d,n_a,n_z,vfoptions.n_e, N_d,N_a,N_z,N_e, l_d,l_aprime,l_a,l_z,l_e, d_gridvals,aprime_gridvals,a_gridvals,a_grid,z_gridvals,e_gridvals,ze_gridvals,pi_z,pi_z_sparse,pi_e, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, stockvarsNames, stockvarsInPricePathNames, vfoptions, simoptions,transpathoptions);
end

% Switch to structure for output
for pp=1:length(PricePathNames)
    PricePathStruct.(PricePathNames{pp})=PricePath(:,pp)';
end
for gg=1:length(GEeqnNames)
    GEcondnPath.(GEeqnNames{gg})=GEcondnPathmatrix(:,gg)';
end

if nargout==1
    varargout={PricePathStruct};
elseif nargout==2
    varargout={PricePathStruct,GEcondnPath};
end


end
