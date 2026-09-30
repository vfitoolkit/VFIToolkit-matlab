function [PricePathOld,GEcondnPath]=TransitionPath_FHorz_quasiNewton(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_grid,a_grid,z_grid,pi_z, d_gridvals, aprime_gridvals,a_gridvals,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsStruct, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions)
% Damped Newton on the whole price path, with the Jacobian by brute-force finite differences and
% Broyden rank-one updates in between recomputes. Selected by transpathoptions.GEnewprice=1.
%
% Solves F(p)=0, where p is the price path for t=1..T-1 stacked into a vector and F is the general
% eqm conditions along that path. One evaluation of F is one call to TransitionPath_FHorz_singlepathiter,
% which costs the same as one iteration of the shooting algorithm.
%
% Assumes the number of general eqm eqns equals the number of prices, so the Jacobian is square.
% No permute of the eqns into price order is needed: J is the full Jacobian dF/dp, so J\f gives the
% right step as long as row i of J matches element i of f and column k matches element k of p, which
% it does by construction.
%
% The Jacobian is dense. Entries above the diagonal are anticipation, since the value function is
% solved backwards and so the policy at t responds to a price at any k>t. Entries below are
% persistence, since a perturbed distribution is carried forwards. That density is why a Newton step
% cannot be taken period by period.

%% Setup, the shapes of various of these objects vary depending on the setting
[PolicyIndexesPath,N_probs,II1,II2,exceptlastj,exceptfirstj,justfirstj]=TransitionPath_FHorz_substeps_Step0_setup(l_d,l_aprime,N_a,N_semiz,N_z,N_e,N_j,T,transpathoptions,vfoptions,simoptions);

% Semi-exogenous state: the form of pi_e_J the SemiExo value fn wants
pi_e_J_vf=pi_e_J;
if N_semiz>0
    if transpathoptions.fastOLG==1 && N_e>0 && N_z==0
        pi_e_J_vf=reshape(pi_e_J,[N_a*N_j,1,N_e]); % SemiExo value fn keeps the bothz dim, so needs (a,j)-by-1-by-e even when N_z==0
    end
end

nPrices=size(PricePathOld,2);
nF=(T-1)*nPrices;
if length(GeneralEqmEqnsCell)~=nPrices
    error('transpathoptions.GEnewprice=1 (Newton) needs one general eqm eqn per price, so that the Jacobian is square')
end

%% Residual at the starting price path
itercounter=1;
[GEcondnPath,AggVarsPath,PolicyIndexesPath]=TransitionPath_FHorz_singlepathiter(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj);
f=reshape(GEcondnPath,[],1);
% Two different norms, for two different jobs. fnorm is the merit function used to decide whether a
% step is an improvement: with J nonsingular, dx=-J\f gives (J'f)'dx=-||f||^2<0, so the Newton
% direction is guaranteed to descend the 2-norm, and only the 2-norm. GEcondnPathDist below is the
% L-Infinity-over-time convergence measure, which is the economics of every period clearing; the
% Newton direction carries no guarantee about it, so testing steps against it rejects good steps.
fnorm=norm(f);
if transpathoptions.multiGEcriterion==0
    GEcondnPathDist=max(sum(abs(transpathoptions.multiGEweights.*GEcondnPath),2));
elseif transpathoptions.multiGEcriterion==1
    GEcondnPathDist=max(sqrt(sum(transpathoptions.multiGEweights.*(GEcondnPath.^2),2)));
end

regfactor=1; % multiplier on the regularisation. Left at 1: escalating it turned out to do nothing once the Jacobian is well conditioned, which it is at the default epsprice. Kept as the hook in case it is wanted again
backtrackfactor=1; % the Newton step is multiplied by this. Halved whenever a step from a freshly built Jacobian makes things worse, which is overshoot on a nonlinear problem rather than a bad Jacobian
reinitwithoutprogress=0; % set when the Jacobian has been reinitialised and nothing has been accepted since, so a second failure stops rather than looping
stopreason='maxiter'; % how the loop was left, so the warning at the end can say which and give advice that fits it. Set at each break below
backtracking=0; % set while retrying the same direction with a shorter step, so the Jacobian is left exactly as it is
forceJacobian=0; % set when a step makes the residual worse, to trigger a rebuild of the Jacobian
Jacobianisfresh=0; % whether the Jacobian was built from scratch this iteration, rather than Broyden-updated
% 'LudwigStationary' and 'LudwigSSJ' both build from the final stationary eqm: period T of the price
% path, period T of the parameter path, V_final and AgentDist_initial, none of which move while the
% solve runs. So rebuilding either of them reproduces, entry for entry, the matrix already in hand.
% 'LudwigPath' and 'FullJacobian' both difference the CURRENT iterate, so their matrices genuinely
% change from one rebuild to the next and must never be replaced by a stored J0.
Jacobianisinvariant=strcmp(transpathoptions.GEnewprice1.Jacobianmethod,'LudwigStationary') || strcmp(transpathoptions.GEnewprice1.Jacobianmethod,'LudwigSSJ');
J0=[]; % the initial Jacobian, kept so that an invariant method can reinitialise to it for free
dx=zeros(nF,1,'gpuArray');
df=zeros(nF,1,'gpuArray');

%% Iterate
while GEcondnPathDist>transpathoptions.toleranceGEcondns && itercounter<=transpathoptions.maxiter

    p=reshape(PricePathOld(1:T-1,:),[],1);

    if backtracking==1
        % Retrying the same direction with a shorter step, so leave the Jacobian exactly as it is:
        % neither rebuilt (it is not the Jacobian that is wrong) nor Broyden-updated (the step that
        % would feed the update was rejected).
    elseif mod(itercounter-1,transpathoptions.t_updateJacobian)==0 || forceJacobian==1
        if Jacobianisinvariant==1 && ~isempty(J0)
            % Ludwig (2007) Table 1: GSQN reinitialises to J0, it does not re-evaluate. Here that is
            % not just cheaper but exactly equivalent, since a rebuild would return this same matrix.
            % Note this still discards the Broyden updates made since, which is the point of a restart.
            J=J0;
            if transpathoptions.verbose==1
                fprintf('Jacobian reinitialised at iteration %i to the one built at the start (Jacobianmethod=%s builds from the final stationary eqm, so recomputing it would give the same matrix) \n',itercounter,transpathoptions.GEnewprice1.Jacobianmethod)
            end
        elseif strcmp(transpathoptions.GEnewprice1.Jacobianmethod,'FullJacobian')
            % Brute-force Jacobian: one full path solve per column. Column k is the response of the
            % general eqm conditions at every period to perturbing one price in one period. mod(0,Inf)=0
            % and mod(k,Inf)=k, so t_updateJacobian=Inf recomputes only on the first iteration.
            if transpathoptions.GEnewprice1.FullJacobianReuseVpath==0
                J=zeros(nF,nF,'gpuArray');
                for k=1:nF
                    pk=p;
                    pk(k)=pk(k)+transpathoptions.epsprice;
                    PricePathk=PricePathOld;
                    PricePathk(1:T-1,:)=reshape(pk,T-1,nPrices);
                    GEcondnPathk=TransitionPath_FHorz_singlepathiter(PricePathk, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj);
                    J(:,k)=(reshape(GEcondnPathk,[],1)-f)/transpathoptions.epsprice;
                end
            else
                % The same matrix, built by restarting the backward pass at the perturbed period
                % rather than redoing all of it. V at a period cannot depend on a price at any
                % earlier period, so the later periods of the baseline solve are reused exactly and
                % this is not an approximation: both branches must give the same J.
                J=TransitionPath_FHorz_FullJacobiantriangular(p, f, nF, nPrices, PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj);
            end
            if transpathoptions.verbose==1
                % A column is all zero when perturbing that one price in that one period changed no
                % policy anywhere, which is what happens if transpathoptions.epsprice is too small for the
                % discretized choice. Enough zero columns and J is singular and the step is meaningless.
                fprintf('Jacobian at iteration %i (epsprice=%g): %i of %i columns are all zero, rcond=%g \n',itercounter,transpathoptions.epsprice,sum(all(J==0,1)),size(J,2),rcond(gather(J)))
            end
        elseif strcmp(transpathoptions.GEnewprice1.Jacobianmethod,'LudwigPath')
            % Ludwig's structure, but with Omega measured on the CURRENT price path rather than at a
            % steady state: perturb one price in every period at once and difference. Omega(i,j) is
            % then the average over t of the row-sum of the (i,j) time-block of the full Jacobian, so
            % it keeps each block's total sensitivity while the kron below assumes that sensitivity is
            % contemporaneous. Costs nPrices path solves against (T-1)*nPrices for 'FullJacobian'.
            Omega=zeros(length(GeneralEqmEqnsCell),nPrices);
            for jj=1:nPrices
                PricePathk=PricePathOld;
                PricePathk(1:T-1,jj)=PricePathk(1:T-1,jj)+transpathoptions.epsprice;
                GEcondnPathk=TransitionPath_FHorz_singlepathiter(PricePathk, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj);
                Omega(:,jj)=mean((GEcondnPathk-GEcondnPath),1)'/transpathoptions.epsprice;
            end
            J=gpuArray(kron(gather(Omega),eye(T-1)));
            if transpathoptions.verbose==1
                fprintf('LudwigPath Jacobian at iteration %i (epsprice=%g, Omega from perturbing the current price path): Omega=[%s], rcond=%g \n',itercounter,transpathoptions.epsprice,num2str(gather(Omega(:))',' %.4g'),rcond(gather(J)))
            end
        elseif strcmp(transpathoptions.GEnewprice1.Jacobianmethod,'LudwigStationary')
            % Ludwig (2007) GSQN. The Jacobian of a price path collapses to Omega kron I when the
            % model is time-invariant, because the partial derivatives then depend only on the lead or
            % lag and not on the period itself, so the whole T-by-T block is a scalar times the
            % identity. That leaves an nPrices-by-nPrices object rather than an (T-1)*nPrices square
            % one: 4 numbers instead of 198^2 here.
            %
            % Omega is the response of the general eqm conditions to a PERMANENT change in each price,
            % which is exactly the Jacobian of the stationary general eqm conditions, so it is taken at
            % the final stationary eqm and costs nPrices+1 stationary solves rather than any path
            % solves. Ludwig sec 3.2: W comes from the (fast) steady state calculation.
            Omega=TransitionPath_FHorz_LudwigWstationary(jequalOneDist,AgeWeightsParamNames,n_d,n_a,n_z,N_j,pi_z,d_grid,a_grid,z_grid,ReturnFn,FnsToEvaluate,GeneralEqmEqnsStruct,GEeqnNames,Parameters,DiscountFactorParamNames,PricePathNames,PricePathSizeVec,PricePathOld,ParamPathNames,ParamPathSizeVec,ParamPath,T,use_tminus1price,use_tminus1params,use_tplus1price,use_tminus1AggVars,use_stockvars,vfoptions,simoptions,transpathoptions);
            % p is stacked price-major (all periods of price 1, then of price 2), and so is f, so the
            % time-diagonal structure is exactly kron(Omega,eye(nT))
            J=gpuArray(kron(gather(Omega),eye(T-1)));
            if transpathoptions.verbose==1
                fprintf('LudwigStationary Jacobian at iteration %i (epsprice=%g, W from the final stationary eqm): Omega=[%s], rcond=%g \n',itercounter,transpathoptions.epsprice,num2str(gather(Omega(:))',' %.4g'),rcond(gather(J)))
            end
        elseif strcmp(transpathoptions.GEnewprice1.Jacobianmethod,'LudwigSSJ')
            % Ludwig with the cross-time structure kept instead of collapsed to the identity: the
            % initial matrix is the sequence-space Jacobian, which carries the anticipation (t<s) and
            % persistence (t>s) that both Ludwig methods throw away, and costs O(T) backward steps rather than
            % the (T-1)*nPrices path solves of 'FullJacobian'.
            %
            % It is a linearisation about the final stationary eqm, built by the fake-news algorithm over
            % the age-structured agent state (a,j,z). Its rows and columns are stacked equation-major and
            % price-major, the same layout as f and p here, so no reindexing is required.
            if use_tminus1price==1 || use_tplus1price==1 || use_tminus1params==1 || use_tminus1AggVars==1 || use_stockvars==1
                error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigSSJ'' cannot be used when the general eqm eqns or FnsToEvaluate refer to the previous or next period (t-1 or t+1 prices, parameters or aggregate variables, or stockvars); use ''LudwigPath'' or ''FullJacobian'' instead')
            end
            J=gpuArray(TransitionPath_FHorz_FakeNewsJacobian(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,N_j, N_d,N_a,N_z, l_d,l_aprime,l_a,l_z, d_gridvals,aprime_gridvals,a_gridvals,a_grid, z_gridvals_J,pi_z_J,pi_z_J_sim,semizze_gridvals_J_fastOLG, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, vfoptions, simoptions, transpathoptions));
            if transpathoptions.verbose==1
                fprintf('LudwigSSJ Jacobian at iteration %i: %i-by-%i, rcond=%g \n',itercounter,size(J,1),size(J,2),rcond(gather(J)))
            end
        else
            error('transpathoptions.GEnewprice1.Jacobianmethod must be ''LudwigPath'', ''LudwigStationary'', ''FullJacobian'' or ''LudwigSSJ''')
        end
        % Applies to every Jacobianmethod, not just the finite-differenced one: a non-finite Jacobian
        % otherwise travels silently into the linear solve and surfaces as an unrelated complaint from
        % lsqminnorm about the tolerance type.
        if ~all(isfinite(J(:)))
            error('TransitionPath_FHorz_quasiNewton: the Jacobian built by Jacobianmethod=''%s'' contains non-finite entries at iteration %i. For FullJacobian this usually means the price path has left the region where the model can be solved; for Ludwig or LudwigSSJ it usually means an aggregate used by the general eqm eqns came back as zero or non-finite',transpathoptions.GEnewprice1.Jacobianmethod,itercounter)
        end
        if Jacobianisinvariant==1 && isempty(J0)
            J0=J; % first build, so keep it: every later rebuild is a reset to this
        end
        Jacobianisfresh=1;
    else
        % Broyden rank-one update, from the step actually taken and the residual change it produced.
        % Skipped when the step was zero: dx'*dx would be zero and the update 0/0, which fills J with
        % NaN and takes everything after it with it.
        if dx'*dx>0
            J=J+((df-J*dx)*dx')/(dx'*dx);
        end
        Jacobianisfresh=0;
    end

    %% Damped Newton step
    PricePathprevious=PricePathOld; % kept so a step that makes things worse can be undone
    GEcondnPathprevious=GEcondnPath; % kept too, or a revert returns the rejected step's general eqm conditions alongside the accepted price path
    GEcondnPathDistold=GEcondnPathDist;
    fnormold=fnorm;
    % The Jacobian of a price path is badly conditioned: neighbouring columns are nearly parallel,
    % because a price change at period k and one at period k+1 have almost the same effect. A plain
    % J\f divides each direction by its own singular value, so the least-determined direction (which
    % holds mostly discretization and finite-difference noise) dominates the step. Both options below
    % stop that, and regfactor lets the trigger further down escalate them.
    Jc=gather(J); fc=gather(f); % the solve is on a nF-by-nF matrix, small enough that the cpu is fine
    if strcmp(transpathoptions.GEnewprice1.BroydenRegularisation,'minimumnorm')
        % Truncated SVD: singular values below the tolerance are dropped rather than divided by, so
        % the step only uses the directions the Jacobian actually resolves. regfactor=1 reproduces
        % the tolerance lsqminnorm would have used by itself.
        dx=-transpathoptions.GEnewprice1.factor*gpuArray(lsqminnorm(Jc,fc,regfactor*max(size(Jc))*eps(norm(Jc))));
    elseif strcmp(transpathoptions.GEnewprice1.BroydenRegularisation,'TikhonovRegularisation')
        % Same job done smoothly: 1/s_i becomes s_i/(s_i^2+lambda^2), so well determined directions
        % pass through unchanged and poorly determined ones fade out instead of being cut off. lambda
        % is taken relative to the largest singular value, so it does not depend on the units of the
        % prices or of the general eqm conditions.
        lambda=regfactor*transpathoptions.GEnewprice1.Tikhonovlambda*norm(Jc);
        dx=-transpathoptions.GEnewprice1.factor*gpuArray((Jc'*Jc+lambda^2*eye(size(Jc,1)))\(Jc'*fc));
    elseif strcmp(transpathoptions.GEnewprice1.BroydenRegularisation,'stepcap')
        % The crudest of the three: take the plain Newton step, but do not let it move the price path
        % by more than a fraction of the path's own length. That bounds the excursion, which is what
        % stops the model being handed prices it cannot evaluate. It only shortens the step though,
        % it does not change its direction: if the step points along a direction the Jacobian does
        % not resolve, a shorter step still points the same wrong way.
        dx=-transpathoptions.GEnewprice1.factor*gpuArray(Jc\fc);
        stepmax=transpathoptions.GEnewprice1.stepcap*norm(gather(p))/regfactor;
        if norm(dx)>stepmax
            dx=dx*(stepmax/norm(dx));
        end
    else
        error('transpathoptions.GEnewprice1.BroydenRegularisation must be ''minimumnorm'', ''TikhonovRegularisation'' or ''stepcap''')
    end
    if all(dx==0)
        % The regularisation has removed every direction, so no further progress is possible.
        stopreason='zerostep';
        break
    end
    dx=backtrackfactor*dx;
    p=p+dx;
    PricePathOld(1:T-1,:)=reshape(p,T-1,nPrices);

    %% Residual at the new price path
    fold=f;
    [GEcondnPath,AggVarsPath,PolicyIndexesPath]=TransitionPath_FHorz_singlepathiter(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj);
    f=reshape(GEcondnPath,[],1);
    fnorm=norm(f);
    df=f-fold;
    % How well did the Jacobian predict the residual change its own step just caused? If J is the
    % derivative of F, then df is J*dx to first order and this ratio goes to zero as the step
    % shrinks. If it stays of order one, J is simply not the derivative, and no choice of
    % regularisation, merit function or step length can fix that: the fault is in how J is built.
    % Note this is a much sharper question than rcond, which says how well conditioned J is, not
    % whether it is right. A smooth but badly biased Jacobian is well conditioned and wrong.
    if transpathoptions.verbose==1
        Jdx=J*dx;
        fprintf('Linearization error: ||df-J*dx||/||J*dx|| = %g   (near zero means J really is the derivative) \n',norm(df-Jdx)/norm(Jdx))
    end

    % Same convergence measure the shooting algorithm uses: scalarize the conditions within each
    % period, then the L-Infinity norm over time
    if transpathoptions.multiGEcriterion==0
        GEcondnPathDist=max(sum(abs(transpathoptions.multiGEweights.*GEcondnPath),2));
    elseif transpathoptions.multiGEcriterion==1
        GEcondnPathDist=max(sqrt(sum(transpathoptions.multiGEweights.*(GEcondnPath.^2),2)));
    end

    %% If the step made things worse, suspect the Jacobian rather than the step
    % Undo the step and rebuild the Jacobian from scratch. Jacobianisfresh stops this looping: a
    % Jacobian that was just rebuilt gets to take its step before it can trigger another rebuild.
    % Note a rebuild costs (T-1)*nPrices evaluations of the path, so this is not cheap.
    if fnorm>fnormold && Jacobianisfresh==0
        if transpathoptions.verbose==1
            warning('TransitionPath_FHorz_quasiNewton: at iteration %i the step increased the 2-norm of the general eqm conditions (from %g to %g). Undoing that step and recomputing the Jacobian from scratch, as the Broyden updates since it was last built are the likely cause',itercounter,fnormold,fnorm)
        end
        PricePathOld=PricePathprevious;
        GEcondnPath=GEcondnPathprevious;
        f=fold;
        fnorm=fnormold;
        GEcondnPathDist=GEcondnPathDistold;
        forceJacobian=1;
        backtracking=0; % a rebuild is coming, so this is not a backtracking retry
    elseif fnorm>fnormold % Jacobianisfresh==1
        % The Jacobian was built from scratch and its step still made things worse. On a well
        % conditioned Jacobian that is not a Jacobian problem, it is overshoot: the linear model is
        % fine locally and the step simply goes too far for a nonlinear problem. So halve the step
        % and try the same direction again, rather than rebuilding (198 path solves to reproduce the
        % same matrix at the same point, which just cycles) or regularising harder (which does
        % nothing when the singular values are healthy).
        PricePathOld=PricePathprevious;
        GEcondnPath=GEcondnPathprevious;
        f=fold;
        fnorm=fnormold;
        GEcondnPathDist=GEcondnPathDistold;
        if backtrackfactor<=2^(-transpathoptions.GEnewprice1.quasiNewton_reinitJacobian)
            % The line search has failed. Ludwig (2007) sec 3.3: because the Jacobian is not exact
            % there is no guarantee the direction descends, so reinitialise rather than persist.
            if reinitwithoutprogress==1
                stopreason='linesearch';
                break
            end
            if transpathoptions.verbose==1
                warning('TransitionPath_FHorz_quasiNewton: at iteration %i the line search failed after %i halvings, so the Jacobian is reinitialised rather than persisting with the direction',itercounter,transpathoptions.GEnewprice1.quasiNewton_reinitJacobian)
            end
            backtrackfactor=1;
            backtracking=0;
            forceJacobian=1;
            reinitwithoutprogress=1;
        else
            backtrackfactor=backtrackfactor/2;
            backtracking=1;
            forceJacobian=0;
        end
    else
        forceJacobian=0;
        backtrackfactor=1; % the step worked, so go back to taking the full Newton step
        backtracking=0;
        reinitwithoutprogress=0;
    end

    % Create plots of the transition path
    createTPathFeedbackPlots(PricePathNames,AggVarNames,GEeqnNames,PricePathOld,AggVarsPath,GEcondnPath,transpathoptions);

    if transpathoptions.verbose==1
        fprintf('Number of iterations on transition path (Newton): %i \n',itercounter)
        fprintf('Current distance of the general eqm conditions from zero: %8.6f \n', GEcondnPathDist)
        fprintf('2-norm of the general eqm conditions (the merit function the steps are judged on): %g \n', fnorm)
        fprintf('Size of the step just taken (2-norm of dx): %g, and dx''*dx=%g \n', norm(dx), dx'*dx)
        fprintf('Ratio of current distance to the convergence tolerance: %.2f (convergence when reaches 1) \n',GEcondnPathDist/transpathoptions.toleranceGEcondns)
    end

    itercounter=itercounter+1;
end

if GEcondnPathDist>transpathoptions.toleranceGEcondns
    % Two ways out of the loop that are not convergence, and they want opposite advice, so say which
    if strcmp(stopreason,'linesearch')
        warning(['TransitionPath_FHorz_quasiNewton: stopped at iteration %i of a maximum %i, ' ...
            'without convergence. It stopped because the line search failed on a Jacobian that had ' ...
            'just been reinitialised, so there was no shorter step and no fresher matrix left to ' ...
            'try. The general eqm conditions are %g from zero, against toleranceGEcondns=%g. ' ...
            'Increasing transpathoptions.maxiter will NOT help, since it stopped well short of the ' ...
            'one it had. Consider a different GEnewprice1.Jacobianmethod, a smaller ' ...
            'GEnewprice1.factor, or a toleranceGEcondns that is not tighter than the stationary ' ...
            'general eqm the path is anchored on.'], ...
            itercounter,transpathoptions.maxiter,GEcondnPathDist,transpathoptions.toleranceGEcondns)
    elseif strcmp(stopreason,'zerostep')
        warning(['TransitionPath_FHorz_quasiNewton: stopped at iteration %i of a maximum %i, ' ...
            'without convergence. It stopped because the regularised Newton step came out exactly ' ...
            'zero, so the regularisation had removed every direction the Jacobian could offer. The ' ...
            'general eqm conditions are %g from zero, against toleranceGEcondns=%g. Increasing ' ...
            'transpathoptions.maxiter will NOT help. Consider a different ' ...
            'GEnewprice1.BroydenRegularisation, or a larger transpathoptions.epsprice if the ' ...
            'Jacobian is close to singular.'], ...
            itercounter,transpathoptions.maxiter,GEcondnPathDist,transpathoptions.toleranceGEcondns)
    else
        warning(['TransitionPath_FHorz_quasiNewton: reached maxiter (%i) without convergence; ' ...
            'the general eqm conditions are %g from zero, against toleranceGEcondns=%g. Consider ' ...
            'increasing transpathoptions.maxiter, a different GEnewprice1.Jacobianmethod, or a ' ...
            'smaller GEnewprice1.factor.'], ...
            transpathoptions.maxiter,GEcondnPathDist,transpathoptions.toleranceGEcondns)
    end
end

end
