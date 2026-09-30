function Jp=TransitionPath_FHorz_FakeNewsJacobian(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,N_j, N_d,N_a,N_z, l_d,l_aprime,l_a,l_z, d_gridvals,aprime_gridvals,a_gridvals,a_grid, z_gridvals_J,pi_z_J,pi_z_J_sim,semizze_gridvals_J_fastOLG, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, vfoptions, simoptions, transpathoptions)
% The sequence-space Jacobian of the general eqm conditions of an FHorz transition path, with respect to
% the price path, by the fake-news algorithm of Auclert, Bardoczy, Rognlie & Straub (2021). It is the
% initial Jacobian of transpathoptions.GEnewprice1.Jacobianmethod='LudwigSSJ', and the FHorz counterpart
% of MEP_SSJ_Jacobians_fakenewsalgo (which is InfHorz only).
%
% Jp is (T-1)*nGEeqns-by-(T-1)*nPrices, stacked the way TransitionPath_FHorz_quasiNewton stacks f and p:
% rows (gg-1)*(T-1)+(1:T-1) are general eqm eqn gg over time, columns (ip-1)*(T-1)+(1:T-1) are price ip.
%
% It is a linearisation about the FINAL stationary eqm: period T of the price path and of the parameter
% path, V_final, and the stationary agent distribution that goes with them.
%
% The agent state is x=(a,j,z), in the fastOLG ordering, and the population is the whole age profile. A
% finite horizon changes nothing about the fake-news argument: at the steady state the one-period maps
% are time-invariant, so the Jacobian of the household block is still built from
%   - backward pass: for a price change at anticipation horizon s, the date-0 policy response, obtained
%     by iterating the steady-state value fn step s times on the perturbed continuation value. From it
%       curlyY[s](o) = change in aggregate o at date 0     (distribution held at the steady state)
%       curlyD[s]    = change in the date-1 distribution
%   - expectation vectors: E^o_t(x) = E[y_o(x_t)|x_0=x] weighted by the age weights, E^o_0=AgeWeights.*y_o,
%     E^o_t = Lambda' E^o_{t-1}, where Lambda is the linear part of the one-period distribution map
%   - fake-news matrix: F[0,s]=curlyY[s], F[t,s]=E^o_{t-1}'curlyD[s] (t>=1)
%   - Jacobian recursion: J[t,s]=J[t-1,s-1]+F[t,s]
% What is particular to FHorz is only Lambda. The distribution map is affine rather than linear: ages
% 1..N_j-1 move to ages 2..N_j by the policy and pi_z, and age 1 is refilled by jequalOneDist, which does
% not depend on the prices. So a change in the distribution never touches age 1, and Lambda is the
% ageing step alone; its adjoint takes an expectation vector at ages 2..N_j back to ages 1..N_j-1, and
% nobody continues past N_j. The distribution is normalised to one within each age, with the age weights
% applied at aggregation (as everywhere on the FHorz path), which is why they sit in E^o_0.
%
% Everything except Lambda is done with the tested transition-path substeps (the single-step value fn,
% Step2 AdjustPolicy, Step3tt IterAgentDist and Step4tt AggVars), called on a one-period 'path', so the
% policy decoding, the grid interpolation layer and the aggregation are exactly the ones the residual uses.

%% What this cannot do
% The same scope as MEP_SSJ_Jacobians_fakenewsalgo: a standard endogenous state with z shocks and no e.
if ~(N_z>0 && prod(vfoptions.n_e)==0 && prod(vfoptions.n_semiz)==0)
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigSSJ'' requires z shocks and no e or semi-exogenous shocks; use ''LudwigPath'' or ''FullJacobian'' instead')
end
if vfoptions.experienceasset>=1 || vfoptions.experienceassetz>=1
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigSSJ'' is not implemented for experience assets; use ''LudwigPath'' or ''FullJacobian'' instead')
end
if ~strcmp(vfoptions.exoticpreferences,'None')
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigSSJ'' is not implemented for exotic preferences; use ''LudwigPath'' or ''FullJacobian'' instead')
end
if simoptions.fastOLG==0
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigSSJ'' requires simoptions.fastOLG=1 (the default); use ''LudwigPath'' or ''FullJacobian'' instead')
end
if transpathoptions.zpathtrivial==0
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigSSJ'' is not implemented when the z grid or its transition matrix vary over the transition path; use ''LudwigPath'' or ''FullJacobian'' instead')
end
if any(PricePathSizeVec(2,:)~=PricePathSizeVec(1,:))
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigSSJ'' needs every price to be a scalar (not age-dependent); use ''LudwigPath'' or ''FullJacobian'' instead')
end
if transpathoptions.useintermediateEqns==1
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigSSJ'' does not yet allow for intermediateEqns, which the general eqm eqns would depend on through the aggregates; use ''LudwigPath'' or ''FullJacobian'' instead')
end

nT=T-1;
nP=length(PricePathNames);
nAgg=length(AggVarNames);
nG=length(GEeqnNames);
if nG~=nP
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigSSJ'' needs one general eqm eqn per price')
end
epsprice=transpathoptions.epsprice;
N=N_a*N_j*N_z; % the agent state (a,j,z)
M=N_a*(N_j-1)*N_z; % the agent state at ages 1..N_j-1, which is what the ageing step moves

%% The steady state: period T of the price and parameter paths
Psteady=Parameters;
for pp=1:nP
    Psteady.(PricePathNames{pp})=PricePathOld(T,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp));
end
for pp=1:length(ParamPathNames)
    Psteady.(ParamPathNames{pp})=ParamPath(T,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp));
end
if transpathoptions.trivialjequalonedist==0
    jequalOneDistSS=jequalOneDist(:,end); % jequalOneDist is then jequalOneDist_T
else
    jequalOneDistSS=jequalOneDist;
end
AgeWeightsSS=AgeWeights_T(:,T); % [N_a*N_j*N_z,1], the fastOLG form

% A one-period 'path', so that the substeps can be used as they are. Step0 sizes PolicyIndexesPath for
% periods 1..T-1, so T=2 gives exactly the one period.
[PolicyIndexesPath1,N_probs,II1,II2,exceptlastj,exceptfirstj,justfirstj]=TransitionPath_FHorz_substeps_Step0_setup(l_d,l_aprime,N_a,0,N_z,0,N_j,2,transpathoptions,vfoptions,simoptions);

% Steady-state policy: one value fn step from V_final at the steady-state prices
if transpathoptions.fastOLG==0
    [~,Policy_ss]=ValueFnIter_FHorz_TPath_SingleStep(V_final,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Psteady, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    [~,Policy_ss]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG(V_final,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Psteady, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end
PolicyIndexesPath1(:,:,:,:,1)=Policy_ss;
[PolicyForDist_ss,PolicyProbs_ss,PolicyValues_ss]=TransitionPath_FHorz_substeps_Step2_AdjustPolicy(PolicyIndexesPath1,2,Psteady,n_d,n_a,n_z,0,N_j,l_d,l_aprime,N_a,N_z,0,N_probs,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions);

% Steady-state agent distribution. With a finite horizon the whole population is replaced within N_j
% periods, so N_j steps of the steady-state policy from any starting distribution give the stationary one
% exactly (AgentDist_initial is the initial distribution, which only coincides with the final one for a
% null reform).
D_ss=AgentDist_initial;
for jj=1:N_j
    D_ss=TransitionPath_FHorz_substeps_Step3tt_IterAgentDist(D_ss,PolicyForDist_ss,PolicyProbs_ss,1,N_a,N_z,0,N_j,N_probs,pi_z_J,pi_z_J_sim,[],[],II1,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDistSS,transpathoptions,simoptions);
end

% Steady-state aggregates
AggVars_ss=TransitionPath_FHorz_substeps_Step4tt_AggVars(D_ss,AgeWeightsSS,PolicyValues_ss(:,:,:,:,1),1,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Psteady,N_j,l_d,l_aprime,l_a,0,l_z,0,N_d,N_a,0,N_z,0,a_gridvals,semizze_gridvals_J_fastOLG,transpathoptions);

%% Per-state outcomes at the steady state, weighted by the age weights: these are E^o_0
% EvalFnOnAgentDist_Grid_J works on (a,z,j), so the policy values are permuted there and the result is
% permuted back to the fastOLG (a,j,z). The z grid is taken from semizze_gridvals_J_fastOLG, [1,N_j,N_z,l_z],
% which is the form the aggregation itself uses, and turned into the [N_z,l_z,N_j] that Grid_J wants.
PolicyValuesPermute=permute(PolicyValues_ss(:,:,:,:,1),[1,3,2,4]); % [N_a,N_z,N_j,l_d+l_aprime]
zgridvals_Grid_J=permute(semizze_gridvals_J_fastOLG,[3,4,2,1]); % [N_z,l_z,N_j]
Evec=cell(nAgg,1);
yweighted=zeros(N,nAgg);
for oo=1:nAgg
    if isempty(FnsToEvaluateParamNames(oo).Names)
        ParamCell=cell(0,1);
    else
        FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Psteady, FnsToEvaluateParamNames(oo).Names,N_j);
        ParamCell=cell(size(FnToEvaluateParamsAgeMatrix,2),1);
        for ii=1:size(FnToEvaluateParamsAgeMatrix,2)
            ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-2)}; % (a,z,j), so j is after N_a and N_z
        end
    end
    Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluateCell{oo},ParamCell,PolicyValuesPermute,l_d+l_aprime,n_a,n_z,a_gridvals,zgridvals_Grid_J); % [N_a,N_z,N_j]
    y=gather(reshape(permute(reshape(Values,[N_a,N_z,N_j]),[1,3,2]),[N,1])).*gather(AgeWeightsSS);
    y(isnan(y))=0; % NaN only where the state is infeasible, which has no mass
    yweighted(:,oo)=y;
    % Guard: these per-state outcomes must reproduce the aggregate the path itself computes. If they do
    % not, the (a,j,z) orderings disagree somewhere, and the Jacobian would be silently wrong.
    aggcheck=sum(y.*gather(D_ss));
    if abs(aggcheck-gather(AggVars_ss.(AggVarNames{oo}).Mean))>1e-8*max(1,abs(aggcheck))
        error('TransitionPath_FHorz_FakeNewsJacobian: the per-state outcomes for %s aggregate to %g, but the transition path aggregates them to %g, so the two do not use the same ordering of the agent state',AggVarNames{oo},aggcheck,gather(AggVars_ss.(AggVarNames{oo}).Mean))
    end
end

%% Lambda, the ageing step of the distribution at the steady state
% Exactly the linear part of AgentDist_FHorz_TPath_SingleStep_IterFast_raw (and its nProbs twin): the
% distribution at ages 1..N_j-1 is moved by the policy (Gammatranspose) and then by pi_z_J_sim, which is
% block-diagonal in j and so also does the ageing, and lands on ages 2..N_j.
if N_probs==1
    Gammatranspose=sparse(gather(PolicyForDist_ss(:,1)),II1,II2,M,M);
else
    Gammatranspose=sparse(gather(PolicyForDist_ss(:,:,1)),II1,gather(PolicyProbs_ss(:,:,1)),M,M);
end
pi_z_J_simtranspose=gather(pi_z_J_sim)';
exceptlastj_cpu=gather(exceptlastj);
exceptfirstj_cpu=gather(exceptfirstj);

%% Expectation vectors E^o_t, t=0..nT-1 (column t+1 of Evec{oo})
% Forward, Lambda maps x at ages 1..N_j-1 to vec(reshape(Gammatranspose*x,[N_a,(N_j-1)*N_z])*pi_z_J_sim) at
% ages 2..N_j. The adjoint is therefore Gammatranspose'*vec(reshape(e at ages 2..N_j)*pi_z_J_sim'), put back
% at ages 1..N_j-1, with zero at age N_j (nobody there is still around next period).
for oo=1:nAgg
    Evec{oo}=zeros(N,nT);
    e=yweighted(:,oo);
    Evec{oo}(:,1)=e;
    for t=2:nT
        etemp=reshape(e(exceptfirstj_cpu),[N_a,(N_j-1)*N_z])*pi_z_J_simtranspose;
        e=zeros(N,1);
        e(exceptlastj_cpu)=Gammatranspose'*etemp(:);
        Evec{oo}(:,t)=e;
    end
end

%% Household Jacobians J^{o,ip}, one price at a time
J_Ap=zeros(nT,nT,nAgg,nP);
for ip=1:nP
    Pperturb0=Psteady;
    Pperturb0.(PricePathNames{ip})=Pperturb0.(PricePathNames{ip})+epsprice; % the price change, at the date it happens

    Vb=V_final; % unperturbed continuation value
    Vp=V_final; % perturbed continuation value
    curlyY=zeros(nT,nAgg);
    curlyD=zeros(N,nT);
    for k=1:nT % k indexes the anticipation horizon s=k-1
        if k==1
            Pp=Pperturb0; % the price has changed at date 0 itself, so it enters the return fn and the outcomes directly
        else
            Pp=Psteady; % the price changes s periods from now, so today only the continuation value differs
        end
        % The baseline is iterated alongside rather than taken as the steady state, so that the two
        % differ only by the perturbation and not by any rounding in how V_final was originally solved
        if transpathoptions.fastOLG==0
            [Vb,Policyb]=ValueFnIter_FHorz_TPath_SingleStep(Vb,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Psteady, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            [Vp,Policyp]=ValueFnIter_FHorz_TPath_SingleStep(Vp,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Pp, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [Vb,Policyb]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG(Vb,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Psteady, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            [Vp,Policyp]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG(Vp,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Pp, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end

        PolicyIndexesPath1(:,:,:,:,1)=Policyb;
        [PolicyForDistb,PolicyProbsb,PolicyValuesb]=TransitionPath_FHorz_substeps_Step2_AdjustPolicy(PolicyIndexesPath1,2,Psteady,n_d,n_a,n_z,0,N_j,l_d,l_aprime,N_a,N_z,0,N_probs,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions);
        PolicyIndexesPath1(:,:,:,:,1)=Policyp;
        [PolicyForDistp,PolicyProbsp,PolicyValuesp]=TransitionPath_FHorz_substeps_Step2_AdjustPolicy(PolicyIndexesPath1,2,Pp,n_d,n_a,n_z,0,N_j,l_d,l_aprime,N_a,N_z,0,N_probs,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions);

        % curlyY: the aggregates at date 0, with the distribution held at the steady state
        AggVarsb=TransitionPath_FHorz_substeps_Step4tt_AggVars(D_ss,AgeWeightsSS,PolicyValuesb(:,:,:,:,1),1,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Psteady,N_j,l_d,l_aprime,l_a,0,l_z,0,N_d,N_a,0,N_z,0,a_gridvals,semizze_gridvals_J_fastOLG,transpathoptions);
        AggVarsp=TransitionPath_FHorz_substeps_Step4tt_AggVars(D_ss,AgeWeightsSS,PolicyValuesp(:,:,:,:,1),1,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Pp,N_j,l_d,l_aprime,l_a,0,l_z,0,N_d,N_a,0,N_z,0,a_gridvals,semizze_gridvals_J_fastOLG,transpathoptions);
        for oo=1:nAgg
            curlyY(k,oo)=gather(AggVarsp.(AggVarNames{oo}).Mean-AggVarsb.(AggVarNames{oo}).Mean)/epsprice;
        end

        % curlyD: the distribution at date 1. The age-1 entries are jequalOneDist in both, so they cancel.
        AgentDistb=TransitionPath_FHorz_substeps_Step3tt_IterAgentDist(D_ss,PolicyForDistb,PolicyProbsb,1,N_a,N_z,0,N_j,N_probs,pi_z_J,pi_z_J_sim,[],[],II1,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDistSS,transpathoptions,simoptions);
        AgentDistp=TransitionPath_FHorz_substeps_Step3tt_IterAgentDist(D_ss,PolicyForDistp,PolicyProbsp,1,N_a,N_z,0,N_j,N_probs,pi_z_J,pi_z_J_sim,[],[],II1,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDistSS,transpathoptions,simoptions);
        curlyD(:,k)=gather(AgentDistp-AgentDistb)/epsprice;
    end

    % The fake-news matrix, and the Jacobian from it
    for oo=1:nAgg
        F=zeros(nT,nT);
        F(1,:)=curlyY(:,oo)'; % F[0,s]=curlyY[s]
        for t=2:nT
            F(t,:)=Evec{oo}(:,t-1)'*curlyD; % F[t,s]=E^o_{t-1}'curlyD[s]
        end
        Joi=zeros(nT,nT);
        Joi(1,:)=F(1,:);
        for t=2:nT
            Joi(t,2:nT)=Joi(t-1,1:nT-1)+F(t,2:nT); % J[t,s]=J[t-1,s-1]+F[t,s]
            Joi(t,1)=F(t,1);
        end
        J_Ap(:,:,oo,ip)=Joi;
    end
    if transpathoptions.verbose==1
        fprintf('TransitionPath_FHorz_FakeNewsJacobian: household Jacobian for price %s done (%i of %i) \n',PricePathNames{ip},ip,nP)
    end
end

%% The general eqm eqns: their direct derivatives in the prices (H_p) and the aggregates (H_A), at the steady state
Pge=Psteady;
for oo=1:nAgg
    Pge.(AggVarNames{oo})=AggVars_ss.(AggVarNames{oo}).Mean;
end
GEbase=zeros(nG,1);
for gg=1:nG
    GEbase(gg)=gather(real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg},GeneralEqmEqnParamNames(gg).Names,Pge)));
end
H_p=zeros(nG,nP);
for ip=1:nP
    Pgeperturb=Pge;
    Pgeperturb.(PricePathNames{ip})=Pgeperturb.(PricePathNames{ip})+epsprice;
    for gg=1:nG
        H_p(gg,ip)=(gather(real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg},GeneralEqmEqnParamNames(gg).Names,Pgeperturb)))-GEbase(gg))/epsprice;
    end
end
H_A=zeros(nG,nAgg);
for oo=1:nAgg
    Pgeperturb=Pge;
    Pgeperturb.(AggVarNames{oo})=Pgeperturb.(AggVarNames{oo})+epsprice;
    for gg=1:nG
        H_A(gg,oo)=(gather(real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg},GeneralEqmEqnParamNames(gg).Names,Pgeperturb)))-GEbase(gg))/epsprice;
    end
end

%% Chain rule: dGE_t/dp_s = H_p*1(t=s) + sum_o H_A(o)*J^{o,p}(t,s)
Jp=zeros(nG*nT,nP*nT);
InT=eye(nT);
for gg=1:nG
    for ip=1:nP
        block=H_p(gg,ip)*InT;
        for oo=1:nAgg
            block=block+H_A(gg,oo)*J_Ap(:,:,oo,ip);
        end
        Jp((gg-1)*nT+(1:nT),(ip-1)*nT+(1:nT))=block;
    end
end

end
