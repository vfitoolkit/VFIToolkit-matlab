function [AggVarsMatrix,AgentDistPathout]=MEP_SSJ_HHblockpath(PricePathMatrix,ParamPathMatrix,B)
% Evaluate the household block along a given price path and parameter (aggregate-shock) path, around the
% stationary-eqm linearization point, WITHOUT any general eqm clearing. Returns the path of aggregate variables.
%   AggVarsMatrix is [T-1, length(AggVarNames)]; AggVarsMatrix(tt,ff) is E[FnsToEvaluate_ff] in period tt.
%   AgentDistPathout (optional) is [N_a*N_z, T-1]; AgentDistPathout(:,tt) is the period-tt agent distribution.
%
% This is the inner map used by the SSJ initial guess (both the brute-force oracle and, later, the fake-news
% Jacobian). It reuses the standard InfHorz transition-path substeps (Step0/Step1/Step2/Step3tt/Step4tt), so the
% grid interpolation layer is handled natively and the deterministic MIT-shock path is computed exactly the same
% way the standard perfect-foresight solver does.
%
% B is a bundle (struct) of everything that is constant across map calls (grids, param-name lists, the
% linearization-point V_final and AgentDist_initial, size vectors, options). It is assembled once in
% MEP_SSJ_InitialGuess.
%
% Scope: N_z>0, N_e==0, no experienceasset (covers Model 1 / Krusell-Smith 1998). The underlying substeps
% error() on the not-yet-implemented noz / e / experienceasset cases.

% Unpack the bundle
T=B.T;
n_d=B.n_d; n_a=B.n_a; n_z=B.n_z;
N_a=B.N_a; N_z=B.N_z;
l_d=B.l_d; l_aprime=B.l_aprime;
d_gridvals=B.d_gridvals; a_grid=B.a_grid; a_gridvals=B.a_gridvals; aprime_gridvals=B.aprime_gridvals;
z_gridvals=B.z_gridvals; pi_z=B.pi_z; pi_z_sparse=B.pi_z_sparse;
ReturnFn=B.ReturnFn; ReturnFnParamNames=B.ReturnFnParamNames; DiscountFactorParamNames=B.DiscountFactorParamNames;
FnsToEvaluateCell=B.FnsToEvaluateCell; FnsToEvaluateParamNames=B.FnsToEvaluateParamNames; AggVarNames=B.AggVarNames;
V_final=B.V_final; AgentDist_initial=B.AgentDist_initial;
PricePathNames=B.PricePathNames; ParamPathNames=B.ParamPathNames;
PricePathSizeVec=B.PricePathSizeVec; ParamPathSizeVec=B.ParamPathSizeVec;
Parameters=B.Parameters;
transpathoptions=B.transpathoptions; vfoptions=B.vfoptions; simoptions=B.simoptions;

% N_e==0 throughout this scope
n_e=0; N_e=0; e_gridvals=[]; pi_e=[];

%% Step 0: preallocate PolicyIndexesPath and the forward-iteration index helpers
[PolicyIndexesPath,N_probs,II1,II2]=TransitionPath_InfHorz_substeps_Step0_setup(l_d,l_aprime,N_a,N_z,N_e,T,transpathoptions,vfoptions,simoptions);

%% Step 1: backward pass — one Bellman step per period from the terminal V_final, given the price/param path
[~,PolicyIndexesPath]=TransitionPath_InfHorz_substeps_Step1_ValueFnIter(T,PolicyIndexesPath,V_final,Parameters,PricePathMatrix,ParamPathMatrix,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_z,N_e,d_gridvals,a_grid,z_gridvals,e_gridvals,pi_z,pi_e,ReturnFn,DiscountFactorParamNames,ReturnFnParamNames,transpathoptions,vfoptions);

%% Step 2: reshape the policy into the forms needed for forward iteration and for evaluating FnsToEvaluate
[PolicyPath_ForAgentDistIter,PolicyProbsPath,PolicyValuesPath]=TransitionPath_InfHorz_substeps_Step2_AdjustPolicy(PolicyIndexesPath,T,Parameters,n_d,n_a,n_z,n_e,l_d,l_aprime,N_a,N_z,N_e,N_probs,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions);

%% Steps 3 & 4: iterate the agent distribution forward and compute aggregate variables period-by-period
AgentDist=AgentDist_initial;
AggVarsMatrix=zeros(T-1,length(AggVarNames));
if nargout>=2
    AgentDistPathout=zeros(N_a*N_z,T-1);
end
for tt=1:T-1
    % Put the period-tt prices and aggregate shocks into Parameters (needed by any FnsToEvaluate that use them directly)
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePathMatrix(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPathMatrix(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end

    if nargout>=2
        AgentDistPathout(:,tt)=gather(AgentDist);
    end

    % Iterate the agent distribution one period forward (uses period-tt policy)
    AgentDistnext=TransitionPath_InfHorz_substeps_Step3tt_IterAgentDist(AgentDist,PolicyPath_ForAgentDistIter,PolicyProbsPath,tt,N_a,N_z,N_e,N_probs,pi_z_sparse,pi_e,II1,II2,transpathoptions,simoptions);

    % Aggregate variables in period tt (uses the period-tt distribution, before it is iterated)
    AggVars=TransitionPath_InfHorz_substeps_Step4tt_AggVars(AgentDist,PolicyValuesPath(:,:,:,tt),tt,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,n_a,n_z,n_e,N_z,N_e,a_gridvals,z_gridvals,transpathoptions);
    for ff=1:length(AggVarNames)
        AggVarsMatrix(tt,ff)=AggVars.(AggVarNames{ff}).Mean;
    end

    AgentDist=AgentDistnext;
end

end
