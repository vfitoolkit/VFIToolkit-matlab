function Omega=TransitionPath_InfHorz_LudwigWstationary(n_d,n_a,n_z,pi_z,d_gridvals,a_grid,z_gridvals,ReturnFn,FnsToEvaluateCell,AggVarNames,GeneralEqmEqnsStruct,GEeqnNames,Parameters,DiscountFactorParamNames,ReturnFnParamNames,PricePathNames,PricePathSizeVec,PricePathOld,ParamPathNames,ParamPathSizeVec,ParamPath,T,V_final,use_tminus1price,use_tminus1params,use_tplus1price,use_tminus1AggVars,use_stockvars,vfoptions,simoptions,transpathoptions)
% W, the low-dimensional matrix behind Ludwig (2007)'s GSQN, evaluated at the FINAL stationary eqm.
%
% Ludwig sec 3.1: in a steady state the partial derivatives of the general eqm conditions depend only
% on the lead or the lag, not on the period itself, so the Jacobian of the whole price path collapses
% to Omega kron I (his eq 13). Omega(i,j) is the sum over all leads and lags of d(GE condn i)/d(price
% j), which is the response to a PERMANENT change in price j. A stationary equilibrium at a different
% price vector is exactly that: the price changed in every period, forever. So Omega is nothing more
% than the Jacobian of the STATIONARY general eqm conditions with respect to the prices, taken at the
% final stationary eqm.
%
% That is the whole point of GSQN, and it is what sec 3.2 means by using "the Jacobi matrix derived
% during (fast) steady state calculations": this costs nPrices+1 STATIONARY solves and no path solves
% at all, where a path solve is T value fn steps and T agent dist steps.
%
% The stationary conditions are evaluated by calling HeteroAgentStationaryEqm_InfHorz with
% heteroagentoptions.maxiter=0, which is its documented 'just evaluate the general eqm eqns at the
% current prices' mode. It reads the prices out of Parameters, so perturbing price j means setting
% Parameters.(PricePathNames{j}); and it zeroes constrainpositive/constrain0to1/constrainAtoB first,
% so this is all in raw price space with no transformations to undo.

nPrices=size(PricePathOld,2);

%% A warning, because this method has not yet been made to work
% On the test bank models it descends well and then stalls short of the tolerance. A stationary
% Jacobian measures the LONG-RUN response, while the path being solved is truncated at T, and the two
% are not the same object: on the model without d, d(LabourMarket)/dw is +0.88 measured on the path
% and -3.0 measured at the steady state, because in the long run the capital stock moves enough for
% the indirect effect to overwhelm the direct one. A sign-flipped diagonal sends the step the wrong
% way. Perturbing the stationary conditions also degenerates as epsprice shrinks: below about 1e-4 the
% bump stops moving any discretised policy and Omega collapses to the identity.
warning('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigStationary'' usually fails to reach toleranceGEcondns, and is kept for comparison rather than for use. It builds the Jacobian at the final stationary eqm, which measures a long-run response, whereas the transition path being solved is truncated at T; the two differ, and have been seen to differ in sign. Prefer ''LudwigPath'' (the same structure, measured on the current price path), or ''LudwigSSJ'', or ''FullJacobian''.')

%% What this method cannot do
if use_stockvars==1
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigStationary'' cannot be used with stock variables: a stock variable has no general steady-state law of motion that could be imposed here, so there is nothing safe to assume. Use Jacobianmethod=''LudwigSSJ'' or ''FullJacobian'' instead')
end
if length(PricePathNames)~=nPrices
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigStationary'' needs one scalar price per column of the price path (found %i names for %i columns), because the stationary general eqm conditions are differenced one named price at a time. Use Jacobianmethod=''LudwigSSJ'' or ''FullJacobian'' instead',length(PricePathNames),nPrices)
end

%% The general eqm eqns to use at the steady state
% The path's own general eqm eqns may refer to the previous or the next period: t-1 prices, t+1
% prices, t-1 parameters, t-1 aggregate variables. Every one of those equals its same-period value in
% a steady state, but HeteroAgentStationaryEqm_InfHorz has no notion of them and would go looking for
% parameters that do not exist. So when the path uses any of them, the user gives a copy of the
% general eqm eqns written with same-period names only.
if use_tminus1price==1 || use_tplus1price==1 || use_tminus1params==1 || use_tminus1AggVars==1
    if ~isfield(transpathoptions,'GEnewprice1') || ~isfield(transpathoptions.GEnewprice1,'LudwigGeneralEqmEqns')
        error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigStationary'': your GeneralEqmEqns use the previous or next period (t-1 or t+1 prices, parameters or aggregate variables), which have no meaning in the stationary general eqm that W is built from. Set transpathoptions.GEnewprice1.LudwigGeneralEqmEqns to a copy of your GeneralEqmEqns written with same-period names only (in a steady state they are the same thing), with the same eqn names in the same order')
    end
    GeneralEqmEqnsSS=transpathoptions.GEnewprice1.LudwigGeneralEqmEqns;
    SSeqnNames=fieldnames(GeneralEqmEqnsSS);
    % Same names in the same order, because the rows of Omega have to line up with the columns of
    % GEcondnPath, and both orderings are just fieldnames() of their respective structures
    if length(SSeqnNames)~=length(GEeqnNames)
        error('transpathoptions.GEnewprice1.LudwigGeneralEqmEqns must have one eqn for each of your GeneralEqmEqns (found %i, expected %i)',length(SSeqnNames),length(GEeqnNames))
    end
    for gg=1:length(GEeqnNames)
        if ~strcmp(SSeqnNames{gg},GEeqnNames{gg})
            error('transpathoptions.GEnewprice1.LudwigGeneralEqmEqns must use the same eqn names in the same order as GeneralEqmEqns (eqn %i is ''%s'' but should be ''%s'')',gg,SSeqnNames{gg},GEeqnNames{gg})
        end
    end
else
    GeneralEqmEqnsSS=GeneralEqmEqnsStruct;
end

%% Set up the call
% FnsToEvaluate as a structure: the cell entries are the anonymous functions themselves, so this is
% exactly the structure the transition path was given
FnsToEvaluate=struct();
for ii=1:length(AggVarNames)
    FnsToEvaluate.(AggVarNames{ii})=FnsToEvaluateCell{ii};
end

% Parameters at the final stationary eqm: period T of the price path is the terminal condition, and
% period T of the parameter path is where the reform has settled down
ParametersSS=Parameters;
for pp=1:length(PricePathNames)
    ParametersSS.(PricePathNames{pp})=PricePathOld(T,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp));
end
for pp=1:length(ParamPathNames)
    ParametersSS.(ParamPathNames{pp})=ParamPath(T,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp));
end

heteroagentoptionsW=struct();
heteroagentoptionsW.maxiter=0; % just evaluate the general eqm eqns, do not solve for anything
heteroagentoptionsW.outputGEstruct=2; % without this the maxiter=0 branch never evaluates them at all
heteroagentoptionsW.verbose=0; % this gets called nPrices+1 times, so stay quiet
if transpathoptions.useintermediateEqns==1
    % HeteroAgentStationaryEqm_InfHorz sets up the cell form and the parameter names itself, and
    % evaluates them before the general eqm eqns, exactly as the transition path does
    heteroagentoptionsW.intermediateEqns=transpathoptions.intermediateEqns;
end

% Warm start every value fn iteration from the terminal value fn, which is the steady state these are
% all perturbations of, so each solve is a few sweeps rather than a cold start. Infinities are
% replaced by large finite values: a -Inf here can meet a zero weight and give 0*(-Inf)=NaN, while
% -10^9 gives 0 and still orders the states the same way, and the value stays recoverable.
vfoptionsW=vfoptions;
vfoptionsW.V0=V_final;
vfoptionsW.V0(V_final==Inf)=10^9;
vfoptionsW.V0(V_final==-Inf)=-10^9;

%% One stationary solve for the base point, then one per price
% FnsToEvaluateParamNames and GeneralEqmEqnParamNames are passed empty because
% HeteroAgentStationaryEqm_InfHorz rebuilds both from the structures with getAnonymousFnInputNames,
% and the ones the transition path holds would be for the path's general eqm eqns rather than these
[~,GEcondnsbase]=HeteroAgentStationaryEqm_InfHorz(n_d,n_a,n_z,0,pi_z,d_gridvals,a_grid,z_gridvals, ReturnFn, FnsToEvaluate, GeneralEqmEqnsSS, ParametersSS, DiscountFactorParamNames, ReturnFnParamNames, [], [], PricePathNames, heteroagentoptionsW, simoptions, vfoptionsW);
GEcondnsbase=GEcondnsbase(:);

Omega=zeros(length(GEeqnNames),nPrices);
for jj=1:nPrices
    ParametersSSjj=ParametersSS;
    ParametersSSjj.(PricePathNames{jj})=ParametersSS.(PricePathNames{jj})+transpathoptions.epsprice;
    [~,GEcondnsjj]=HeteroAgentStationaryEqm_InfHorz(n_d,n_a,n_z,0,pi_z,d_gridvals,a_grid,z_gridvals, ReturnFn, FnsToEvaluate, GeneralEqmEqnsSS, ParametersSSjj, DiscountFactorParamNames, ReturnFnParamNames, [], [], PricePathNames, heteroagentoptionsW, simoptions, vfoptionsW);
    Omega(:,jj)=(GEcondnsjj(:)-GEcondnsbase)/transpathoptions.epsprice;
end

end
