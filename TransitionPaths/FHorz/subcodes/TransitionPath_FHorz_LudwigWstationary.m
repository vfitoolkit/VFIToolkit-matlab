function Omega=TransitionPath_FHorz_LudwigWstationary(jequalOneDist,AgeWeightsParamNames,n_d,n_a,n_z,N_j,pi_z,d_grid,a_grid,z_grid,ReturnFn,FnsToEvaluate,GeneralEqmEqnsStruct,GEeqnNames,Parameters,DiscountFactorParamNames,PricePathNames,PricePathSizeVec,PricePathOld,ParamPathNames,ParamPathSizeVec,ParamPath,T,use_tminus1price,use_tminus1params,use_tplus1price,use_tminus1AggVars,use_stockvars,vfoptions,simoptions,transpathoptions)
% W, the low-dimensional matrix behind Ludwig (2007)'s GSQN, evaluated at the FINAL stationary eqm.
% The FHorz version of TransitionPath_InfHorz_LudwigWstationary; see there for the reasoning.
%
% Ludwig sec 3.1: in a steady state the partial derivatives of the general eqm conditions depend only
% on the lead or the lag, not on the period itself, so the Jacobian of the whole price path collapses
% to Omega kron I (his eq 13). Omega(i,j) is the sum over all leads and lags of d(GE condn i)/d(price
% j), which is the response to a PERMANENT change in price j. A stationary equilibrium at a different
% price vector is exactly that, so Omega is the Jacobian of the STATIONARY general eqm conditions with
% respect to the prices, taken at the final stationary eqm. It costs nPrices+1 stationary solves and
% no path solves at all.
%
% The stationary conditions are evaluated by calling HeteroAgentStationaryEqm_Case1_FHorz with
% heteroagentoptions.maxiter=0, which is its 'just evaluate the general eqm eqns at the current
% prices' mode. It reads the prices out of Parameters, so perturbing price j means setting
% Parameters.(PricePathNames{j}).

nPrices=size(PricePathOld,2);

%% A warning, because this method has not yet been made to work
% On the InfHorz test bank models it descends well and then stalls short of the tolerance: a
% stationary Jacobian measures the LONG-RUN response, while the path being solved is truncated at T,
% and the two have been seen to differ in sign.
warning('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigStationary'' usually fails to reach toleranceGEcondns, and is kept for comparison rather than for use. It builds the Jacobian at the final stationary eqm, which measures a long-run response, whereas the transition path being solved is truncated at T; the two differ, and have been seen to differ in sign. Prefer ''LudwigPath'' (the same structure, measured on the current price path), or ''LudwigSSJ'', or ''FullJacobian''.')

%% What this method cannot do
if use_stockvars==1
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigStationary'' cannot be used with stock variables: a stock variable has no general steady-state law of motion that could be imposed here, so there is nothing safe to assume. Use Jacobianmethod=''LudwigSSJ'' or ''FullJacobian'' instead')
end
if length(PricePathNames)~=nPrices
    error('transpathoptions.GEnewprice1.Jacobianmethod=''LudwigStationary'' needs one scalar price per column of the price path (found %i names for %i columns), because the stationary general eqm conditions are differenced one named price at a time. Use Jacobianmethod=''LudwigPath'' or ''FullJacobian'' instead',length(PricePathNames),nPrices)
end

%% The general eqm eqns to use at the steady state
% The path's own general eqm eqns may refer to the previous or the next period, which have no
% meaning in the stationary general eqm, so the user then gives a copy written with same-period names.
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
% Parameters at the final stationary eqm: period T of the price path is the terminal condition, and
% period T of the parameter path is where the reform has settled down (this includes the age weights,
% when they are on the parameter path)
ParametersSS=Parameters;
for pp=1:length(PricePathNames)
    ParametersSS.(PricePathNames{pp})=PricePathOld(T,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp));
end
for pp=1:length(ParamPathNames)
    ParametersSS.(ParamPathNames{pp})=ParamPath(T,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp));
end
% And the age j=1 distribution at period T, when it is given as a path
if transpathoptions.trivialjequalonedist==0
    jequalOneDistSS=jequalOneDist(:,end);
else
    jequalOneDistSS=jequalOneDist;
end

heteroagentoptionsW=struct();
heteroagentoptionsW.maxiter=0; % just evaluate the general eqm eqns, do not solve for anything
heteroagentoptionsW.outputGEstruct=2; % without this the maxiter=0 branch never evaluates them at all
heteroagentoptionsW.verbose=0; % this gets called nPrices+1 times, so stay quiet
if transpathoptions.useintermediateEqns==1
    % HeteroAgentStationaryEqm_Case1_FHorz sets up the cell form and the parameter names itself, and
    % evaluates them before the general eqm eqns, exactly as the transition path does
    heteroagentoptionsW.intermediateEqns=transpathoptions.intermediateEqns;
end

%% One stationary solve for the base point, then one per price
% ReturnFnParamNames, FnsToEvaluateParamNames and GeneralEqmEqnParamNames are passed empty because
% HeteroAgentStationaryEqm_Case1_FHorz rebuilds them from the functions themselves, and the ones the
% transition path holds would be for the path's general eqm eqns rather than these
[~,GEcondnsbase]=HeteroAgentStationaryEqm_Case1_FHorz(jequalOneDistSS,AgeWeightsParamNames,n_d,n_a,n_z,N_j,0,pi_z,d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqnsSS, ParametersSS, DiscountFactorParamNames, [], [], [], PricePathNames, heteroagentoptionsW, simoptions, vfoptions);
GEcondnsbase=GEcondnsbase(:);

Omega=zeros(length(GEeqnNames),nPrices);
for jj=1:nPrices
    ParametersSSjj=ParametersSS;
    ParametersSSjj.(PricePathNames{jj})=ParametersSS.(PricePathNames{jj})+transpathoptions.epsprice;
    [~,GEcondnsjj]=HeteroAgentStationaryEqm_Case1_FHorz(jequalOneDistSS,AgeWeightsParamNames,n_d,n_a,n_z,N_j,0,pi_z,d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqnsSS, ParametersSSjj, DiscountFactorParamNames, [], [], [], PricePathNames, heteroagentoptionsW, simoptions, vfoptions);
    Omega(:,jj)=(GEcondnsjj(:)-GEcondnsbase)/transpathoptions.epsprice;
end

end
