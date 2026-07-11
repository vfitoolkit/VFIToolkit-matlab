function [PricePath,VPath,AgentDistPath,AggVarsPath]=MEP_SSJ_InitialGuess(p_eqm,V,Policy,StationaryDist,AggVars,...
    T,ss_ind_T,n_d,n_a,n_z,n_S,N_a,N_z,N_S,d_grid,a_grid,initialguessobjects,AggShockNames,AggShocksPath,...
    AggVarNames,ReturnFn,ReturnFnParamNames,FnsToEvaluate,FnsToEvaluateCell,FnsToEvaluateParamNames,...
    GeneralEqmEqnsStruct,Parameters,DiscountFactorParamNames,...
    GEPriceParamNames,GEeqnNames,recursiveeqmoptions,vfoptions,simoptions)
% Build a linearized initial guess for the aggregate-shocks path using the sequence-space Jacobian (SSJ).
%
% Called by MEP_CreateInitialGuess when initialguessobjects.methodforguess==3. On entry the stationary general
% eqm at S=E[S] has already been solved, and is passed in as p_eqm, V, Policy, StationaryDist, AggVars (all at
% S=E[S]). initialguessobjects.Svalue holds E[S], and AggShocksPath is the realized aggregate-shock path, so
% dS = AggShocksPath - E[S] is the shock deviation that drives the linearized guess.
%
% recursiveeqmoptions.SSJmethod: =1 fake-news algorithm (efficient; not yet implemented),
%                                =2 brute-force oracle (slow, O(T^2); permanent validation reference).
%
% STATUS (build in progress): the shared inner map MEP_SSJ_HHblockpath (price/param path -> aggregate path) is
% built, and this routine currently runs a BASELINE SELF-CHECK of it (a path held at the steady state must
% reproduce the stationary AggVars) and then returns the flat (methodforguess==1) guess. Still to add: the
% perturbation loop that builds the household-block Jacobians, the general-eqm-equation Jacobians, and the
% linear solve dp = -Jp \ (JS*dS) that overwrites PricePath.

%% Scope guard: only the N_z>0, N_e==0, no-experienceasset case is implemented so far (covers Model 1 / KS1998)
if N_z==0
    error('MEP_SSJ_InitialGuess: initialguessmethod==3 (SSJ) not yet implemented for models with no z (N_z==0)')
end
if isfield(vfoptions,'n_e') && ~isempty(vfoptions.n_e) && prod(vfoptions.n_e)>1
    error('MEP_SSJ_InitialGuess: initialguessmethod==3 (SSJ) not yet implemented for models with an e variable')
end
if isfield(simoptions,'experienceasset') && simoptions.experienceasset>=1
    error('MEP_SSJ_InitialGuess: initialguessmethod==3 (SSJ) not yet implemented for experienceasset models')
end

%% Sizes and grid-value forms
l_a=length(n_a); l_z=length(n_z); l_aprime=l_a;
if isscalar(n_d) && n_d(1)==0
    l_d=0; N_d=0; d_gridvals=[];
else
    l_d=length(n_d); N_d=prod(n_d); d_gridvals=CreateGridvals(n_d,gpuArray(d_grid),1);
end

a_gridvals=CreateGridvals(n_a,a_grid,1); % [N_a,l_a]

% aprime_gridvals (fine grid when using the grid interpolation layer), following TransitionPath_InfHorz setup
if vfoptions.gridinterplayer==0
    aprime_gridvals=a_gridvals;
elseif vfoptions.gridinterplayer==1
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
end

% Idiosyncratic shock at the linearization point (S averaged out; held constant along the path)
z_gridvals=initialguessobjects.z_gridvals;
pi_z=initialguessobjects.pi_z;
pi_z_sparse=sparse(gpuArray(pi_z)); % [N_z,N_zprime], natural orientation used as AgentDist*pi_z_sparse

% Linearization point
V_final=reshape(gpuArray(V),[N_a,N_z]);
AgentDist_initial=gather(reshape(StationaryDist,[N_a*N_z,1]));

%% Options for the transition-path substeps (trivial paths, no experienceasset/e)
transpathoptions=struct();
transpathoptions.zpathtrivial=1;
transpathoptions.epathtrivial=1;
transpathoptions.zepathtrivial=1;
ssjvfoptions=vfoptions; ssjvfoptions.experienceasset=0;
ssjsimoptions=simoptions; ssjsimoptions.experienceasset=0; ssjsimoptions.fastOLG=0;
if ~isfield(ssjsimoptions,'gridinterplayer'); ssjsimoptions.gridinterplayer=vfoptions.gridinterplayer; end
if ~isfield(ssjsimoptions,'ngridinterp') && isfield(vfoptions,'ngridinterp'); ssjsimoptions.ngridinterp=vfoptions.ngridinterp; end

%% Baseline price/param paths (held at the steady state) and their matrix + size-vector forms
PricePathStruct=struct(); ParamPathStruct=struct();
for pp=1:length(GEPriceParamNames)
    PricePathStruct.(GEPriceParamNames{pp})=p_eqm.(GEPriceParamNames{pp})*ones(1,T);
end
for ss=1:length(AggShockNames)
    ParamPathStruct.(AggShockNames{ss})=initialguessobjects.Svalue(ss)*ones(1,T);
end
[PricePathBaseMatrix,ParamPathBaseMatrix,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePathStruct,ParamPathStruct,T);

%% Assemble the bundle of constants for the inner map
B=struct();
B.T=T; B.n_d=n_d; B.n_a=n_a; B.n_z=n_z; B.N_a=N_a; B.N_z=N_z;
B.l_d=l_d; B.l_aprime=l_aprime; B.l_a=l_a; B.l_z=l_z;
B.d_gridvals=d_gridvals; B.a_grid=a_grid; B.a_gridvals=a_gridvals; B.aprime_gridvals=aprime_gridvals;
B.z_gridvals=z_gridvals; B.pi_z=pi_z; B.pi_z_sparse=pi_z_sparse;
B.ReturnFn=ReturnFn; B.ReturnFnParamNames=ReturnFnParamNames; B.DiscountFactorParamNames=DiscountFactorParamNames;
B.FnsToEvaluateCell=FnsToEvaluateCell; B.FnsToEvaluateParamNames=FnsToEvaluateParamNames; B.AggVarNames=AggVarNames;
B.V_final=V_final; B.AgentDist_initial=AgentDist_initial;
B.PricePathNames=PricePathNames; B.ParamPathNames=ParamPathNames;
B.PricePathSizeVec=PricePathSizeVec; B.ParamPathSizeVec=ParamPathSizeVec;
B.Parameters=Parameters;
B.transpathoptions=transpathoptions; B.vfoptions=ssjvfoptions; B.simoptions=ssjsimoptions;

%% BASELINE SELF-CHECK: a path held at the steady state must reproduce the stationary AggVars
% Since the inputs are constant at the steady state and AgentDist starts at the stationary distribution, the
% household block should stay at the steady state, so every period's aggregate should equal the stationary AggVars.
if recursiveeqmoptions.verbose>=1
    fprintf('MEP_SSJ_InitialGuess: running baseline self-check of the inner household-block map \n')
end
baseAgg=MEP_SSJ_HHblockpath(PricePathBaseMatrix,ParamPathBaseMatrix,B); % [T-1,nAgg]
for ff=1:length(AggVarNames)
    ssval=AggVars.(AggVarNames{ff}).Mean;
    maxdev=max(abs(baseAgg(:,ff)-ssval));
    fprintf('  %s: stationary=%12.8f, max|path-stationary| over t=1..T-1 = %.3e \n',AggVarNames{ff},ssval,maxdev);
end
fprintf('MEP_SSJ_InitialGuess: baseline self-check done. If the deviations above are ~machine-precision, the \n');
fprintf('  inner map is validated and the next step is the perturbation loop + GE-eqn Jacobians + linear solve. \n');

%% TODO: perturbation loop -> household Jacobians J^{A,p}, J^{A,S}*dS; GE-eqn Jacobians H_p,H_A,H_S;
%        solve dp = -Jp \ (H_S*dS + H_A*(J^{A,S}*dS)); PricePath = p_eqm + dp; forward pass for consistent paths.
%        recursiveeqmoptions.SSJmethod selects: =2 brute-force oracle (built on baseAgg + per-horizon bumps),
%        =1 fake-news algorithm.

%% For now, return the flat (methodforguess==1) guess so methodforguess==3 remains runnable
warning('MEP_SSJ_InitialGuess: SSJ price-path solve not yet implemented; returning the flat (methodforguess==1) initial guess for now')
VPath=repelem(reshape(V,[N_a,1,N_z]),1,T,1); % fastOLG means (a,t)-by-z
AgentDistPath=repelem(reshape(StationaryDist,[N_a,1,N_z]),1,T,1); % fastOLG means (a,t,z)-by-1
VPath=reshape(VPath,[N_a*T,N_z]);
AgentDistPath=reshape(AgentDistPath,[N_a*T*N_z,1]);
for ff=1:length(AggVarNames)
    AggVarsPath.(AggVarNames{ff}).Mean=repmat(AggVars.(AggVarNames{ff}).Mean,1,T);
end
for pp=1:length(GEPriceParamNames)
    PricePath.(GEPriceParamNames{pp})=p_eqm.(GEPriceParamNames{pp})*ones(1,T);
end

end
