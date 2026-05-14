function [PricePath,VPath,AgentDistPath,AggVarsPath,GEcheck]=MatchedExpectationsPath_CreateInitialGuess(T,ss_ind_T,n_d,n_a,n_z,n_S,N_a,N_z,N_S,d_grid,a_grid,initialguessobjects,AggShockNames,AggVarNames,ReturnFn,FnsToEvaluate,GeneralEqmEqnsStruct,Parameters,DiscountFactorParamNames, GEPriceParamNames,GEeqnNames,recursiveeqmoptions,vfoptions,simoptions)
% initialguessobjects.methodforguess
% =1: replace S with E[S]
% =2: treat S as idiosyncratic shock

if initialguessobjects.methodforguess==1
    % Replace all instances of S with E[S], both in agent problem and everywhere else

    % General eqm should be able use S as an input, so I need to put in some kind of value of S into Parameters that can be used for the general eqm eqns
    for SS_c=1:length(n_S)
        Parameters.(AggShockNames{SS_c})=initialguessobjects.Svalue(SS_c);
    end
    % Because it is InfHorz, turn off divide-and-conquer if that is used
    vfoptions.divideandconquer=0;
    % initialguessobjects.heteroagentoptions.verbose=1
    [p_eqm,GEcheck]=HeteroAgentStationaryEqm_Case1(n_d, n_a, n_z, 0, initialguessobjects.pi_z, d_grid, a_grid, initialguessobjects.z_gridvals, ReturnFn, FnsToEvaluate, GeneralEqmEqnsStruct, Parameters, DiscountFactorParamNames, [], [], [], GEPriceParamNames, initialguessobjects.heteroagentoptions, simoptions, vfoptions);
    for pp=1:length(GEPriceParamNames)
        Parameters.(GEPriceParamNames{pp})=p_eqm.(GEPriceParamNames{pp});
    end
    [V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,initialguessobjects.z_gridvals, initialguessobjects.pi_z, ReturnFn, Parameters, DiscountFactorParamNames, [], vfoptions);
    StationaryDist=StationaryDist_Case1(Policy,n_d,n_a,n_z,initialguessobjects.pi_z,simoptions,Parameters);
    AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist, Policy, FnsToEvaluate, Parameters, [], n_d, n_a, n_z, d_grid, a_grid, initialguessobjects.z_gridvals, simoptions);
    % Create the objects needed as the initial guess setup for the matched expectations
    VPath=repelem(reshape(V,[N_a,1,N_z]),1,T,1); % fastOLG means (a,t)-by-z
    AgentDistPath=repelem(reshape(StationaryDist,[N_a,1,N_z]),1,T,1); % fastOLG means (a,t,z)-by-1
    VPath=reshape(VPath,[N_a*T,N_z]);
    AgentDistPath=reshape(AgentDistPath,[N_a*T*N_z,1]);
    for ff=1:length(AggVarNames)
        AggVarsPath.(AggVarNames{ff}).Mean=repmat(AggVars.(AggVarNames{ff}).Mean,1,T);
    end
    % Setup PricePath as the general eqm prices
    for pp=1:length(GEPriceParamNames)
        PricePath.(GEPriceParamNames{pp})=p_eqm.(GEPriceParamNames{pp})*ones(1,T);
    end

elseif initialguessobjects.methodforguess==2
    % Just pretend that the aggregate shock is an idiosyncratic shock, solve the stationary general eqm of that model.
    n_zS=[n_z,n_S];

    % General eqm should be able use S as an input, so I need to put in some kind of value of S into Parameters that can be used for the general eqm eqns
    for SS_c=1:length(n_S)
        Parameters.(AggShockNames{SS_c})=initialguessobjects.Svalue(SS_c);
    end
    % Because it is InfHorz, turn off divide-and-conquer if that is used
    vfoptions.divideandconquer=0;
    % initialguessobjects.heteroagentoptions.verbose=1
    [p_eqm,GEcheck]=HeteroAgentStationaryEqm_Case1(n_d, n_a, n_zS, 0, initialguessobjects.pi_zS, d_grid, a_grid, initialguessobjects.zS_gridvals, ReturnFn, FnsToEvaluate, GeneralEqmEqnsStruct, Parameters, DiscountFactorParamNames, [], [], [], GEPriceParamNames, initialguessobjects.heteroagentoptions, simoptions, vfoptions);
    for pp=1:length(GEPriceParamNames)
        Parameters.(GEPriceParamNames{pp})=p_eqm.(GEPriceParamNames{pp});
    end
    [V,Policy]=ValueFnIter_Case1(n_d,n_a,n_zS,d_grid,a_grid,initialguessobjects.zS_gridvals, initialguessobjects.pi_zS, ReturnFn, Parameters, DiscountFactorParamNames, [], vfoptions);
    StationaryDist=StationaryDist_Case1(Policy,n_d,n_a,n_zS,initialguessobjects.pi_zS,simoptions,Parameters);
    AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist, Policy, FnsToEvaluate, Parameters, [], n_d, n_a, n_zS, d_grid, a_grid, initialguessobjects.zS_gridvals, simoptions);
    % Create the objects needed as the initial guess setup for the matched expectations
    V=reshape(V,[N_a,1,N_z,N_S]);
    StationaryDist=reshape(StationaryDist,[N_a,1,N_z,N_S]);
    VPath=zeros(N_a,T,N_z,'gpuArray'); % fastOLG means (a,t)-by-z
    AgentDistPath=zeros(N_a,T,N_z,'gpuArray'); % fastOLG means (a,t,z)-by-1
    for tt=1:T
        VPath(:,tt,:)=V(:,1,:,ss_ind_T(tt));
        AgentDistPath(:,tt,:)=StationaryDist(:,1,:,ss_ind_T(tt)); % this will not be the whole mass, so renormalize immediately after we finish looping
    end
    AgentDistPath=AgentDistPath./sum(sum(AgentDistPath,3),1);
    VPath=reshape(VPath,[N_a*T,N_z]);
    AgentDistPath=reshape(AgentDistPath,[N_a*T*N_z,1]);
    for ff=1:length(AggVarNames)
        AggVarsPath.(AggVarNames{ff}).Mean=repmat(AggVars.(AggVarNames{ff}).Mean,1,T);
    end
    % Setup PricePath as the general eqm prices
    for pp=1:length(GEPriceParamNames)
        PricePath.(GEPriceParamNames{pp})=p_eqm.(GEPriceParamNames{pp})*ones(1,T);
    end

end


if recursiveeqmoptions.verbose==2
    fprintf(' \n')
    fprintf(' \n')
    fprintf(' \n')
    fprintf('Finished solving the initial guess stationary eqm that will be used to setup initial guess for path \n')
    fprintf('Following is info about it, \n')
    fprintf('GE prices: \n')
    for pp=1:length(GEPriceParamNames)
        fprintf('	%s: %8.6f \n',GEPriceParamNames{pp},p_eqm.(GEPriceParamNames{pp}))
    end
    fprintf('Current aggregate variables: \n')
    for aa=1:length(AggVarNames)
        fprintf('	%s: %8.6f \n',AggVarNames{aa},AggVars.(AggVarNames{aa}).Mean)
    end
    fprintf('Current GeneralEqmEqns: \n')
    for gg=1:length(GEeqnNames)
        fprintf('	%s: %8.8f \n',GEeqnNames{gg},GEcheck.(GEeqnNames{gg}))
    end
    fprintf('End of info about the initial guess stationary eqm. \n')
end



end