function AggVarsPath=EvalFnOnTransPath_AggVars_InfHorz(FnsToEvaluate,AgentDistPath,PolicyPath,PricePath,ParamPath, Parameters, T, n_d, n_a, n_z, d_grid, a_grid,z_grid,transpathoptions,simoptions)
% AggVarsPath is T periods long (periods 0 (before the reforms are announced) & T are the initial and final values).
% No z is n_z=0 (z_grid=[]). iid e shocks are given in simoptions (n_e, e_grid, pi_e), as for EvalFnOnAgentDist_AggVars_InfHorz;
% e_grid may vary over the path. AgentDistPath is [n_a,(n_z),(n_e),T] and PolicyPath [rows,n_a,(n_z),(n_e),T].
% Semi-exogenous states are given in simoptions (n_semiz, semiz_grid, SemiExoStateFn or pi_semiz, l_dsemiz), as for
% EvalFnOnAgentDist_AggVars_InfHorz; then AgentDistPath is [n_a,n_semiz,(n_z),(n_e),T], PolicyPath [rows,n_a,n_semiz,(n_z),(n_e),T]
% and FnsToEvaluate inputs are (d,aprime,a,semiz,z,e,...).

if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.experienceasset=0;
    simoptions.experienceassetz=0;
    simoptions.experienceassete=0;
    simoptions.experienceassetze=0;
    simoptions.gridinterplayer=0;
    simoptions.n_e=0;
    simoptions.n_semiz=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    if ~isfield(simoptions,'experienceassetz')
        simoptions.experienceassetz=0;
    end
    if ~isfield(simoptions,'experienceassete')
        simoptions.experienceassete=0;
    end
    if ~isfield(simoptions,'experienceassetze')
        simoptions.experienceassetze=0;
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
end

l_d=length(n_d);
if n_d(1)==0
    l_d=0;
end
l_a=length(n_a);
l_aprime=l_a-simoptions.experienceasset;

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end
% Read here, because ExogShockSetup_InfHorz_TPath below removes n_e from simoptions
n_e=simoptions.n_e;
N_e=prod(n_e);
if N_e==0
    l_e=0;
else
    l_e=length(n_e);
end
if N_e>0 && simoptions.experienceasset>=1
    error('EvalFnOnTransPath_AggVars_InfHorz: experienceasset together with an e variable is not yet implemented')
end
% (z,e) are kept as one exogenous dimension, z first, which is a singleton when there is neither
N_ze=max(N_z,1)*max(N_e,1);
if N_z==0 && N_e==0
    n_ze=0;
elseif N_z>0 && N_e==0
    n_ze=n_z;
elseif N_z==0 && N_e>0
    n_ze=n_e;
elseif N_z>0 && N_e>0
    n_ze=[n_z,n_e];
end
% Semi-exogenous states (simoptions.n_semiz, with semiz_grid, and SemiExoStateFn or pi_semiz, l_dsemiz) go in front of
% (z,e), semiz fastest: from here on the exogenous dimension (n_ze, N_ze) is (semiz,z,e). FnsToEvaluate inputs are
% (d,aprime,a,semiz,z,e,...). AgentDistPath is then [n_a,n_semiz,(n_z),(n_e),T] and PolicyPath [rows,n_a,n_semiz,(n_z),(n_e),T].
N_semiz=prod(simoptions.n_semiz);
N_zeonly=N_ze; % (z,e) alone, without semiz
if N_semiz==0
    l_semiz=0;
else
    l_semiz=length(simoptions.n_semiz);
    if n_d(1)==0
        error('EvalFnOnTransPath_AggVars_InfHorz: semi-exogenous states (simoptions.n_semiz) require a decision variable: the last simoptions.l_dsemiz decision variables determine the semiz transitions')
    end
    if simoptions.experienceasset>=1
        error('EvalFnOnTransPath_AggVars_InfHorz: semi-exogenous states (simoptions.n_semiz) are not yet implemented together with an experience asset')
    end
    N_ze=N_semiz*N_ze;
    if N_z==0 && N_e==0
        n_ze=simoptions.n_semiz;
    else
        n_ze=[simoptions.n_semiz,n_ze];
    end
end

%%
% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);

%%
AggVarNames=fieldnames(FnsToEvaluate);
for ff=1:length(AggVarNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
    if length(temp)>(l_d+l_aprime+l_a+l_semiz+l_z+l_e)
        FnsToEvaluateParamNames(ff).Names={temp{l_d+l_aprime+l_a+l_semiz+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,semiz,z,e)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluateCell{ff}=FnsToEvaluate.(AggVarNames{ff});
end
% For the subfunctions we want the following
simoptions.outputasstructure=0;
simoptions.AggVarNames=AggVarNames;

%% Check if using _tminus1 and/or _tplus1 variables.
[tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames,ParamPathNames,{},transpathoptions);


%% The _tminus1 and _tplus1 values come from transpathoptions.initialvalues
% Checked here rather than at first use, so that a missing initialvalues names itself
% instead of surfacing as 'Unrecognized field name' from inside the loop over t.
if (use_tminus1price==1 || use_tminus1params==1 || use_tminus1AggVars==1) && ~isfield(transpathoptions,'initialvalues')
    error('EvalFnOnTransPath_AggVars_InfHorz: a _tminus1 input is used, so transpathoptions.initialvalues must be given (it supplies the period-1 values)')
end
%%
a_gridvals=CreateGridvals(n_a,a_grid,1);

PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_ze,T]);
% Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
% (PolicyInd2Val_InfHorz_TPath takes the exogenous sizes, (semiz,z,e) together, as its n_z input)
PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_ze,T,d_grid,a_grid,simoptions,1);
if N_z==0 && N_e==0 && N_semiz==0
    PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,T]),[2,1,3]); %[N_a,l_d+l_a,T]
else
    PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_ze,T]),[2,3,1,4]); %[N_a,N_z*N_e,l_d+l_a,T]
end

%% Set up exogenous shock processes
% gridpiboth=1: FnsToEvaluate use the grid and not the transition matrix, which is why pi_z is
% passed as []. The setup writes zpathtrivial and z_gridvals_T back into transpathoptions, which is
% why its output is taken; the caller's own fields are left untouched.
% ze_gridvals is the (z,e) grid (z varies first), and transpathoptions.zepathtrivial=0 puts its path in
% transpathoptions.ze_gridvals_T (when either z or e varies over the path).
[~, ~, ~, ~, ~, ~, ze_gridvals, transpathoptions, simoptions]=ExogShockSetup_InfHorz_TPath(n_z,z_grid,[],Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,simoptions,1);
if N_semiz>0
    % The semiz grid (fixed over the path), put in front of the (z,e) grid, semiz fastest
    [semiz_gridvals, ~, transpathoptions, ~]=SemiExogShockSetup_InfHorz_TPath(n_d,d_grid,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,simoptions,1);
    if N_z==0 && N_e==0
        ze_gridvals=semiz_gridvals;
    else
        ze_gridvals=[repmat(semiz_gridvals,N_zeonly,1),repelem(ze_gridvals,N_semiz,1)];
    end
end

%%
AgentDistPath=reshape(AgentDistPath,[N_a*N_ze,T]);

AggVarsPath=zeros(length(AggVarNames),T,'gpuArray');

for tt=1:T
    % The _tminus1 values first, while Parameters still holds the previous period's prices and parameters
    % (setting them after period tt is written in would make every _tminus1 equal to period tt)
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            if tt>1
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
            else
                Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
    end
    if use_tminus1params==1
        for pp=1:length(tminus1paramNames)
            if tt>1
                Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
            else
                Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
            end
        end
    end
    if use_tminus1AggVars==1
        for pp=1:length(tminus1AggVarsNames)
            if tt>1
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=AggVarsPath(strcmp(AggVarNames,tminus1AggVarsNames{pp}),tt-1);
            else
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
            end
        end
    end

    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end

    if transpathoptions.zepathtrivial==0
        if N_semiz==0
            ze_gridvals=transpathoptions.ze_gridvals_T(:,:,tt);
        else
            ze_gridvals=[repmat(semiz_gridvals,N_zeonly,1),repelem(transpathoptions.ze_gridvals_T(:,:,tt),N_semiz,1)]; % (semiz,z,e), semiz fastest
        end
    end
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            % Period T is the final stationary eqm, so the price after it is the same as at T
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(min(tt+1,T),PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
    end

    AgentDist=AgentDistPath(:,tt);

    if N_z==0 && N_e==0 && N_semiz==0
        PolicyValuesPermute=PolicyValuesPath(:,:,tt);
        AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars_noz(AgentDist, PolicyValuesPermute, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames,AggVarNames, n_a, a_gridvals,0);
    else
        PolicyValuesPermute=PolicyValuesPath(:,:,:,tt);
        AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(AgentDist, PolicyValuesPermute, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames,AggVarNames, n_a, n_ze, a_gridvals, ze_gridvals,0);
    end

    AggVarsPath(:,tt)=AggVars;
end


%%
% Change the output into a structure
AggVarsPath2=AggVarsPath;
clear AggVarsPath
AggVarsPath=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
for ff=1:length(AggVarNames)
    AggVarsPath.(AggVarNames{ff}).Mean=AggVarsPath2(ff,:);
end


end
