function AggVarsPath=EvalFnOnTransPath_AggVars_InfHorz(FnsToEvaluate,AgentDistPath,PolicyPath,PricePath,ParamPath, Parameters, T, n_d, n_a, n_z, d_grid, a_grid,z_grid,simoptions)
% AggVarsPath is T periods long (periods 0 (before the reforms are announced) & T are the initial and final values).

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
l_z=length(n_z);

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%%
% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);

%%
AggVarNames=fieldnames(FnsToEvaluate);
for ff=1:length(AggVarNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
    if length(temp)>(l_d+l_aprime+l_a+l_z)
        FnsToEvaluateParamNames(ff).Names={temp{l_d+l_aprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluateCell{ff}=FnsToEvaluate.(AggVarNames{ff});
end
% For the subfunctions we want the following
simoptions.outputasstructure=0;
simoptions.AggVarNames=AggVarNames;

%% Check if using _tminus1 and/or _tplus1 variables.
[tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames,{},{},simoptions);

%%
a_gridvals=CreateGridvals(n_a,a_grid,1);

PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);
% Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_z,T,d_grid,a_grid,simoptions,1);
PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_z,T]),[2,3,1,4]); %[N_a,N_z,l_d+l_a,T-1]

%% Set up exogenous shock processes
% gridpiboth=1: FnsToEvaluate use the grid and not the transition matrix, which is why pi_z is
% passed as []. transpathoptions is local: this command does not take one, and the setup only uses
% it to report back zpathtrivial and z_gridvals_T.
transpathoptions=struct();
[z_gridvals, ~, ~, ~, ~, ~, ~, transpathoptions, simoptions]=ExogShockSetup_InfHorz_TPath(n_z,z_grid,[],Parameters,PricePathNames,ParamPathNames,T,transpathoptions,simoptions,1);

%%
AgentDistPath=reshape(AgentDistPath,[N_a,N_z,T]);

AggVarsPath=zeros(length(AggVarNames),T,'gpuArray');

for tt=1:T
    % The _tminus1 values first, while Parameters still holds the previous period's prices and parameters
    % (setting them after period tt is written in would make every _tminus1 equal to period tt)
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            if tt>1
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
            else
                Parameters.([tminus1priceNames{pp},'_tminus1'])=simoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
    end
    if use_tminus1params==1
        for pp=1:length(tminus1paramNames)
            if tt>1
                Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
            else
                Parameters.([tminus1paramNames{pp},'_tminus1'])=simoptions.initialvalues.(tminus1paramNames{pp});
            end
        end
    end
    if use_tminus1AggVars==1
        for pp=1:length(tminus1AggVarsNames)
            if tt>1
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=AggVarsPath(strcmp(AggVarNames,tminus1AggVarsNames{pp}),tt-1);
            else
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=simoptions.initialvalues.(tminus1AggVarsNames{pp});
            end
        end
    end

    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end

    if transpathoptions.zpathtrivial==0
        z_gridvals=transpathoptions.z_gridvals_T(:,:,tt);
    end
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            % Period T is the final stationary eqm, so the price after it is the same as at T
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(min(tt+1,T),PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
    end

    PolicyValuesPermute=PolicyValuesPath(:,:,:,tt);
    AgentDist=AgentDistPath(:,:,tt);

    AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(AgentDist(:), PolicyValuesPermute, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames,AggVarNames, n_a, n_z, a_gridvals, z_gridvals,0);

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
