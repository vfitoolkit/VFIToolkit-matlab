function AggVarsPath=EvalFnOnTransPath_AggVars_Case1(FnsToEvaluate,AgentDistPath,PolicyPath,PricePath,ParamPath, Parameters, T, n_d, n_a, n_z, d_grid, a_grid,z_grid,simoptions)
% AggVarsPath is T periods long (periods 0 (before the reforms are announced) & T are the initial and final values).

if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.experienceasset=0;
    simoptions.gridinterplayer=0;
    simoptions.n_e=0;
    simoptions.n_semiz=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
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
    if length(temp)>(l_d+l_a+l_a+l_z)
        FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluateCell{ff}=FnsToEvaluate.(AggVarNames{ff});
end
% For the subfunctions we want the following
simoptions.outputasstructure=0;
simoptions.AggVarNames=AggVarNames;

%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate)
    [tplus1priceNames,tminus1priceNames,~,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames);
    % tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk.
    % But omit tminus1AggVarsNames as AggVars are anyway not allowed to take AggVars as inputs.
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tplus1pricePathkk=[];
end

use_tplus1price=0;
if ~isempty(tplus1priceNames)
    use_tplus1price=1;
end
use_tminus1price=0;
if ~isempty(tminus1priceNames)
    use_tminus1price=1;
    for tt=1:length(tminus1priceNames)
        if ~isfield(simoptions.initialvalues,tminus1priceNames{tt})
            dbstack
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{tt})
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

%%
a_gridvals=CreateGridvals(n_a,a_grid,1);
z_gridvals=CreateGridvals(n_z,z_grid,1);

PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);
% Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_z,T,d_grid,a_grid,simoptions,1);
PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_z,T]),[2,3,1,4]); %[N_a,N_z,l_d+l_a,T-1]

%%
AgentDistPath=reshape(AgentDistPath,[N_a,N_z,T]);

AggVarsPath=zeros(length(AggVarNames),T,'gpuArray');

for tt=1:T
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            if tt>1
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
            else
                Parameters.([tminus1priceNames{pp},'_tminus1'])=simoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
    end
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
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
