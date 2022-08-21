function AggVarsPath=EvalFnOnTransPath_AggVars_Case1_Direct(FnsToEvaluate,PricePath,ParamPath, Parameters, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn,transpathoptions)
% AggVarsPath is T periods long (periods 0 (before the reforms are announced) & T are the initial and final values).

if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    transpathoptions.lowmemory=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if isfield(transpathoptions,'lowmemory')==0
        transpathoptions.lowmemory=0;
    end
end

l_d=length(n_d);
if n_d(1)==0
    l_d=0;
end
l_a=length(n_a);
l_z=length(n_z);

%%
% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
PricePathNames=fieldnames(PricePath);
PricePathStruct=PricePath; 
PricePathSizeVec=zeros(1,length(PricePathNames)); % Allows for a given price param to depend on age (or permanent type)
for tt=1:length(PricePathNames)
    temp=PricePathStruct.(PricePathNames{tt});
    tempsize=size(temp);
    PricePathSizeVec(tt)=tempsize(tempsize~=T); % Get the dimension which is not T
end
PricePathSizeVec=cumsum(PricePathSizeVec);
if length(PricePathNames)>1
    PricePathSizeVec=[[1,PricePathSizeVec(1:end-1)+1];PricePathSizeVec];
else
    PricePathSizeVec=[1;PricePathSizeVec];
end
PricePath=zeros(T,PricePathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for tt=1:length(PricePathNames)
    if size(PricePathStruct.(PricePathNames{tt}),1)==T
        PricePath(:,PricePathSizeVec(1,tt):PricePathSizeVec(2,tt))=PricePathStruct.(PricePathNames{tt});
    else % Need to transpose
        PricePath(:,PricePathSizeVec(1,tt):PricePathSizeVec(2,tt))=PricePathStruct.(PricePathNames{tt})';
    end
    %     PricePath(:,ii)=PricePathStruct.(PricePathNames{ii});
end

ParamPathNames=fieldnames(ParamPath);
ParamPathStruct=ParamPath;
ParamPathSizeVec=zeros(1,length(ParamPathNames)); % Allows for a given price param to depend on age (or permanent type)
for tt=1:length(ParamPathNames)
    temp=ParamPathStruct.(ParamPathNames{tt});
    tempsize=size(temp);
    ParamPathSizeVec(tt)=tempsize(tempsize~=T); % Get the dimension which is not T
end
ParamPathSizeVec=cumsum(ParamPathSizeVec);
if length(ParamPathNames)>1
    ParamPathSizeVec=[[1,ParamPathSizeVec(1:end-1)+1];ParamPathSizeVec];
else
    ParamPathSizeVec=[1;ParamPathSizeVec];
end
ParamPath=zeros(T,ParamPathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for tt=1:length(ParamPathNames)
    if size(ParamPathStruct.(ParamPathNames{tt}),1)==T
        ParamPath(:,ParamPathSizeVec(1,tt):ParamPathSizeVec(2,tt))=ParamPathStruct.(ParamPathNames{tt});
    else % Need to transpose
        ParamPath(:,ParamPathSizeVec(1,tt):ParamPathSizeVec(2,tt))=ParamPathStruct.(ParamPathNames{tt})';
    end
%     ParamPath(:,ii)=ParamPathStruct.(ParamPathNames{ii});
end

%%
% Create ReturnFnParamNames
temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_a+l_a+l_z)
    ReturnFnParamNames={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    ReturnFnParamNames={};
end


if ~isstruct(FnsToEvaluate)
    error('Transition paths only work with version 2+ (FnsToEvaluate has to be a structure)')
end
AggVarNames=fieldnames(FnsToEvaluate);
for ff=1:length(AggVarNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
    if length(temp)>(l_d+l_a+l_a+l_z)
        FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
end
FnsToEvaluate=FnsToEvaluate2;
% For the subfunctions we want the following
simoptions.outputasstructure=0;
simoptions.AggVarNames=AggVarNames;

%%
if l_d==0
    if transpathoptions.lowmemory==1
        AggVarsPath=EvalFnOnTransPath_AggVars_Case1_lowmem_no_d(FnsToEvaluate, FnsToEvaluateParamNames,PricePath,PricePathNames,PricePathSizeVec, ParamPath, ParamPathNames,ParamPathSizeVec, Parameters, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames, simoptions, transpathoptions);
    elseif transpathoptions.lowmemory==0
        AggVarsPath=EvalFnOnTransPath_AggVars_Case1_no_d(FnsToEvaluate, FnsToEvaluateParamNames,PricePath,PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, Parameters, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames, simoptions, transpathoptions);
    end
else
    if transpathoptions.lowmemory==1
        AggVarsPath=EvalFnOnTransPath_AggVars_Case1_lowmem(FnsToEvaluate, FnsToEvaluateParamNames,PricePath,PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, Parameters, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames, simoptions, transpathoptions);
    elseif transpathoptions.lowmemory==0
        AggVarsPath=EvalFnOnTransPath_AggVars_Case1_raw(FnsToEvaluate, FnsToEvaluateParamNames,PricePath,PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, Parameters, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames, simoptions, transpathoptions);
    end
end


end
