function AggVarsPath=EvalFnOnTransPath_AggVars_Case1(FnsToEvaluate,AgentDistPath,PolicyPath,PricePath,ParamPath, Parameters, T, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames,transpathoptions)
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

if ~exist('simoptions','var')
    %If simoptions is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
end

l_d=length(n_d);
if n_d(1)==0
    l_d=0;
end
l_a=length(n_a);
l_z=length(n_z);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

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

%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate) && isstruct(GeneralEqmEqns)
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames);
    tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tminus1AggVarsNames=[];
    tplus1pricePathkk=[];
end

use_tplus1price=0;
if length(tplus1priceNames)>0
    use_tplus1price=1;
end
use_tminus1price=0;
if length(tminus1priceNames)>0
    use_tminus1price=1;
    for tt=1:length(tminus1priceNames)
        if ~isfield(transpathoptions.initialvalues,tminus1priceNames{tt})
            fprintf('ERROR: Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{tt})
            dbstack
            break
        end
    end
end
use_tminus1AggVars=0;
if length(tminus1AggVarsNames)>0
    use_tminus1AggVars=1;
    for tt=1:length(tminus1AggVarsNames)
        if ~isfield(transpathoptions.initialvalues,tminus1AggVarsNames{tt})
            fprintf('ERROR: Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1AggVarsNames{tt})
            dbstack
            break
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

%%
PolicyPath=KronPolicyIndexes_TransPath_Case1(PolicyPath, n_d, n_a, n_z,T);
AgentDistPath=reshape(AgentDistPath,[N_a,N_z,T]);

if simoptions.parallel==2
    unkronoptions.parallel=2;
    AggVarsPath=zeros(T,length(AggVarNames),'gpuArray');
    
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
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                end
            end
        end
        if use_tplus1price==1
            for pp=1:length(tplus1priceNames)
                kk=tplus1pricePathkk(pp);
                Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
            end
        end
%         if use_tminus1AggVars==1
%             for pp=1:length(use_tminus1AggVars)
%                 if tt>1
%                     % The AggVars have not yet been updated, so they still contain previous period values
%                     Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
%                 else
%                     Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
%                 end
%             end
%         end

        if N_d>0
            Policy=PolicyPath(:,:,:,tt);
        else
            Policy=PolicyPath(:,:,tt);            
        end
        Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,unkronoptions);
        AgentDist=AgentDistPath(:,:,tt);
        AggVars=EvalFnOnAgentDist_AggVars_Case1(AgentDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, 2,simoptions);
        
        AggVarsPath(tt,:)=AggVars;
    end
elseif simoptions.parallel==1
    AggVarsPath=zeros(T,length(AggVarsNames));
    
    if N_d>0 % Has to be outside so that parfor can slice PolicyPath properly
        parfor tt=1:T
            Parameters_tt=Parameters; % This is just to help matlab figure out how to slice the parfor
            Policy=PolicyPath(:,:,:,tt);
            for kk=1:length(PricePathNames)
                Parameters_tt.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters_tt.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end
            if use_tminus1price==1
                for pp=1:length(tminus1priceNames)
                    if tt>1
                        Parameters_tt.([tminus1priceNames{pp},'_tminus1'])=Parameters_tt.(tminus1priceNames{pp});
                    else
                        Parameters_tt.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                    end
                end
            end
            if use_tplus1price==1
                for pp=1:length(tplus1priceNames)
                    ll=tplus1pricePathkk(pp);
                    Parameters_tt.([tplus1priceNames{pp},'_tplus1'])=PricePath(tt+1,PricePathSizeVec(1,ll):PricePathSizeVec(2,ll)); % Make is so that the time t+1 variables can be used
                end
            end
            %         if use_tminus1AggVars==1
            %             for pp=1:length(use_tminus1AggVars)
            %                 if tt>1
            %                     % The AggVars have not yet been updated, so they still contain previous period values
            %                     Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
            %                 else
            %                     Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
            %                 end
            %             end
            %         end
            
            Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,unkronoptions);
            AgentDist=AgentDistPath(:,:,tt);
            AggVars=EvalFnOnAgentDist_AggVars_Case1(AgentDist, Policy, FnsToEvaluate, Parameters_tt, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, 2,simoptions);
            
            AggVarsPath(tt,:)=AggVars;
        end
    else % N_d==0
        parfor tt=1:T
            Parameters_tt=Parameters; % This is just to help matlab figure out how to slice the parfor
            Policy=PolicyPath(:,:,tt);
            for kk=1:length(PricePathNames)
                Parameters_tt.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters_tt.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end
            if use_tminus1price==1
                for pp=1:length(tminus1priceNames)
                    if tt>1
                        Parameters_tt.([tminus1priceNames{pp},'_tminus1'])=Parameters_tt.(tminus1priceNames{pp});
                    else
                        Parameters_tt.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                    end
                end
            end
            if use_tplus1price==1
                for pp=1:length(tplus1priceNames)
                    ll=tplus1pricePathkk(pp);
                    Parameters_tt.([tplus1priceNames{pp},'_tplus1'])=PricePath(tt+1,PricePathSizeVec(1,ll):PricePathSizeVec(2,ll)); % Make is so that the time t+1 variables can be used
                end
            end
            %         if use_tminus1AggVars==1
            %             for pp=1:length(use_tminus1AggVars)
            %                 if tt>1
            %                     % The AggVars have not yet been updated, so they still contain previous period values
            %                     Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
            %                 else
            %                     Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
            %                 end
            %             end
            %         end
            
            Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,unkronoptions);
            AgentDist=AgentDistPath(:,:,tt);
            AggVars=EvalFnOnAgentDist_AggVars_Case1(AgentDist, Policy, FnsToEvaluate, Parameters_tt, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, 2,simoptions);
            
            AggVarsPath(tt,:)=AggVars;
        end
    end
elseif simoptions.parallel==o
    AggVarsPath=zeros(T,length(AggVarsNames));
    
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
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                end
            end
        end
        if use_tplus1price==1
            for pp=1:length(tplus1priceNames)
                kk=tplus1pricePathkk(pp);
                Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
            end
        end
%         if use_tminus1AggVars==1
%             for pp=1:length(use_tminus1AggVars)
%                 if tt>1
%                     % The AggVars have not yet been updated, so they still contain previous period values
%                     Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
%                 else
%                     Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
%                 end
%             end
%         end

        if N_d>0
            Policy=PolicyPath(:,:,:,tt);
        else
            Policy=PolicyPath(:,:,tt);            
        end
        Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,unkronoptions);
        AgentDist=AgentDistPath(:,:,tt);
        AggVars=EvalFnOnAgentDist_AggVars_Case1(AgentDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, 2,simoptions);
        
        AggVarsPath(tt,:)=AggVars;
    end
end


%%
% Change the output into a structure
AggVarsPath2=AggVarsPath;
clear AggVarsPath
AggVarsPath=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
for ff=1:length(AggVarNames)
    AggVarsPath.(AggVarNames{ff}).Mean=AggVarsPath2(:,ff);
end


end