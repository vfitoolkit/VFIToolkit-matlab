function AggVarsPath=EvalFnOnTransPath_AggVars_Case1(FnsToEvaluate,AgentDistPath,PolicyPath,PricePath,ParamPath, Parameters, T, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid,simoptions)
% AggVarsPath is T periods long (periods 0 (before the reforms are announced) & T are the initial and final values).

if ~exist('simoptions','var')
    %If simoptions is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    simoptions.experienceasset=0;
    simoptions.gridinterplayer=0;
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
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
    FnsToEvaluateCell{ff}=FnsToEvaluate.(AggVarNames{ff});
end
% For the subfunctions we want the following
simoptions.outputasstructure=0;
simoptions.AggVarNames=AggVarNames;

%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate)
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames);
    % tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk
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
        if ~isfield(simoptions.initialvalues,tminus1priceNames{tt})
            dbstack
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{tt})
        end
    end
end
use_tminus1AggVars=0;
if length(tminus1AggVarsNames)>0
    use_tminus1AggVars=1;
    for tt=1:length(tminus1AggVarsNames)
        if ~isfield(simoptions.initialvalues,tminus1AggVarsNames{tt})
            dbstack
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1AggVarsNames{tt})
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

%%
d_gridvals=CreateGridvals(n_d,d_grid,1);
a_gridvals=CreateGridvals(n_a,a_grid,1);
if simoptions.experienceasset==0
    if simoptions.gridinterplayer==1
        if isscalar(n_a)
            N_aprime=N_a+(N_a-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,N_a,N_a)',a_grid(1:n_a(1)),linspace(1,N_a,N_aprime)');
            aprime_grid=temp;
            n_aprime=n_a;
        else
            N_a1prime=n_a(1)+(n_a(1)-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,n_a(1),n_a(1))',a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)');
            aprime_grid=[temp; a_grid(n_a(1)+1:end)];
            n_aprime=[N_a1prime,n_a(2:end)];
        end
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
    else
        aprime_gridvals=a_gridvals;
    end
elseif simoptions.experienceasset==1
    % omit a2 from aprime_gridvals
    if simoptions.gridinterplayer==1
        N_a1prime=n_a(1)+(n_a(1)-1)*simoptions.ngridinterp;
        temp=interp1(linspace(1,n_a(1),n_a(1))',a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)');
        if length(n_a)==2
            aprime_grid=temp; % omit a2
            n_aprime=N_a1prime; % omit a2
        elseif length(n_a)>2 % more than one a1
            aprime_grid=[temp; a_grid(n_a(1)+1:end-1)];
            n_aprime=[N_a1prime,n_a(2:end-1)];
        end
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
    else
        aprime_gridvals=CreateGridvals(n_a(1:end-1),a_grid(1:sum(n_a(1:end-1))),1); % omit a2
    end

end
z_gridvals=CreateGridvals(n_z,z_grid,1);

PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_z,T,d_gridvals,aprime_gridvals,simoptions,0);

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

    PolicyValues=PolicyValuesPath(:,:,:,tt);
    AgentDist=AgentDistPath(:,:,tt);

    AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(AgentDist(:), PolicyValues, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames,[], n_a, n_z, a_gridvals, z_gridvals,0);

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
