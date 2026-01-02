function SimPanelValues=SimPanelValues_TransPath_Case1(PolicyPath, PricePath, ParamPath, T, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, FnsToEvaluate, Parameters, simoptions, transpathoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 

% Simulates a panel based on PolicyPath of 'numbersims' agents of length
% 'T' beginning from randomly drawn InitialDist.
% SimPanelValues is a 3-dimensional matrix with first dimension being the
% number of 'variables' to be simulated, second dimension is FHorz, and
% third dimension is the number-of-simulations
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)

%% Check which transpathoptions and simoptions have been declared, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=2;
    transpathoptions.lowmemory=0;
    transpathoptions.verbose=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=2;
    end
    if isfield(transpathoptions,'lowmemory')==0
        transpathoptions.lowmemory=0;
    end
    if isfield(transpathoptions,'verbose')==0
        transpathoptions.verbose=0;
    end
end

if exist('simoptions','var')==1
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=2;
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'simperiods')==0
        simoptions.simperiods=T; % This can be made shorter, but not longer
    end
    if isfield(simoptions,'numbersims')==0
        simoptions.numbersims=10^3;
    end    
else
    %If simoptions is not given, just use all the defaults
    simoptions.parallel=2;
    simoptions.verbose=0;
    simoptions.simperiods=T;
    simoptions.numbersims=10^3;
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
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

%%
% Move everything to cpu for what remains.
d_grid=gather(d_grid);
a_grid=gather(a_grid);
z_grid=gather(z_grid);
PricePath=gather(PricePath);
ParamPath=gather(ParamPath);

%% Simulate the indexes
PolicyPath=KronPolicyIndexes_TransPath_Case1(PolicyPath, n_d, n_a, n_z,T);
if l_d==0
    fprintf('SimPanelValues_TransPath_Case1 with no d variable has not yet been implemented: please email me if you want to be able to do this \n')
    % SimPanelIndexes=SimPanelIndexes_TransPath_Case1_nod(PolicyPath, PricePath, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, Parameters, transpathoptions,simoptions);
    % NOTE: ACTUALLY, JUST NEED TO IMPLEMENT THE SimPanelIndexes FOR no_d,
    % THE LOWER PART OF THIS SCRIPT ALREADY ALLOWS FOR no d variable
    dbstack
    return
else
    if transpathoptions.lowmemory==1
        % The lowmemory option is going to use gpu (but loop over z instead of parallelize) for value fn.
        SimPanelIndexes=SimPanelIndexes_TransPath_Case1_lowmem(PolicyPath, T, AgentDist_initial, n_d, n_a, n_z, pi_z, simoptions);
    else
        SimPanelIndexes=SimPanelIndexes_TransPath_Case1(PolicyPath, T, AgentDist_initial, n_d, n_a, n_z, pi_z, simoptions);
    end
end

%% Switch from the indexes to the values themselves
SimPanelValues_core=zeros(l_d+l_a+l_a+l_z, simoptions.simperiods, simoptions.numbersims);
for d_c=1:l_d
    if d_c==1
        temp=d_grid(1:n_d(d_c));
    else
        temp=d_grid(sum(n_d(1:d_c-1)):sum(n_d(1:d_c)));
    end
    SimPanelValues_core(d_c,:,:)=temp(SimPanelIndexes(d_c,:,:));
end
for a_c=1:l_a
    if a_c==1
        temp=a_grid(1:n_a(a_c));
    else
        temp=a_grid(sum(n_a(1:a_c-1)):sum(n_a(1:a_c)));
    end
    SimPanelValues_core(a_c+l_d,:,:)=temp(SimPanelIndexes(a_c+l_d,:,:)); % aprime
    SimPanelValues_core(a_c+l_d+l_a,:,:)=temp(SimPanelIndexes(a_c+l_d+l_a,:,:)); % a    
end
for z_c=1:l_z
    if z_c==1
        temp=z_grid(1:n_z(z_c));
    else
        temp=z_grid(sum(n_z(1:z_c-1)):sum(n_z(1:z_c)));
    end
    SimPanelValues_core(z_c+l_d+l_a+l_a,:,:)=temp(SimPanelIndexes(z_c+l_d+l_a+l_a,:,:));
end


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
SimPanelValues=zeros(length(FnsToEvaluate), simoptions.simperiods, simoptions.numbersims);

% THIS SHOULD BE PARFOR, BUT MATLAB CANNOT REALISE Parameters IS A STRUCTURE.
% I SHOULD parfor SOMETHING LIKE THE INNER LOOPS OVER ii
for tt=1:simoptions.simperiods
    SimPanelValues_tt=zeros(length(FnsToEvaluate),1, simoptions.numbersims);
    SimPanelValues_core_tt=SimPanelValues_core(:,tt,:);
    
    % Set Parameters appropriately for the current tt
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
            kk1=tplus1pricePathkk(pp);
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(tt+1,PricePathSizeVec(1,kk1):PricePathSizeVec(2,kk1)); % Make is so that the time t+1 variables can be used
        end
    end
    if use_tminus1AggVars==1
        for pp=1:length(use_tminus1AggVars)
            if tt>1
                % The AggVars have not yet been updated, so they still contain previous period values
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
            else
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
            end
        end
    end
    
    if l_d==0
        for vv=1:length(FnsToEvaluate)
            if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames(vv).Names={}'
                FnsToEvaluateParamsVec=[];
            else
                FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names)';
            end
            for ii=1:simoptions.numbersims
                aprime_val=SimPanelValues_core_tt(1:l_a,1,ii);
                a_val=SimPanelValues_core_tt(1+l_a:l_a+l_a,1,ii);
                z_val=SimPanelValues_core_tt(1+l_a+l_a:l_z+l_a+l_a,1,ii);
                tempcell=num2cell([aprime_val;a_val;z_val;FnsToEvaluateParamsVec]);
                SimPanelValues_tt(vv,1,ii)=FnsToEvaluate{vv}(tempcell{:});
            end
        end
    else
        for vv=1:length(FnsToEvaluate)
            if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames(vv).Names={}'
                FnsToEvaluateParamsVec=[];
            else
                FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names)';
            end
            for ii=1:simoptions.numbersims
                d_val=SimPanelValues_core_tt(1:l_d,1,ii);
                aprime_val=SimPanelValues_core_tt(1+l_d:l_a+l_d,1,ii);
                a_val=SimPanelValues_core_tt(1+l_a+l_d:l_a+l_a+l_d,1,ii);
                z_val=SimPanelValues_core_tt(1+l_a+l_a+l_d:l_z+l_a+l_a+l_d,1,ii);
                tempcell=num2cell([d_val;aprime_val;a_val;z_val;FnsToEvaluateParamsVec]);
                SimPanelValues_tt(vv,1,ii)=FnsToEvaluate{vv}(tempcell{:});
            end
        end
    end
    
    SimPanelValues(:,tt,:)=SimPanelValues_tt;
end


%% Turn output into structure
SimPanelValues2=SimPanelValues;
clear SimPanelValues
SimPanelValues=struct();
for ff=1:length(AggVarNames)
    SimPanelValues.(AggVarNames{ff})=shiftdim(SimPanelValues2(ff,:,:),1);
end



