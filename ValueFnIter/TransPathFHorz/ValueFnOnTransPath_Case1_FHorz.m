function [VPath,PolicyPath]=ValueFnOnTransPath_Case1_FHorz(PricePath, ParamPath, T, V_final, Policy_final, Parameters, n_d, n_a, n_z, N_j, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, AgeWeightsParamNames, transpathoptions, vfoptions, simoptions)
% transpathoptions, vfoptions and simoptions are optional inputs

%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=2;
    if exist('vfoptions','var')==1 % If vfoptions.exoticpreferences, then set transpathoptions to the same
        if isfield(vfoptions,'exoticpreferences')
            transpathoptions.exoticpreferences=vfoptions.exoticpreferences;
        else
            transpathoptions.exoticpreferences='None';
        end
    else
        transpathoptions.exoticpreferences='None';
    end
    transpathoptions.verbose=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=2;
    end
    if isfield(transpathoptions,'exoticpreferences')==0
        transpathoptions.exoticpreferences='None';
    end
    if isfield(transpathoptions,'verbose')==0
        transpathoptions.verbose=0;
    end
end

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.parallel=1+(gpuDeviceCount>0);
    vfoptions.returnmatrix=2;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.exoticpreferences=transpathoptions.exoticpreferences;
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    if isfield(vfoptions,'lowmemory')==0
        vfoptions.lowmemory=0;
    end
    if isfield(vfoptions,'verbose')==0
        vfoptions.verbose=0;
    end
    vfoptions.exoticpreferences=transpathoptions.exoticpreferences; % Note that if vfoptions.exoticpreferences exists then it has already been used to set transpathoptions.exoticpreferences anyway.
    if isfield(vfoptions,'polindorval')==0
        vfoptions.polindorval=1;
    end
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
    end
end

%% Check which simoptions have been used, set all others to defaults 
if exist('simoptions','var')==0
    simoptions.nsims=10^4;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    simoptions.ncores=feature('numcores'); % Number of CPU cores
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if isfield(simoptions,'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
    if isfield(simoptions,'nsims')==0
        simoptions.nsims=10^4;
    end
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'ncores')==0
        simoptions.ncores=feature('numcores'); % Number of CPU cores
    end
    if isfield(simoptions,'iterate')==0
        simoptions.iterate=1;
    end
end


%%
% Internally PricePathOld is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
PricePathStruct=PricePath; % I do this here just to make it easier for the user to read and understand the inputs.
PricePathNames=fieldnames(PricePathStruct);
ParamPathStruct=ParamPath; % I do this here just to make it easier for the user to read and understand the inputs.
ParamPathNames=fieldnames(ParamPathStruct);
if transpathoptions.parallel==2 
    PricePath=zeros(T,length(PricePathNames),'gpuArray');
    for ii=1:length(PricePathNames)
        PricePath(:,ii)=gpuArray(PricePathStruct.(PricePathNames{ii}));
    end
    ParamPath=zeros(T,length(ParamPathNames),'gpuArray');
    for ii=1:length(ParamPathNames)
        ParamPath(:,ii)=gpuArray(ParamPathStruct.(ParamPathNames{ii}));
    end
else
    PricePath=zeros(T,length(PricePathNames));
    for ii=1:length(PricePathNames)
        PricePath(:,ii)=gather(PricePathStruct.(PricePathNames{ii}));
    end
    ParamPath=zeros(T,length(ParamPathNames));
    for ii=1:length(ParamPathNames)
        ParamPath(:,ii)=gather(ParamPathStruct.(ParamPathNames{ii}));
    end
end

% % The outputted VPath and PolicyPath are T-1 periods long (periods 0 (before the reforms are announced) & T are the initial and final values; they are not created by this command and instead can be used to provide double-checks of the output (the T-1 and the final should be identical if convergence has occoured).
% if n_d(1)==0
%     PolicyPath=zeros([length(n_d),n_a,n_z,N_j,T-1],'gpuArray'); %Periods 1 to T-1
% else
%     PolicyPath=zeros([length(n_d)+length(n_a),n_a,n_z,N_j,T-1],'gpuArray'); %Periods 1 to T-1
% end
% VPath=zeros([n_a,n_z,N_j,T-1],'gpuArray'); %Periods 1 to T-1

% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 

% PricePath is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

% AgeWeightsParamNames are not actually needed as an input, but require
% them anyway to make it easier to 'copy-paste' input lists from other
% similar functions the user is likely to be using.

%% Create ReturnFnParamNames
l_d=0;
if ~isempty(n_d)
    if n_d(1)~=0
        l_d=length(n_d);
    end
end
l_a=length(n_a);
l_z=length(n_z);

temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_a+l_a+l_z)
    ReturnFnParamNames={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    ReturnFnParamNames={};
end


%%
if ~strcmp(transpathoptions.exoticpreferences,'None')
    disp('ERROR: Only transpathoptions.exoticpreferences==0 is supported by TransitionPath_Case1')
    dbstack
else
    if length(DiscountFactorParamNames)~=1
        disp('WARNING: DiscountFactorParamNames should be of length one')
        dbstack
    end
end

if transpathoptions.parallel~=2
    disp('ERROR: Only transpathoptions.parallel==2 is supported by TransitionPath_Case1')
else
    d_grid=gpuArray(d_grid); a_grid=gpuArray(a_grid); z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    PricePath=gpuArray(PricePath);
end
unkronoptions.parallel=2;

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);
l_p=size(PricePath,2);

if transpathoptions.parallel==2
    % Make sure things are on gpu where appropriate.
    if N_d>0
        d_grid=gather(d_grid);
    end
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
end

if transpathoptions.verbose==1
    transpathoptions
end

PricePathDist=Inf;
pathcounter=0;

V_final=reshape(V_final,[N_a,N_z,N_j]);
V=zeros(size(V_final),'gpuArray');
if N_d>0
    Policy=zeros(2,N_a,N_z,N_j,'gpuArray');
else
    Policy=zeros(N_a,N_z,N_j,'gpuArray');
end
if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

%%

if N_d>0
    PolicyIndexesPath=zeros(2,N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T-1
    PolicyIndexesPath(:,:,:,:,T)=KronPolicyIndexes_FHorz_Case1(Policy_final, n_d, n_a, n_z,N_j);
else
    PolicyIndexesPath=zeros(N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T-1
    PolicyIndexesPath(:,:,:,T)=KronPolicyIndexes_FHorz_Case1(Policy_final, n_d, n_a, n_z,N_j);
end

%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for i=1:T-1 %so t=T-i
    
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(T-i,kk);
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(T-i,kk);
    end
    
    [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep(Vnext,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % The VKron input is next period value fn, the VKron output is this period.
    % Policy is kept in the form where it is just a single-value in (d,a')
    
    if N_d>0
        PolicyIndexesPath(:,:,:,:,T-i)=Policy;
    else
        PolicyIndexesPath(:,:,:,T-i)=Policy;
    end
    VKronPath(:,:,:,T-i)=V;
    Vnext=V;
end

%% Unkron to get into the shape for output
VPath=reshape(VKronPath,[n_a,n_z,N_j,T-1]);
PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz(PolicyIndexesPath, n_d, n_a, n_z, N_j,T,vfoptions);



end