function [VPath,PolicyPath]=ValueFnOnTransPath_Case1(PricePath, ParamPath, T, V_final, Policy_final, Parameters, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)
% transpathoptions, vfoptions and simoptions are optional inputs

%% Check which transpathoptions have been used, set all others to defaults 
if ~exist('transpathoptions','var')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=2;
    transpathoptions.verbose=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'parallel')
        transpathoptions.parallel=2;
    end
    if ~isfield(transpathoptions,'verbose')
        transpathoptions.verbose=0;
    end
end

%% Check which vfoptions have been used, set all others to defaults 
vfoptions.parallel=2; % GPU, has to be or transpath will already have thrown an error
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.exoticpreferences='None';
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.solnmethod='purediscretization'; % Currently this does nothing
    vfoptions.divideandconquer=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if ~isfield(vfoptions,'exoticpreferences')
        vfoptions.exoticpreferences='None';
    end
    if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        if ~isfield(vfoptions,'quasi_hyperbolic')
            vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
        elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            fprintf('ERROR: when using Quasi-Hyperbolic discounting vfoptions.quasi_hyperbolic must be either Naive or Sophisticated \n')
            dbstack
            return
        end
    end
    if ~isfield(vfoptions,'polindorval')
        vfoptions.polindorval=1;
    end
    if ~isfield(vfoptions,'policy_forceintegertype')
        vfoptions.policy_forceintegertype=0;
    end
    if ~isfield(vfoptions,'solnmethod')
        vfoptions.solnmethod='purediscretization'; % Currently this does nothing
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0;
    end
end

if vfoptions.divideandconquer==1
    if ~isfield(vfoptions,'level1n')
        vfoptions.level1n=5*length(n_a);
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

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

%%

if transpathoptions.parallel~=2
    error('A GPU is required for any codes that relate to transition paths')
else
    d_grid=gpuArray(d_grid); a_grid=gpuArray(a_grid); z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    PricePath=gpuArray(PricePath);
end

if transpathoptions.verbose==1
    transpathoptions
end

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);
l_p=size(PricePath,2);

PricePathDist=Inf;
pathcounter=0;

V_final=reshape(V_final,[N_a,N_z]);
V=zeros(size(V_final),'gpuArray');
if N_d>0
    Policy=zeros(2,N_a,N_z,'gpuArray');
else
    Policy=zeros(N_a,N_z,'gpuArray');
end
if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

VKronPath=zeros(N_a,N_z,T);
VKronPath(:,:,T)=V_final;

%%
if N_d==0
    PolicyIndexesPath=zeros(N_a,N_z,T,'gpuArray'); %Periods 1 to T-1
    PolicyIndexesPath(:,:,T)=KronPolicyIndexes_Case1(Policy_final, n_d, n_a, n_z);

    % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    Vnext=V_final;
    for ttr=1:T-1 %so t=T-i

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(T-ttr,kk);
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,kk);
        end

        [V, Policy]=ValueFnIter_Case1_TPath_SingleStep(Vnext,n_d,n_a,n_z,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The VKron input is next period value fn, the VKron output is this period. Policy is kept in the form where it is just a single-value in (d,a')

        if N_d>0
            PolicyIndexesPath(:,:,:,T-ttr)=Policy;
        else
            PolicyIndexesPath(:,:,T-ttr)=Policy;
        end
        VKronPath(:,:,T-ttr)=V;
        Vnext=V;
    end
elseif N_d>0
    PolicyIndexesPath=zeros(2,N_a,N_z,T,'gpuArray'); %Periods 1 to T-1
    PolicyIndexesPath(:,:,:,T)=KronPolicyIndexes_Case1(Policy_final, n_d, n_a, n_z);

    % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    Vnext=V_final;
    for ttr=1:T-1 %so t=T-i

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(T-ttr,kk);
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,kk);
        end

        [V, Policy]=ValueFnIter_Case1_TPath_SingleStep(Vnext,n_d,n_a,n_z,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The VKron input is next period value fn, the VKron output is this period. Policy is kept in the form where it is just a single-value in (d,a')

        if N_d>0
            PolicyIndexesPath(:,:,:,T-ttr)=Policy;
        else
            PolicyIndexesPath(:,:,T-ttr)=Policy;
        end
        VKronPath(:,:,T-ttr)=V;
        Vnext=V;
    end
end




%% Unkron to get into the shape for output
VPath=reshape(VKronPath,[n_a,n_z,T]);
PolicyPath=UnKronPolicyIndexes_Case1_TransPath(PolicyIndexesPath, n_d, n_a, n_z,T,vfoptions);


end