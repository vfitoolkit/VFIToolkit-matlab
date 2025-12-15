function AgentDistPath=AgentDistOnTransPath_Case1(AgentDist_initial, PolicyPath,n_d,n_a,n_z,pi_z,T,simoptions,Parameters,PricePath,ParamPath)
n_e=0; % NOT YET IMPLEMENTED FOR TRANSITION PATHS

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

%% Check which simoptions have been used, set all others to defaults 
if exist('simoptions','var')==0
    simoptions.verbose=0;
    simoptions.gridinterplayer=0;
    simoptions.experienceasset=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
end

%% Some setups need Parameters, PricePath and ParamPath to be input
if simoptions.experienceasset==1
    if ~exist('Parameters','var') || ~exist('PricePath','var') || ~exist('ParamPath','var')
        error('When using AgentDistOnTransPath_Case1() with experienceasset you must include Parameters, PricePath, ParamPath as inputs after simoptions')
    end

    %%
    % Note: Internally PricePath is matrix of size T-by-'number of prices'.
    % ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
    [PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);
end

%%
AgentDistPath=zeros(N_a*N_z,T,'gpuArray');
% Call AgentDist the current periods distn
AgentDist=reshape(AgentDist_initial,[N_a*N_z,1]);
pi_z_sparse=sparse(pi_z);
AgentDistPath(:,1)=AgentDist;

if simoptions.experienceasset==0
    l_aprime=l_a; % standard endogenous states
    
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);
    if l_a==1
        PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:),[N_a*N_z,T]); % aprime index
    elseif l_a==2
        PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:)+n_a(1)*(PolicyPath(l_d+2,:,:)-1),[N_a*N_z,T]);
    elseif l_a==3
        PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:)+n_a(1)*(PolicyPath(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(PolicyPath(l_d+3,:,:)-1),[N_a*N_z,T]);
    elseif l_a==4
        PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:)+n_a(1)*(PolicyPath(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(PolicyPath(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyPath(l_d+4,:,:)-1),[N_a*N_z,T]);
    end
    PolicyaprimezPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1);
    
    if simoptions.gridinterplayer==0
        II1=gpuArray(1:1:N_a*N_z); % Index for this period (a,z)
        IIones=ones(N_a*N_z,1,'gpuArray'); % Next period 'probabilities'
        for tt=1:T-1
            AgentDist=StationaryDist_InfHorz_TPath_SingleStep(AgentDist,gather(PolicyaprimezPath(:,tt)),II1,IIones,N_a,N_z,pi_z_sparse);
            AgentDistPath(:,tt+1)=AgentDist;
        end
    elseif simoptions.gridinterplayer==1
        PolicyProbsPath=zeros(N_a*N_z,2,T,'gpuArray'); % preallocate
        II2=gpuArray([1:1:N_a*N_z; 1:1:N_a*N_z]'); % Index for this period (a,z), note the 2 copies

        PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_z,1,T]); % reinterpret this as lower grid index
        PolicyaprimezPath=repelem(PolicyaprimezPath,1,2,1); % create copy that will be the upper grid index
        PolicyaprimezPath(:,2,:)=PolicyaprimezPath(:,2,:)+1; % upper grid index
        PolicyProbsPath(:,2,:)=reshape(PolicyPath(l_d+l_aprime+1,:,:),[N_a*N_z,1,T]); % L2 index
        PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
        PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point

        for tt=1:T-1
            AgentDist=StationaryDist_InfHorz_TPath_SingleStep_nProbs(AgentDist,PolicyaprimezPath(:,:,tt),II2,PolicyProbsPath(:,:,tt),N_a,N_z,pi_z_sparse);
            AgentDistPath(:,tt+1)=AgentDist;
        end
    end
elseif simoptions.experienceasset==1

    % Split decision variables into the standard ones and the one relevant to the experience asset
    n_d2=n_d(end); % n_d2 is the decision variable that influences next period vale of the experience asset
    if isscalar(n_d)
        ndvars=1; % just d2
    else
        ndvars=2; % includes d1
    end

    % Split endogenous assets into the standard ones and the experience asset
    if isscalar(n_a)
        n_a1=0;
        l_a1=0;
    else
        n_a1=n_a(1:end-1);
        l_a1=length(n_a1);
    end
    n_a2=n_a(end); % n_a2 is the experience asset

    if isfield(simoptions,'aprimeFn')
        aprimeFn=simoptions.aprimeFn;
    else
        error('To use an experience asset you must define simoptions.aprimeFn')
    end

    if isfield(simoptions,'a_grid')
        a2_grid=simoptions.a_grid(sum(n_a1)+1:end);
    else
        error('To use an experience asset you must define simoptions.a_grid')
    end
    if isfield(simoptions,'d_grid')
        d_grid=simoptions.d_grid;
    else
        error('To use an experience asset you must define simoptions.d_grid')
    end

    % aprimeFnParamNames in same fashion
    l_d2=length(n_d2);
    l_a2=length(n_a2);
    temp=getAnonymousFnInputNames(aprimeFn);
    if length(temp)>(l_d2+l_a2)
        aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
    else
        aprimeFnParamNames={};
    end
   
    PolicyPath=KronPolicyIndexes_InfHorz_TransPath_ExpAsset(PolicyPath, n_d, n_a, n_z,T,simoptions);

    % Precompute
    Policy_a2prime=zeros(N_a,N_z,2,'gpuArray'); % the lower grid point
    PolicyProbs=zeros(N_a,N_z,2,'gpuArray'); % preallocate
    Policy_aprime=zeros(N_a,N_z,2,'gpuArray'); % preallocate
    II2=gpuArray([1:1:N_a*N_z; 1:1:N_a*N_z]'); % Index for this period (a,z), note the 2 copies
    
    if simoptions.gridinterplayer==0

        for tt=1:T-1
            % Get the current optimal policy, and iterate the agent dist
            Policy=PolicyPath(:,:,:,tt);

            %% Update the parameters
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            whichisdforexpasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
            aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames);
            [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy,simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_z, d_grid, a2_grid, aprimeFnParamsVec);
            % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z]
            % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
            Policy_a2prime(:,:,1)=a2primeIndexes; % lower grid point
            Policy_a2prime(:,:,2)=a2primeIndexes+1; % upper grid point
            PolicyProbs(:,:,1)=a2primeProbs; % probability of lower grid point
            PolicyProbs(:,:,2)=1-a2primeProbs; % probability of upper grid point

            if l_a1==0 % just experienceasset
                Policy_aprime=Policy_a2prime;
            elseif l_a1==1 % one other asset, then experience asset
                Policy_aprime(:,:,1)=reshape(Policy(ndvars+1,:,:),[N_a,N_z,1])+n_a1*(Policy_a2prime(:,:,1)-1);
                Policy_aprime(:,:,2)=reshape(Policy(ndvars+1,:,:),[N_a,N_z,1])+n_a1*Policy_a2prime(:,:,1); % Note: upper grid point minus 1 is anyway just lower grid point
            end
            PolicyaprimezPath=reshape(Policy_aprime+N_a*(0:1:N_z-1),[N_a*N_z,2]);
            
            AgentDist=StationaryDist_InfHorz_TPath_SingleStep_TwoProbs(AgentDist,PolicyaprimezPath,II2,PolicyProbs,N_a,N_z,pi_z_sparse);
            AgentDistPath(:,tt+1)=AgentDist;
        end

    elseif simoptions.gridinterplayer==1

    end
end

AgentDistPath=reshape(AgentDistPath,[n_a,n_z,T]);


end