function AgentDistPath=AgentDistOnTransPath_InfHorz(AgentDist_initial, PricePath,ParamPath, PolicyPath,n_d,n_a,n_z,pi_z,T,Parameters,transpathoptions,simoptions)
% transpathoptions and simoptions are optional inputs
% iid e shocks are given in simoptions (n_e, e_grid, pi_e), as for StationaryDist_InfHorz. pi_e may be [N_e,T],
% in which case pi_e(:,t) is the distribution of the e realized IN period t.

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

%% Check which transpathoptions have been used, set all others to defaults
if exist('transpathoptions','var')==0
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.verbose=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'verbose')
        transpathoptions.verbose=0;
    end
end

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
% Read here, because ExogShockSetup_InfHorz_TPath below removes n_e from simoptions
if ~isfield(simoptions,'n_e')
    simoptions.n_e=0;
end
n_e=simoptions.n_e;
N_e=prod(n_e);
if N_e>0 && simoptions.experienceasset>=1
    error('AgentDistOnTransPath_InfHorz(): experienceasset together with an e variable is not yet implemented')
end
% Semi-exogenous states (simoptions.n_semiz): the last simoptions.l_dsemiz decision variables (d2) drive the semiz
% transitions pi_semiz(semiz,semiz',d2). Needs simoptions.d_grid (as StationaryDist_InfHorz does), and SemiExoStateFn
% or pi_semiz, and semiz_grid.
if ~isfield(simoptions,'n_semiz')
    simoptions.n_semiz=0;
end
n_semiz=simoptions.n_semiz;
N_semiz=prod(n_semiz);
if N_semiz>0
    if N_d==0
        error('AgentDistOnTransPath_InfHorz(): semi-exogenous states (simoptions.n_semiz) require a decision variable: the last simoptions.l_dsemiz decision variables determine the semiz transitions')
    end
    if l_a>1
        error('AgentDistOnTransPath_InfHorz(): semi-exogenous states (simoptions.n_semiz) only allow a single endogenous state (cannot have length(n_a)>1)')
    end
    if simoptions.experienceasset>=1
        error('AgentDistOnTransPath_InfHorz(): semi-exogenous states (simoptions.n_semiz) are not yet implemented together with an experience asset')
    end
    if ~isfield(simoptions,'d_grid')
        error('AgentDistOnTransPath_InfHorz(): when using simoptions.n_semiz you must declare simoptions.d_grid (needed to create the semiz transitions, which depend on the decision variable)')
    end
    if simoptions.gridinterplayer==1 && ~isfield(simoptions,'ngridinterp')
        error('When using simoptions.gridinterplayer=1 you must set simoptions.ngridinterp')
    end
end
% The agent dist keeps one exogenous dimension, (z,e) together with z first, which is a singleton when
% there is neither. With semiz it is (semiz,z,e), semiz first.
N_ze=max(N_z,1)*max(N_e,1);
if N_semiz>0
    N_ze=N_semiz*N_ze;
end

%% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);

%% Because of Tan improvement the AgentDist is done as a spare cpu matrix, and only afterwards do we store it on gpu
AgentDistPath=zeros(N_a*N_ze,T,'gpuArray');
% Call AgentDist the current periods distn
AgentDist=gather(sparse(reshape(AgentDist_initial,[N_a*N_ze,1])));
% gridpiboth=2: the agent dist wants the transition probabilities (including the sparse copy) and
% not the grid, which is why z_grid is passed as []. The setup overwrites zpathtrivial and pi_z_T
% in transpathoptions, which is how the path-varying z is passed on to the loops below.
% Same for e: transpathoptions.epathtrivial=0 puts the path in transpathoptions.pi_e_T.
[~, pi_z, pi_z_sparse, ~, ~, pi_e_sparse, ~, transpathoptions, simoptions]=ExogShockSetup_InfHorz_TPath(n_z,[],pi_z,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,simoptions,2);
AgentDistPath(:,1)=gpuArray(full(AgentDist));

if N_semiz>0
    % pi_semiz [N_semiz,N_semiz,N_dsemiz] (cpu); when transpathoptions.semizpathtrivial=0 the path is in
    % transpathoptions.pi_semiz_T [N_semiz,N_semiz,N_dsemiz,T] (cpu), slice tt being the transition from tt to tt+1
    [~, pi_semiz, transpathoptions, simoptions]=SemiExogShockSetup_InfHorz_TPath(n_d,simoptions.d_grid,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,simoptions,2);
    N_dsemiz=simoptions.setup_semiexo.N_dsemiz;
    n_d2=simoptions.setup_semiexo.n_d2;
    l_d1=simoptions.setup_semiexo.l_d1;
    l_d2=simoptions.setup_semiexo.l_d2;

    % PolicyPath is [l_d1+l_d2+l_a(+2),n_a,n_semiz,(n_z),(n_e),T]: d1 rows, then d2 rows, then aprime (then the L2
    % index and L2flag with the grid interpolation layer)
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a*N_ze,T]);
    Policy_aprimePath=gather(reshape(PolicyPath(l_d+1,:,:),[N_a*N_ze,T])); % aprime index
    Policy_dsemizPath=reshape(PolicyPath(l_d1+1,:,:),[N_a*N_ze,T]); % joint d2 index
    for ii=2:l_d2
        Policy_dsemizPath=Policy_dsemizPath+prod(n_d2(1:ii-1))*(reshape(PolicyPath(l_d1+ii,:,:),[N_a*N_ze,T])-1);
    end
    Policy_dsemizPath=gather(Policy_dsemizPath);
    if simoptions.gridinterplayer==1
        % Lower and upper grid points, and their probabilities from the L2 index (with the L2flag override)
        L2index=reshape(PolicyPath(l_d+l_a+1,:,:),[N_a*N_ze,1,T]);
        L2flag=reshape(PolicyPath(l_d+l_a+2,:,:),[N_a*N_ze,1,T]); % 1=force lower, 2=usual, 3=force upper
        L2index(L2flag==1)=1;                                % force all weight to lower
        L2index(L2flag==3)=1+simoptions.ngridinterp+1;       % force all weight to upper
        PolicyProbsPath=zeros(N_a*N_ze,2,T);
        PolicyProbsPath(:,2,:)=gather((L2index-1)/(1+simoptions.ngridinterp)); % probability of upper grid point
        PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
        Policy_aprimePath=reshape(Policy_aprimePath,[N_a*N_ze,1,T]); % reinterpret as lower grid index
        Policy_aprimePath=repelem(Policy_aprimePath,1,2,1);
        Policy_aprimePath(:,2,:)=Policy_aprimePath(:,2,:)+1; % upper grid index
    end

    for tt=1:T-1
        if transpathoptions.semizpathtrivial==0
            pi_semiz=transpathoptions.pi_semiz_T(:,:,:,tt);
        end
        if N_z>0 && transpathoptions.zpathtrivial==0
            pi_z_sparse=sparse(gather(transpathoptions.pi_z_T(:,:,tt)));
        end
        if N_e>0 && transpathoptions.epathtrivial==0
            pi_e_sparse=sparse(gather(transpathoptions.pi_e_T(:,tt+1))); % the e of period tt+1
        end
        if simoptions.gridinterplayer==0
            if N_z==0 && N_e==0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_SemiExo_noz_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,tt),N_dsemiz,N_a,N_semiz,pi_semiz);
            elseif N_z>0 && N_e==0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_SemiExo_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,tt),N_dsemiz,N_a,N_semiz,N_z,pi_semiz,pi_z_sparse);
            elseif N_z==0 && N_e>0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_SemiExo_noz_e_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,tt),N_dsemiz,N_a,N_semiz,N_e,pi_semiz,pi_e_sparse);
            elseif N_z>0 && N_e>0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_SemiExo_e_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,tt),N_dsemiz,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z_sparse,pi_e_sparse);
            end
        else
            if N_z==0 && N_e==0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_SemiExo_noz_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,:,tt),PolicyProbsPath(:,:,tt),2,N_dsemiz,N_a,N_semiz,pi_semiz);
            elseif N_z>0 && N_e==0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_SemiExo_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,:,tt),PolicyProbsPath(:,:,tt),2,N_dsemiz,N_a,N_semiz,N_z,pi_semiz,pi_z_sparse);
            elseif N_z==0 && N_e>0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_SemiExo_noz_e_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,:,tt),PolicyProbsPath(:,:,tt),2,N_dsemiz,N_a,N_semiz,N_e,pi_semiz,pi_e_sparse);
            elseif N_z>0 && N_e>0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_SemiExo_e_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,:,tt),PolicyProbsPath(:,:,tt),2,N_dsemiz,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z_sparse,pi_e_sparse);
            end
        end
        AgentDistPath(:,tt+1)=gpuArray(full(AgentDist));
    end

elseif simoptions.experienceasset==0
    l_aprime=l_a; % standard endogenous states

    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_ze,T]);
    if l_a==1
        PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:),[N_a*N_ze,T]); % aprime index
    elseif l_a==2
        PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:)+n_a(1)*(PolicyPath(l_d+2,:,:)-1),[N_a*N_ze,T]);
    elseif l_a==3
        PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:)+n_a(1)*(PolicyPath(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(PolicyPath(l_d+3,:,:)-1),[N_a*N_ze,T]);
    elseif l_a==4
        PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:)+n_a(1)*(PolicyPath(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(PolicyPath(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyPath(l_d+4,:,:)-1),[N_a*N_ze,T]);
    end
    % The first step of the Tan improvement maps this period's state into (aprime,z), next period's e being
    % independent of everything. So the index is aprime+N_a*(z-1), repeated over e, and just aprime without z.
    if N_z>0
        PolicyaprimezPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1),max(N_e,1),1);
    else
        PolicyaprimezPath=PolicyaprimePath;
    end

    if simoptions.gridinterplayer==0
        II1=(1:1:N_a*N_ze); % Index for this period (a,z,e)
        IIones=ones(N_a*N_ze,1); % Next period 'probabilities'
        for tt=1:T-1
            if N_z>0 && transpathoptions.zpathtrivial==0
                pi_z_sparse=sparse(gather(transpathoptions.pi_z_T(:,:,tt)));
            end
            if N_e>0 && transpathoptions.epathtrivial==0
                pi_e_sparse=sparse(gather(transpathoptions.pi_e_T(:,tt+1))); % the e of period tt+1
            end
            if N_z==0 && N_e==0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_noz_raw(AgentDist,PolicyaprimezPath(:,tt),II1,IIones,N_a);
            elseif N_z>0 && N_e==0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_raw(AgentDist,PolicyaprimezPath(:,tt),II1,IIones,N_a,N_z,pi_z_sparse);
            elseif N_z==0 && N_e>0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_noz_e_raw(AgentDist,PolicyaprimezPath(:,tt),II1,IIones,N_a,N_e,pi_e_sparse);
            elseif N_z>0 && N_e>0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_e_raw(AgentDist,PolicyaprimezPath(:,tt),II1,IIones,N_a,N_z,N_e,pi_z_sparse,pi_e_sparse);
            end
            AgentDistPath(:,tt+1)=gpuArray(full(AgentDist));
        end
    elseif simoptions.gridinterplayer==1
        PolicyProbsPath=zeros(N_a*N_ze,2,T,'gpuArray'); % preallocate
        II2=([1:1:N_a*N_ze; 1:1:N_a*N_ze]'); % Index for this period (a,z,e), note the 2 copies

        PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_ze,1,T]); % reinterpret this as lower grid index
        PolicyaprimezPath=repelem(PolicyaprimezPath,1,2,1); % create copy that will be the upper grid index
        PolicyaprimezPath(:,2,:)=PolicyaprimezPath(:,2,:)+1; % upper grid index
        L2index=reshape(PolicyPath(l_d+l_aprime+1,:,:),[N_a*N_ze,1,T]); % L2 index
        L2flag=reshape(PolicyPath(l_d+l_aprime+2,:,:),[N_a*N_ze,1,T]); % 1=force lower, 2=usual, 3=force upper
        L2index(L2flag==1)=1;                                % force all weight to lower
        L2index(L2flag==3)=1+simoptions.ngridinterp+1;       % force all weight to upper
        PolicyProbsPath(:,2,:)=L2index;
        PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
        PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point

        for tt=1:T-1
            if N_z>0 && transpathoptions.zpathtrivial==0
                pi_z_sparse=sparse(gather(transpathoptions.pi_z_T(:,:,tt)));
            end
            if N_e>0 && transpathoptions.epathtrivial==0
                pi_e_sparse=sparse(gather(transpathoptions.pi_e_T(:,tt+1))); % the e of period tt+1
            end
            if N_z==0 && N_e==0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_noz_raw(AgentDist,PolicyaprimezPath(:,:,tt),II2,PolicyProbsPath(:,:,tt),N_a);
            elseif N_z>0 && N_e==0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_raw(AgentDist,PolicyaprimezPath(:,:,tt),II2,PolicyProbsPath(:,:,tt),N_a,N_z,pi_z_sparse);
            elseif N_z==0 && N_e>0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_noz_e_raw(AgentDist,PolicyaprimezPath(:,:,tt),II2,PolicyProbsPath(:,:,tt),N_a,N_e,pi_e_sparse);
            elseif N_z>0 && N_e>0
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_e_raw(AgentDist,PolicyaprimezPath(:,:,tt),II2,PolicyProbsPath(:,:,tt),N_a,N_z,N_e,pi_z_sparse,pi_e_sparse);
            end
            AgentDistPath(:,tt+1)=gpuArray(full(AgentDist));
        end
    end
elseif simoptions.experienceasset>=1

    % Split decision variables into the standard ones and the one relevant to the experience asset
    n_d2=n_d(end); % n_d2 is the decision variable that influences next period vale of the experience asset

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

    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);

    % Precompute
    Policy_a2prime=zeros(N_a,N_z,2,'gpuArray'); % the lower grid point
    PolicyProbs=zeros(N_a,N_z,2,'gpuArray'); % preallocate
    Policy_aprime=zeros(N_a,N_z,2,'gpuArray'); % preallocate
    II2=([1:1:N_a*N_z; 1:1:N_a*N_z]'); % Index for this period (a,z), note the 2 copies

    if simoptions.gridinterplayer==0

        for tt=1:T-1
            if transpathoptions.zpathtrivial==0
                pi_z_sparse=sparse(gather(transpathoptions.pi_z_T(:,:,tt)));
            end

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
            [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset(Policy,simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_z, d_grid, a2_grid, aprimeFnParamsVec);
            % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z]
            % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
            Policy_a2prime(:,:,1)=a2primeIndexes; % lower grid point
            Policy_a2prime(:,:,2)=a2primeIndexes+1; % upper grid point
            PolicyProbs(:,:,1)=a2primeProbs; % probability of lower grid point
            PolicyProbs(:,:,2)=1-a2primeProbs; % probability of upper grid point

            if l_a1==0 % just experienceasset
                Policy_aprime=Policy_a2prime;
            elseif l_a1==1 % one other asset, then experience asset
                Policy_aprime(:,:,1)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a1*(Policy_a2prime(:,:,1)-1);
                Policy_aprime(:,:,2)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a1*Policy_a2prime(:,:,1); % Note: upper grid point minus 1 is anyway just lower grid point
            end
            PolicyaprimezPath=reshape(Policy_aprime+N_a*(0:1:N_z-1),[N_a*N_z,2]);

            AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_raw(AgentDist,PolicyaprimezPath,II2,PolicyProbs,N_a,N_z,pi_z_sparse);
            AgentDistPath(:,tt+1)=gpuArray(full(AgentDist));
        end

    elseif simoptions.gridinterplayer==1
        % Not yet implemented. experienceasset already splits the mass over two a2prime points and the
        % grid interpolation layer splits it over two a1prime points, so this branch needs the 2-by-2
        % product of the two, rather than a copy of either neighbouring branch.
        % Erroring rather than falling through: AgentDistPath is preallocated to zeros and only the
        % t=1 column is filled before this point, so an empty branch here returns an all-zero agent
        % distribution for every later period, which looks like a converged answer.
        error('AgentDistOnTransPath_InfHorz(): experienceasset together with vfoptions.gridinterplayer=1 is not yet implemented')
    end
end

if N_semiz>0
    if N_z==0 && N_e==0
        AgentDistPath=reshape(AgentDistPath,[n_a,n_semiz,T]);
    elseif N_z>0 && N_e==0
        AgentDistPath=reshape(AgentDistPath,[n_a,n_semiz,n_z,T]);
    elseif N_z==0 && N_e>0
        AgentDistPath=reshape(AgentDistPath,[n_a,n_semiz,n_e,T]);
    elseif N_z>0 && N_e>0
        AgentDistPath=reshape(AgentDistPath,[n_a,n_semiz,n_z,n_e,T]);
    end
elseif N_z==0 && N_e==0
    AgentDistPath=reshape(AgentDistPath,[n_a,T]);
elseif N_z>0 && N_e==0
    AgentDistPath=reshape(AgentDistPath,[n_a,n_z,T]);
elseif N_z==0 && N_e>0
    AgentDistPath=reshape(AgentDistPath,[n_a,n_e,T]);
elseif N_z>0 && N_e>0
    AgentDistPath=reshape(AgentDistPath,[n_a,n_z,n_e,T]);
end


end
