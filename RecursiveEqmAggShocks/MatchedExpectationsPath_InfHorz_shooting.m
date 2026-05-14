function [PricePath,GEcondnPath,VPath,PolicyIndexesPath,AgentDistPath,DistMatches]=MatchedExpectationsPath_InfHorz_shooting(AggShocksPath, AggShockNames, T,SSmask_T,SSprimemask_T,SSprimemask_T_indexes,ss_ind_T, n_d, n_a, n_z, n_S, l_d, l_a, l_z, d_grid, a_grid,z_gridvals_T,z_gridvals_T_fastOLG, pi_Sprime_T, pi_z_T, pi_z_T_sim, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEPriceParamNames, GEeqnNames, GeneralEqmEqnsStruct, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, initialguessobjects, vfoptions, simoptions, recursiveeqmoptions)

N_a=prod(n_a);
N_z=prod(n_z);

N_S=prod(n_S);

l_aprime=length(n_a);

%%
if recursiveeqmoptions.verbose>=1
    % Set up some things to be used later
    pathnametitles=cell(1,2*length(GEPriceParamNames));
    for tt=1:length(GEPriceParamNames)
        pathnametitles{tt}={['Old ',GEPriceParamNames{tt}]};
        pathnametitles{tt+length(GEPriceParamNames)}={['New ',GEPriceParamNames{tt}]};
    end

    priceverbosestr='    of which: ';
    for pp=1:length(GEPriceParamNames)
        priceverbosestr=[priceverbosestr,GEPriceParamNames{pp},' is %8.6f,  '];
    end
    priceverbosestr=[priceverbosestr,' \n'];
    GEcondnverbosestr='    of which: ';
    for gg=1:length(GEeqnNames)
        GEcondnverbosestr=[GEcondnverbosestr,GEeqnNames{gg},' is %8.6f,  '];
    end
    GEcondnverbosestr=[GEcondnverbosestr,' \n'];
end

%% Setup needed for the fastOLG agent dist iterations
if simoptions.gridinterplayer==0
    II1=1:1:N_a*(T-1)*N_z;
    II2=ones(N_a*(T-1)*N_z,1);
    exceptlastj=repmat((1:1:N_a)',(T-1)*N_z,1)+repmat(repelem(N_a*(0:1:T-2)',N_a,1),N_z,1)+repelem(N_a*T*(0:1:N_z-1)',N_a*(T-1),1);
    exceptfirstj=repmat((1:1:N_a)',(T-1)*N_z,1)+repmat(repelem(N_a*(1:1:T-1)',N_a,1),N_z,1)+repelem(N_a*T*(0:1:N_z-1)',N_a*(T-1),1);
    justfirstj=repmat((1:1:N_a)',N_z,1)+N_a*T*repelem((0:1:N_z-1)',N_a,1);
elseif simoptions.gridinterplayer==1
    N_probs=2;
    II=repelem((1:1:N_a*(T-1)*N_z)',1,N_probs);
    exceptlastj=repmat((1:1:N_a)',(T-1)*N_z,1)+repmat(repelem(N_a*(0:1:T-2)',N_a,1),N_z,1)+repelem(N_a*T*(0:1:N_z-1)',N_a*(T-1),1);
    exceptfirstj=repmat((1:1:N_a)',(T-1)*N_z,1)+repmat(repelem(N_a*(1:1:T-1)',N_a,1),N_z,1)+repelem(N_a*T*(0:1:N_z-1)',N_a*(T-1),1);
    justfirstj=repmat((1:1:N_a)',N_z,1)+N_a*T*repelem((0:1:N_z-1)',N_a,1);
end
% We don't have age weights here, as the mass of agents is the same in every t=1:T.
% I'm leaving it as a object as I can imagine there will be some firm
% entry-exit models where you would want this to change over time.
AgeWeights_T=repmat(repelem(ones(T,1,'gpuArray'),N_a,1),N_z,1);

%% How to update the agent dist?
% Iteration i and time period t, call agent dist Phi_t^i
% There are two ways we could update this:
% Hanbaek Lee does the update by iterating on t
%  - First Phi_2^{i+1} is created from Phi_1^{i}
%  - Iterate on t=3:T, Phi_t^{i+1} is created from Phi_{t-1}^{i+1}
% Or we can take the fastOLG style approach
%  - Parallel over t=2:T, we create Phi_t^{i+1} from Phi_{t-1}^{i}
% For now I just use the fastOLG style approach
fastOLGtheAgentDist=1;
% But setting this to zero will do the time-loop version that Lee uses.

%% Create the initial guess
if recursiveeqmoptions.verbose>=1
    fprintf('Solving preliminary stationary eqm problem \n')
    tic;
end
[PricePath,VPath,AgentDistPath,AggVarsPath,GEcheck]=MatchedExpectationsPath_CreateInitialGuess(T,ss_ind_T,n_d,n_a,n_z,n_S,N_a,N_z,N_S,d_grid,a_grid,initialguessobjects,AggShockNames,AggVarNames,ReturnFn,FnsToEvaluate,GeneralEqmEqnsStruct,Parameters,DiscountFactorParamNames, GEPriceParamNames,GEeqnNames,recursiveeqmoptions,vfoptions,simoptions);
% And create a version of PricePath as the matrix
[PricePathOld,~,PricePathNames,~,PricePathSizeVec,~]=PricePathParamPath_StructToMatrix(PricePath,struct(),T);
if recursiveeqmoptions.verbose>=1
    fprintf('preliminary stationary eqm runtime was %4.8f seconds \n', toc)
    fprintf(' \n')
    fprintf(' \n')
    fprintf(' \n')
end

l_p=length(GEPriceParamNames);
PricePathNew=zeros(size(PricePathOld),'gpuArray'); PricePathNew(T,:)=PricePathOld(T,:);

%% We do the time periods tt=1:T all in parallel
% This leverages the fastOLG commands, which use a different shape
% fastOLG so everything is (a,t,z)
% Shapes:
% VPath is [N_a,T,N_z]
% AgentDistPath for fastOLG is [N_a*T*N_z,1]

%%
pathcounter=1;
TransPathConvergence=Inf; % ratio of 'Current Path Distance -to- recursiveeqmoptions.tolerance'
                          % Require convergence in both prices and general eqm conditions


GEcondnPath=zeros(T,length(GEeqnNames),'gpuArray');

a_gridvals=CreateGridvals(n_a,a_grid,1);
if simoptions.gridinterplayer==1
    PolicyProbsPath=zeros(N_a*(T-1)*N_z,N_probs,'gpuArray'); % preallocate
end

AgentDistt0index=repmat(gpuArray(1:1:N_a)',N_z,1)+repelem(N_a*T*gpuArray(0:1:N_z-1)',N_a,1);

%% Setup for the matching-expecations step
recursiveeqmoptions.matchE_nnearest=1; % How many 'nearest' agent distributions to use when constructing expectations
recursiveeqmoptions.matchE_distmeasure=2; % How to measure the distance between agent distributions

%% Solve using the matched-expecations path algorithm
if recursiveeqmoptions.verbose==1
    fprintf('Start solving the matched expectations path \n')
end

% VPath0=VPath;

% The initial agent distribution does not get updated much, so just keep a copy rather than indexing it every iteration
if fastOLGtheAgentDist==1
    AgentDist_initial1=AgentDistPath(AgentDistt0index); % fastOLG means AgentDistPath is (a,t,z)-by-1
else
    AgentDist_initial0=AgentDistPath(AgentDistt0index); % fastOLG means AgentDistPath is (a,t,z)-by-1
    AgentDist_initial0=reshape(AgentDist_initial0,[N_a*N_z,1]);
end


while TransPathConvergence>1 && pathcounter<recursiveeqmoptions.maxiter
    %% Occasionally we might want to update the AgentDist_initial
    % Currently I don't do this, but in principle the AgentDist_initial comes from the stationary eqm of the model without aggregate shocks, and this is not the same as the mean of the model with shocks

    %% Match Expectations (construct next period value fn)
    % Note: this is only taking expectations over the aggregate shocks, not over the idiosyncratic shocks

    tic;

    % Note: fastOLG, so VPath is (a,j)-by-z
    VPath=reshape(VPath,[N_a,T,N_z]);

    Distances=nan(T,T);
    DistMatches=zeros(T,N_S,recursiveeqmoptions.matchE_nnearest,2); % Use the closest few expectations [4th dimension: index 1 contains the t-index for the matches, index 2 contains the distances of the matches]
    % First, we have to find which agent dist with the same aggregate shock today are the 'most similar'
    if recursiveeqmoptions.matchE_distmeasure==1 % Kolmogorov-Smirnoff distance
        % Need to parallelize
        % Calculate distance for the upper triangular
        for tt1=1:T
            for tt2=tt1+1:T
                % Compare AgentDistPath(:,:,tt1) with AgentDistPath(:,:,tt2)
                % dist=  % Note: fastOLG, so AgentDistPath is (a,j,z)-by-1
                error('Not yet implemented')
            end
            Distances(tt1,tt2)=dist;
        end
        % Fill in the lower triangular (distance is symmetric)
        Distances=triu(Distances)+triu(Distances,1)';
        Distances(logical(eye(T,T)))=Inf; % Set diagonals to Inf (as we never want them to be used as matches)
    elseif recursiveeqmoptions.matchE_distmeasure==2 % Distance of first moments [Not needed because the mask eliminates this possibility anyway]
        Distances=abs(AggVarsPath.K.Mean'-AggVarsPath.K.Mean);
    elseif recursiveeqmoptions.matchE_distmeasure==3 % Distance of first moments
        Distances=abs(AggVarsPath.K.Mean'-AggVarsPath.K.Mean)+abs(AggVarsPath.L.Mean'-AggVarsPath.L.Mean);
    end
    Distances=gpuArray(Distances);
    % Put indicators on the closest ones and store them in
    MatchedEV_full=zeros(N_a,T,N_z,N_S,recursiveeqmoptions.matchE_nnearest,'gpuArray'); % note: N_a,T,N_z due to fastOLG
    if recursiveeqmoptions.matchingsetup==1 % Match Sprime based on Distances
        for tt=1:T-1
            currSSprime=ss_ind_T(tt+1);
            % Need to look for periods that are the closest today in terms of having the same S and smallest Distances (same AgentDist), but having a different next period S
            for SSprime_c=1:N_S % Loop over the possible next-period aggregate shocks
                if SSprime_c==currSSprime && tt>recursiveeqmoptions.burnin
                    % The match is this period and distance is zero
                    DistMatches(tt,SSprime_c,:,1)=tt;
                    DistMatches(tt,SSprime_c,:,2)=0;
                    % Put the next period value function into the MatchedEV
                    MatchedEV_full(:,tt,:,SSprime_c,:)=VPath(:,tt+1,:);
                else
                    % SSmask_T(SS_c,:) is binary, and takes value of 1 for the time periods that have SS_c as the aggregate shock
                    % notSSmask_Tplus1(SS_c,:) is binary, and takes value of 1 for the next-period time periods that do NOT have SS_c as the aggregate shock
                    % potential=Distances(tt,:);
                    % potential(notSSmask_Tplus1(SSprime_c,:))=Inf; % is making distance=Inf for anything that is not SSprime_c tomorrow
                    potential=Distances(tt,SSprimemask_T(1,:,SSprime_c)); % Only consider distances that have (currSS,SSprime_c), that is, look for matching S, and then we find each of the Sprime (as we are looping over the Sprime)
                    [matchval,matchind]=mink(potential,recursiveeqmoptions.matchE_nnearest); % Of the time periods that match S, and have the Sprime we are considering at the moment, find the closest as measured by 'Distances'
                    % matchind is currently for the 1s in SSprimemask_T, convert to a tt index
                    matchind=SSprimemask_T_indexes(matchind,SSprime_c);
                    DistMatches(tt,SSprime_c,:,1)=matchind;
                    DistMatches(tt,SSprime_c,:,2)=matchval;
                    for nn=1:recursiveeqmoptions.matchE_nnearest
                        MatchedEV_full(:,tt,:,SSprime_c,:)=VPath(:,matchind+1,:); % matchind+1 is where we are the period after the match [we want next period value fn to create expectations]
                    end
                end
            end
        end
    elseif recursiveeqmoptions.matchingsetup==2 % Match (S,Sprime) based on Distances
        for tt=1:T-1
            currSS=ss_ind_T(tt);
            currSSprime=ss_ind_T(tt+1);
            % Need to look for periods that are the closest today in terms of having the same S and smallest Distances (same AgentDist), but having a different next period S
            for SSprime_c=1:N_S % Loop over the possible next-period aggregate shocks
                if SSprime_c==currSSprime && tt>recursiveeqmoptions.burnin
                    % The match is this period and distance is zero
                    DistMatches(tt,SSprime_c,:,1)=tt;
                    DistMatches(tt,SSprime_c,:,2)=0;
                    % Put the next period value function into the MatchedEV
                    MatchedEV_full(:,tt,:,SSprime_c,:)=VPath(:,tt+1,:);
                else
                    % SSmask_T(SS_c,:) is binary, and takes value of 1 for the time periods that have SS_c as the aggregate shock
                    % notSSmask_Tplus1(SS_c,:) is binary, and takes value of 1 for the next-period time periods that do NOT have SS_c as the aggregate shock
                    % potential=Distances(tt,:);
                    % potential(notSSmask_Tplus1(SSprime_c,:))=Inf; % is making distance=Inf for anything that is not SSprime_c tomorrow
                    potential=Distances(tt,SSprimemask_T(1,:,currSS,SSprime_c)); % Only consider distances that have (currSS,SSprime_c), that is, look for matching S, and then we find each of the Sprime (as we are looping over the Sprime)
                    [matchval,matchind]=mink(potential,recursiveeqmoptions.matchE_nnearest); % Of the time periods that match S, and have the Sprime we are considering at the moment, find the closest as measured by 'Distances'
                    % matchind is currently for the 1s in SSprimemask_T, convert to a tt index
                    matchind=SSprimemask_T_indexes(matchind,currSS,SSprime_c);
                    DistMatches(tt,SSprime_c,:,1)=matchind;
                    DistMatches(tt,SSprime_c,:,2)=matchval;
                    for nn=1:recursiveeqmoptions.matchE_nnearest
                        MatchedEV_full(:,tt,:,SSprime_c,:)=VPath(:,matchind+1,:); % matchind+1 is where we are the period after the match [we want next period value fn to create expectations]
                    end
                end
            end
        end
    end
    % For period T, we just fill MatchedEV with the MatchedEV that is the period most similar in terms of S and 'Distances' [we cannot next period S, so this will do]
    [~,matchindT]=min(Distances(T,:).*SSmask_T(ss_ind_T(T),:));
    MatchedEV_full(:,T,:,:,:)=MatchedEV_full(:,matchindT,:,:,:);
    % Now that we have the 'matches' we construct the expected next period value fn
    MatchedEV=mean(MatchedEV_full,5); % First, take the mean over the 'nearest'
    % MatchedEV is now [N_a,T,N_z,N_S]
    % Second, take expectations with respect to next period S
    MatchedEV=sum(MatchedEV.*pi_Sprime_T,4); % pi_Sprime_T is [1,T,1,N_S], note that S is determined by t, so only needs the Sprime probabilities and does not have a dimension for S
    % MatchedEV is now [N_a,T,N_z]

    matchtime=toc;

    % disp('HERE')
    % DistMatches
    % recursiveeqmoptions
    % AggVarsPath.K.Mean(1:10)



    %% Update Params to contain the current price path
    % Put the PricePathOld matrix into PricePath structure, and store a copy in Parameters
    for pp=1:length(PricePathNames)
        PricePath.(PricePathNames{pp})=PricePathOld(:,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp));
        % Then store PricePath structure in Parameters
        Parameters.(PricePathNames{pp})=PricePath.(PricePathNames{pp});
    end

    % disp('HERE2')
    % PricePath.r(1:10)
    % PricePath.w(1:10)

    %% Since we have all the 'next period value fns', we can compute all the value fns in parallel, there is no iterating in the time dimension

    tic;
    if recursiveeqmoptions.divideT==1
        [VPath, PolicyIndexesPath]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG(MatchedEV,n_d,n_a,n_z,T,d_grid, a_grid, z_gridvals_T, pi_z_T, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % VPath in fastOLG is [N_a*T,N_z]
        % PolicyIndexesPath in fastOLG is [:,N_a,T,N_z] and contains the joint-index for (d,aprime)
        % Note: using vfoptions.EVpre
    else
        VPath=reshape(VPath,[N_a,T,N_z]);
        for rr=1:recursiveeqmoptions.divideT
            t1=recursiveeqmoptions.divideTindexes(rr,1);
            t2=recursiveeqmoptions.divideTindexes(rr,2);
            T_t1t2=t2-t1+1;
            % These next lines about parameters could be much better done
            % [only need to do this for price path (agg vars cannot be here
            % and that is only other thing that changes each iteration)]
            Parameters_rr=Parameters;
            ParamNames_rr=fieldnames(Parameters_rr);
            for pp=1:length(ParamNames_rr)
                if size(Parameters_rr.(ParamNames_rr{pp}),1)==T
                    temp=Parameters_rr.(ParamNames_rr{pp});
                    Parameters_rr.(ParamNames_rr{pp})=temp(t1:t2,:);
                elseif size(Parameters_rr.(ParamNames_rr{pp}),2)==T
                    temp=Parameters_rr.(ParamNames_rr{pp});
                    Parameters_rr.(ParamNames_rr{pp})=temp(:,t1:t2);
                end
            end
            % Do the value fn problem for periods t1-to-t2
            [VPath_t1t2, PolicyIndexesPath_t1t2]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG(MatchedEV(:,t1:t2,:),n_d,n_a,n_z,T_t1t2,d_grid, a_grid, z_gridvals_T(t1:t2,:,:), pi_z_T(t1:t2,:,:), ReturnFn, Parameters_rr, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            VPath(:,t1:t2,:)=reshape(VPath_t1t2,[N_a,T_t1t2,N_z]);
            PolicyIndexesPath(:,:,t1:t2,:)=PolicyIndexesPath_t1t2;
        end
        VPath=reshape(VPath,[N_a*T,N_z]);
    end
    vfitime=toc;

    % disp('HERE3')
    % size(MatchedEV)
    % size(VPath)
    % max(abs(MatchedEV(:)-VPath(:)))
    % size(PolicyIndexesPath)

    %% Modify PolicyIndexesPath into forms needed for forward iteration
    tic;
    % Create version of PolicyIndexesPath in form we want for the agent distribution iteration
    % Creates PolicyaprimezPath, and when using grid interpolation layer also PolicyProbsPath
    if isscalar(n_a)
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:T-1,:),[N_a*(T-1)*N_z,1]); % aprime index
    elseif length(n_a)==2
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:T-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:T-1,:)-1),[N_a*(T-1)*N_z,1]);
    elseif length(n_a)==3
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:T-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:T-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:T-1,:)-1),[N_a*(T-1)*N_z,1]);
    elseif length(n_a)==4
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:T-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:T-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:T-1,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,1:T-1,:)-1),[N_a*(T-1)*N_z,1]);
    end
    PolicyaprimetzPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(T-1)*N_z-1)',N_a,1);
    if simoptions.gridinterplayer==1
        PolicyaprimetzPath=reshape(PolicyaprimetzPath,[N_a*(T-1)*N_z,1]); % reinterpret this as lower grid index
        PolicyaprimetzPath=repelem(PolicyaprimetzPath,1,2); % create copy that will be the upper grid index
        PolicyaprimetzPath(:,2)=PolicyaprimetzPath(:,2)+1; % upper grid index
        PolicyProbsPath(:,2)=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:T-1,:),[N_a*(T-1)*N_z,1]); % L2 index
        PolicyProbsPath(:,2)=(PolicyProbsPath(:,2)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
        PolicyProbsPath(:,1)=1-PolicyProbsPath(:,2); % probability of lower grid point
    end
    % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
    PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyIndexesPath,n_d,n_a,n_z,T,d_grid,a_grid,vfoptions,1);
    PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_z,T]),[2,4,3,1]); %[N_a,T,N_z,l_daprime]

    modtime=toc;

    %% Update agent dist
    % Normally does not update the t=1 agent dist
    tic;

    % % Exception is that the preliminary stationary eqm is probably not quite 'typical', so get rid of it in the second path iteration
    % if 2<=pathcounter && pathcounter<=5
    %     % Replace the initial t=1 AgentDist with something arbitrary from later in the path.
    %     % Need to use something with the same S as in t=1
    %     temp=1:1:T;
    %     arbitraryt=temp(SSmask_T(ss_ind_T(1),:));
    %     arbitraryt=arbitraryt(round(length(arbitraryt)/2)); % pick one halfway along the path that has the same S
    %     arbitraryt
    %     [ss_ind_T(1),ss_ind_T(arbitraryt)] % should be the same
    %     AgentDistarbitrarytindex=repmat(gpuArray(1:1:N_a)',N_z,1)+N_a*(arbitraryt-1)+repelem(N_a*T*gpuArray(0:1:N_z-1)',N_a,1);
    %     AgentDist_initial=AgentDistPath(AgentDistarbitrarytindex);
    %     AgentDistPath(AgentDistt0index)=AgentDist_initial;
    %     if fastOLGtheAgentDist==0
    %         AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,1]); % fastOLG means (a,t,z)-by-1
    %     end
    % end
    % THIS DIDN'T WORK AS THEN GOT A MIX OF THIS AGENT DIST WITH THE
    % EXISTING PERIOD 1 POLICY, ETC. AND THAT JUST TURNED INTO A MESS

    if fastOLGtheAgentDist==1
        if pathcounter<=5
            distiter=5;
        else
            distiter=10;
        end
        for distii=1:distiter
            if simoptions.gridinterplayer==0
                AgentDistPath=AgentDist_FHorz_TPath_SingleStep_IterFast_raw(AgentDistPath,PolicyaprimetzPath,N_a,N_z,T,pi_z_T_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,AgentDist_initial1); % Policy for jj=1:N_j-1
            else
                AgentDistPath=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_raw(AgentDistPath,PolicyaprimetzPath,PolicyProbsPath,N_a,N_z,T,pi_z_T_sim,II,exceptlastj,exceptfirstj,justfirstj,AgentDist_initial1); % Policy for jj=1:N_j-1
            end
        end
    else
        % The loop approach appears much less stable (you get one bad iteration of the matched-expectations path and it blows up)
        AgentDistPath=zeros([N_a*N_z,T],'gpuArray');
        AgentDist=AgentDist_initial0;
        AgentDistPath(:,1)=AgentDist;
        II1b=gpuArray(1:1:N_a*N_z);
        IIones=ones(N_a*N_z,1,'gpuArray');
        if simoptions.gridinterplayer==0
            PolicyaprimePath=reshape(PolicyIndexesPath(1,:,1:T-1,:),[N_a,(T-1),N_z]);
            for tt=1:T-1
                Policy_aprimez=reshape(PolicyaprimePath(:,tt,:),[N_a*N_z,1])+N_a*repelem((0:1:N_z-1)',N_a,1);
                % AgentDist=sparse(reshape(AgentDistPathPre(:,tt,:),[N_a*N_z,1])); % CHECK THAT THEY GIVE THE SAME THING
                AgentDist=AgentDist_InfHorz_TPath_SingleStep_raw(AgentDist,Policy_aprimez,II1b,IIones,N_a,N_z,sparse(squeeze(pi_z_T(tt,:,:))')); % pi_z_T(tt,:,:)' as pi_z_T has shape designed for fastOLG
                AgentDistPath(:,tt+1)=AgentDist;
            end
            AgentDistPath=reshape(permute(reshape(AgentDistPath,[N_a,N_z,T]),[1,3,2]),[N_a*T*N_z,1]); % fastOLG shape for use in AggVars
        else
            error('Not yet implemented')
        end
    end

    agentdisttime=toc;

    % disp('HERE4')
    % size(AgentDist_initial1)
    % size(AgentDistPath)
    % temp=abs(repmat(reshape(AgentDist_initial1,[N_a,1,N_z]),1,T,1)-reshape(AgentDistPath,[N_a,T,N_z]));
    % temp2=max(max(temp,3),1);
    % temp2(1:10)

    %% AggVars
    tic;
    AggVarsPath=EvalFnOnAgentDist_AgeConditionalAggVars_FHorz_fastOLG(AgentDistPath.*AgeWeights_T,PolicyValuesPath(:,:,:,1:l_d),PolicyValuesPath(:,:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,T,l_d,l_a,l_a,l_z,N_a,N_z,a_gridvals,z_gridvals_T_fastOLG,1);
    for ff=1:length(AggVarNames)
        Parameters.(AggVarNames{ff})=AggVarsPath.(AggVarNames{ff}).Mean;
    end

    aggvartime=toc;

    tic;
    %% General Eqm Eqns
    % Evaluate the general eqm conditions, and based on them create PricePathNew (interpretation depends on transpathoptions)
    % I can parallel this, but for now just loop over tt
    for tt=1:T
        for pp=1:length(PricePathNames)
            temp=PricePath.(PricePathNames{pp});
            Parameters.(PricePathNames{pp})=temp(tt);
        end
        for SS_c=1:length(n_S)
            temp=AggShocksPath.(AggShockNames{SS_c});
            Parameters.(AggShockNames{SS_c})=temp(tt);
        end
        for ff=1:length(AggVarNames)
            temp=AggVarsPath.(AggVarNames{ff}).Mean;
            Parameters.(AggVarNames{ff})=temp(tt);
        end
        [PricePathNew_tt,GEcondnPath_tt]=updatePricePathNew_TPath_tt(Parameters,GeneralEqmEqnsCell,GeneralEqmEqnParamNames,PricePathOld(tt,:),recursiveeqmoptions);
        PricePathNew(tt,:)=PricePathNew_tt;
        GEcondnPath(tt,:)=GEcondnPath_tt;
    end
    % Put the aggregate price and shocks paths back into Parameters
    for pp=1:length(PricePathNames)
        Parameters.(PricePathNames{pp})=PricePath.(PricePathNames{pp});
    end
    for SS_c=1:length(n_S)
        Parameters.(AggShockNames{SS_c})=AggShocksPath.(AggShockNames{SS_c}); % Can just set it up as an exogenous parameter that has T as a dimension
    end
    for ff=1:length(AggVarNames)
        Parameters.(AggVarNames{ff})=AggVarsPath.(AggVarNames{ff}).Mean;
    end

    GEtime=toc;


    % disp('HERE5: K and L')
    % AggVarsPath.K.Mean(1:10)
    % AggVarsPath.L.Mean(1:10)
    % [min(AggVarsPath.K.Mean),max(AggVarsPath.K.Mean)]
    % [min(AggVarsPath.L.Mean),max(AggVarsPath.L.Mean)]
    % [~,iiK]=min(AggVarsPath.K.Mean);
    % [~,iiL]=min(AggVarsPath.L.Mean);
    %
    % disp('Min K and L periods')
    % [iiK,iiL]
    % [AggVarsPath.K.Mean(iiK),AggVarsPath.L.Mean(iiK),PricePath.r(iiK),PricePath.w(iiK)]
    % [AggVarsPath.K.Mean(iiL),AggVarsPath.L.Mean(iiL),PricePath.r(iiL),PricePath.w(iiL)]
    % GEcondnPath(iiK,:)
    % GEcondnPath(iiL,:)
    %
    % if pathcounter>1
    %     [AggVarsPath.K.Mean(iiKlag),AggVarsPath.L.Mean(iiKlag),PricePath.r(iiKlag),PricePath.w(iiKlag)]
    %     [AggVarsPath.K.Mean(iiLlag),AggVarsPath.L.Mean(iiLlag),PricePath.r(iiLlag),PricePath.w(iiLlag)]
    %     GEcondnPath(iiKlag,:)
    %     GEcondnPath(iiLlag,:)
    % end
    %
    % iiKlag=iiK;
    % iiLlag=iiL;

    fprintf('Runtimes are: match, vfi, mod, agentdist, aggvar, GE \n')
    [matchtime,vfitime,modtime,agentdisttime,aggvartime,GEtime]
    fprintf('Solving agent dist using fastOLGtheAgentDist=%i (0=loop, 1=parallel) \n', fastOLGtheAgentDist)

    %% Perform update of prices and give feedback

    % See how far apart the price paths are
    CurrentPathDist_price=max(abs(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:)),[],1); % 1-by-prices
    % Notice that the distance is always calculated ignoring the time t=1 & t=T periods, as these needn't ever converges
    % Why look at price paths, why just look at the general eqm conditions??

    % CurrentPathDist=max(abs(GEcondnPath));
    CurrentPathDist_GEcondn=max(abs(GEcondnPath),[],1); % 1-by-GECondns

    % Create plots of the transition path (before we update pricepath)
    createTPathFeedbackPlots(PricePathNames,AggVarNames,GEeqnNames,PricePathOld,AggVarsPath,GEcondnPath,recursiveeqmoptions);

    % Update PricePathOld
    % Set price path to be 9/10ths the old path and 1/10th the new path (but making sure to leave prices in periods 1 & T unchanged).
    if recursiveeqmoptions.weightscheme==0
        PricePathOld=PricePathNew; % The update weights are already in GEnewprice setup
    elseif recursiveeqmoptions.weightscheme==1 % Just a constant weighting
        PricePathOld(1:T-1,:)=recursiveeqmoptions.oldpathweight*PricePathOld(1:T-1)+(1-recursiveeqmoptions.oldpathweight)*PricePathNew(1:T-1,:);
    elseif recursiveeqmoptions.weightscheme==2 % A exponentially decreasing weighting on new path from (1-oldpathweight) in first period, down to 0.1*(1-oldpathweight) in T-1 period.
        % I should precalculate these weighting vectors
        Ttheta=recursiveeqmoptions.Ttheta;
        PricePathOld(1:Ttheta,:)=recursiveeqmoptions.oldpathweight*PricePathOld(1:Ttheta)+(1-recursiveeqmoptions.oldpathweight)*PricePathNew(1:Ttheta,:);
        PricePathOld(Ttheta:T-1,:)=((recursiveeqmoptions.oldpathweight+(1-exp(linspace(0,log(0.2),T-Ttheta)))*(1-recursiveeqmoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(Ttheta:T-1,:)+((exp(linspace(0,log(0.2),T-Ttheta)).*(1-recursiveeqmoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(Ttheta:T-1,:);
    elseif recursiveeqmoptions.weightscheme==3 % A gradually opening window.
        if (pathcounter*3)<T-1
            PricePathOld(1:(pathcounter*3),:)=recursiveeqmoptions.oldpathweight*PricePathOld(1:(pathcounter*3),:)+(1-recursiveeqmoptions.oldpathweight)*PricePathNew(1:(pathcounter*3),:);
        else
            PricePathOld(1:T-1,:)=recursiveeqmoptions.oldpathweight*PricePathOld(1:T-1,:)+(1-recursiveeqmoptions.oldpathweight)*PricePathNew(1:T-1,:);
        end
    elseif recursiveeqmoptions.weightscheme==4 % Combines weightscheme 2 & 3
        if (pathcounter*3)<T-1
            PricePathOld(1:(pathcounter*3),:)=((recursiveeqmoptions.oldpathweight+(1-exp(linspace(0,log(0.2),pathcounter*3)))*(1-recursiveeqmoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(1:(pathcounter*3),:)+((exp(linspace(0,log(0.2),pathcounter*3)).*(1-recursiveeqmoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(1:(pathcounter*3),:);
        else
            PricePathOld(1:T-1,:)=((recursiveeqmoptions.oldpathweight+(1-exp(linspace(0,log(0.2),T-1)))*(1-recursiveeqmoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(1:T-1,:)+((exp(linspace(0,log(0.2),T-1)).*(1-recursiveeqmoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(1:T-1,:);
        end
    end

    TransPathConvergence_prices=max(CurrentPathDist_price)/recursiveeqmoptions.tolerance; % So when this gets to 1 we have convergence, in prices
    TransPathConvergence_GEcondns=max(GEcondnPath(:).^2)/recursiveeqmoptions.tolerance; % So when this gets to 1 we have convergence, in GE condns
    TransPathConvergence=max(TransPathConvergence_prices,TransPathConvergence_GEcondns); % we require convergence in both

    if recursiveeqmoptions.verbose>=1
        fprintf(' \n')
        fprintf('Number of iterations on matched-expecations path: %i \n',pathcounter)
        fprintf('Current distance between old and new price path (in L-Infinity norm): %8.6f \n', max(CurrentPathDist_price))
        fprintf(priceverbosestr, CurrentPathDist_price')
        fprintf('Current General Eqm conditions (in L-Infinity norm): %8.6f \n', max(CurrentPathDist_GEcondn))
        fprintf(GEcondnverbosestr, CurrentPathDist_GEcondn')
        fprintf('Ratio of current distance to the convergence tolerance, in prices: %.2f (convergence when reaches 1) \n',TransPathConvergence_prices)
        fprintf('Ratio of current distance to the convergence tolerance, in GE Condns: %.2f (convergence when reaches 1) \n',TransPathConvergence_GEcondns)
        fprintf('Ratio of current distance to the convergence tolerance: %.2f (convergence when reaches 1; is the minimum of both prices and GEcondns) \n',TransPathConvergence)
        fprintf(' \n')
    end

    if recursiveeqmoptions.historyofpricepath==1
        % Store the whole history of the price path and save it every ten iterations
        PricePathHistory{pathcounter,1}=CurrentPathDist_price;
        PricePathHistory{pathcounter,2}=PricePathOld;
        if rem(pathcounter,10)==1
            save ./SavedOutput/RecursiveGEwAggShocks_Internal.mat PricePathHistory
        end
    end

    pathcounter=pathcounter+1;
end

%% Sort some stuff for output
% Put the PricePathOld matrix into PricePath structure for output
GEcondnPathMatrix=GEcondnPath;
clear GEcondnPath
for gg=1:length(GEeqnNames)
    GEcondnPath.(GEeqnNames{gg})=GEcondnPathMatrix(:,gg);
end
% Put the PricePathOld matrix into PricePath structure for output
for pp=1:length(PricePathNames)
    PricePath.(PricePathNames{pp})=PricePathOld(:,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp));
end

if pathcounter>=recursiveeqmoptions.maxiter
    warning('Stopped due to reaching maxiter (rather than convergence; while computing Recursive General Eqm using Matched Expectations Path algorithm)')
end

end