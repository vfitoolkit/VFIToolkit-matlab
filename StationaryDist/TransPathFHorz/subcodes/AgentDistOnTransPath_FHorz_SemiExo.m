function AgentDistPath=AgentDistOnTransPath_FHorz_SemiExo(AgentDist_initial, jequaloneDist, PolicyPath, AgeWeights, AgeWeights_T, n_d, n_a, n_z, n_e, N_j, pi_z_J, pi_z_J_sim, pi_e_J, pi_e_J_sim, T, Parameters, transpathoptions, simoptions)
% Agent distribution along the transition path, with a semi-exogenous state.
% The semi-exogenous transition (semiz->semiz') depends on the decision d2, so it is folded into the
% single-step iteration (cannot be a fixed markov step). semiz and z are treated as the composite
% bothze=(semiz,z,e) with semiz indexing fastest.
% Note: called from AgentDistOnTransPath_Case1_FHorz before its (non-semiz) reshapes, so this command
% does its own reshaping of AgentDist_initial/jequaloneDist/PolicyPath.

n_semiz=simoptions.n_semiz;
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_e=prod(n_e);

if ~isscalar(n_a)
    error('Transition paths with semi-exogenous states only allow a single endogenous state (cannot have length(n_a)>1)')
end
if N_z>0 && transpathoptions.zpathtrivial==0
    error('Semi-exogenous states with z varying over the transition path are not yet implemented for AgentDistOnTransPath (email me if you want this)')
end
if N_e>0 && transpathoptions.epathtrivial==0
    error('Semi-exogenous states with e varying over the transition path are not yet implemented for AgentDistOnTransPath (email me if you want this)')
end

%% bothze composite
if N_z==0
    if N_e==0
        n_bothze=n_semiz; N_bothze=N_semiz;
    else
        n_bothze=[n_semiz,n_e]; N_bothze=N_semiz*N_e;
    end
else
    if N_e==0
        n_bothze=[n_semiz,n_z]; N_bothze=N_semiz*N_z;
    else
        n_bothze=[n_semiz,n_z,n_e]; N_bothze=N_semiz*N_z*N_e;
    end
end

%% Decision variables: split into d1 (standard) and d2 (those determining the semi-exogenous transition)
if ~isfield(simoptions,'l_dsemiz')
    simoptions.l_dsemiz=1;
end
l_d=length(n_d);
l_d1=l_d-simoptions.l_dsemiz;
N_dsemiz=prod(n_d(l_d1+1:end));

%% Build pi_semiz_J (transition probabilities for the semi-exogenous state; depend on d2)
if ~isfield(simoptions,'d_grid')
    error('To use a semi-exogenous state in AgentDistOnTransPath you must provide simoptions.d_grid')
end
simoptions=SemiExogShockSetup_FHorz(n_d,N_j,simoptions.d_grid,Parameters,simoptions,2);
pi_semiz_J=simoptions.pi_semiz_J; % [N_semiz,N_semiz',N_dsemiz,N_j]


%% pi_e_J [N_e,N_j] is used by the slowOLG raws; pi_e_J_sim (with N_asemiz) by the fastOLG raws.
% For fastOLG the passed-in pi_e_J is the value-fn form (not [N_e,N_j]), so build pi_e_J_sim from the generic
% pi_e_J_sim (the agent-dist form [N_a*(N_j-1)(*N_z),N_e]) by repelem (valid: pi_e is constant within each age block).
if N_e>0
    if simoptions.fastOLG==1
        pi_e_J_sim=gpuArray(repelem(gather(pi_e_J_sim),N_semiz,1)); % [N_a*(N_j-1)(*N_z),N_e] -> [N_asemiz*(N_j-1)(*N_z),N_e]
    else
        pi_e_J=gather(pi_e_J); % [N_e,N_j]
    end
end

%% N_probs and grid interpolation layer
N_probs=1;
if simoptions.gridinterplayer==1
    N_probs=2;
end

%% Reshape PolicyPath to [npol,N_a,N_bothze,N_j,T] and extract Policy_aprime and Policy_dsemiexo (and PolicyProbs for GI)
PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_bothze,N_j,T]);

% Policy_aprime (single endogenous state, so just the aprime index in 1..N_a)
Policy_aprime=reshape(PolicyPath(l_d+1,:,:,:,:),[N_a*N_bothze,N_j,T]);
% Policy_dsemiexo: the d2 index (l_dsemiz decision variables, immediately after the l_d1 standard ones)
if simoptions.l_dsemiz==1
    Policy_dsemiexo=reshape(PolicyPath(l_d1+1,:,:,:,:),[N_a*N_bothze,N_j,T]);
elseif simoptions.l_dsemiz==2
    Policy_dsemiexo=reshape(PolicyPath(l_d1+1,:,:,:,:)+n_d(l_d1+1)*(PolicyPath(l_d1+2,:,:,:,:)-1),[N_a*N_bothze,N_j,T]);
else
    error('simoptions.l_dsemiz>2 not yet implemented for AgentDistOnTransPath with semiz')
end

if simoptions.gridinterplayer==1
    % Build lower/upper aprime points and the corresponding probabilities (from the L2 index)
    aprimeProbs_upper=reshape((PolicyPath(end-1,:,:,:,:)-1)/(simoptions.ngridinterp+1),[N_a*N_bothze,1,N_j,T]); % prob of upper grid point (end is L2flag, end-1 is L2 index)
    Policy_aprime=reshape(Policy_aprime,[N_a*N_bothze,1,N_j,T]);
    Policy_aprime=repmat(Policy_aprime,1,2,1,1);
    Policy_aprime(:,2,:,:)=Policy_aprime(:,2,:,:)+1; % upper grid point
    PolicyProbs=ones([N_a*N_bothze,2,N_j,T],'gpuArray');
    PolicyProbs(:,1,:,:)=1-aprimeProbs_upper; % lower
    PolicyProbs(:,2,:,:)=aprimeProbs_upper; % upper
end

%% Reshape AgentDist_initial to [N_a*N_bothze,N_j], get age weights, check
AgentDist_initial=gpuArray(reshape(AgentDist_initial,[N_a*N_bothze,N_j]));
AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
if transpathoptions.ageweightstrivial==1
    if max(abs(AgeWeights_initial-AgeWeights'))>10^(-9)
        error('AgeWeights differs from the weights implicit in the initial agent distribution')
    end
end

%% jequaloneDist (only the constant case is supported)
jequaloneDist=gpuArray(reshape(jequaloneDist,[N_a*N_bothze,1]));

%% Iterate the agent distribution along the path
if simoptions.fastOLG==0
    %% slowOLG: AgentDist panel is [N_a*N_bothze,N_j]
    AgentDistPath=zeros(N_a*N_bothze,N_j,T,'gpuArray');
    AgentDist=AgentDist_initial./AgeWeights_initial; % remove age weights
    AgentDistPath(:,:,1)=AgentDist;
    for tt=1:T-1
        Policy_dsemiexo_tt=Policy_dsemiexo(:,:,tt);
        Policy_aprime_tt=Policy_aprime(:,:,tt);
        if simoptions.gridinterplayer==0
            if N_z==0 && N_e==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_SemiExo_noz_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,jequaloneDist);
            elseif N_e==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_SemiExo_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,jequaloneDist);
            elseif N_z==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_SemiExo_noz_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J,jequaloneDist);
            else
                AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_SemiExo_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,pi_e_J,jequaloneDist);
            end
        else % gridinterplayer==1
            PolicyProbs_tt=PolicyProbs(:,:,:,tt);
            Policy_aprime_tt=reshape(Policy_aprime(:,:,:,tt),[N_a*N_bothze,N_probs,N_j]);
            if N_z==0 && N_e==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_SemiExo_noz_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,jequaloneDist);
            elseif N_e==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_SemiExo_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,jequaloneDist);
            elseif N_z==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_SemiExo_noz_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J,jequaloneDist);
            else
                AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_SemiExo_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,pi_e_J,jequaloneDist);
            end
        end
        AgentDistPath(:,:,tt+1)=AgentDist;
    end
    AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights
    AgentDistPath=reshape(AgentDistPath,[n_a,n_bothze,N_j,T]);

else
    %% fastOLG: convert AgentDist_initial and jequaloneDist to the fast layout, iterate, reapply age weights
    AgentDist_noweights=AgentDist_initial./AgeWeights_initial; % [N_a*N_bothze,N_j], remove age weights
    % reshape to [N_a,N_semiz,(N_z),(N_e),N_j] then permute so that the order is (a,semiz,j,(z)) with e in trailing columns
    if N_z==0 && N_e==0
        AgentDist=reshape(permute(reshape(AgentDist_noweights,[N_a,N_semiz,N_j]),[1,2,3]),[N_a*N_semiz*N_j,1]);
        jequalOneDist=reshape(jequaloneDist,[N_a*N_semiz,1]);
        AgentDistPath=zeros(N_a*N_semiz*N_j,T,'gpuArray');
    elseif N_e==0
        AgentDist=reshape(permute(reshape(AgentDist_noweights,[N_a,N_semiz,N_z,N_j]),[1,2,4,3]),[N_a*N_semiz*N_j*N_z,1]);
        jequalOneDist=reshape(jequaloneDist,[N_a*N_semiz*N_z,1]);
        AgentDistPath=zeros(N_a*N_semiz*N_j*N_z,T,'gpuArray');
    elseif N_z==0
        AgentDist=reshape(permute(reshape(AgentDist_noweights,[N_a,N_semiz,N_e,N_j]),[1,2,4,3]),[N_a*N_semiz*N_j,N_e]);
        jequalOneDist=reshape(jequaloneDist,[N_a*N_semiz*N_e,1]);
        AgentDistPath=zeros(N_a*N_semiz*N_j,N_e,T,'gpuArray');
    else
        AgentDist=reshape(permute(reshape(AgentDist_noweights,[N_a,N_semiz,N_z,N_e,N_j]),[1,2,5,3,4]),[N_a*N_semiz*N_j*N_z,N_e]);
        jequalOneDist=reshape(jequaloneDist,[N_a*N_semiz*N_z*N_e,1]);
        AgentDistPath=zeros(N_a*N_semiz*N_j*N_z,N_e,T,'gpuArray');
    end
    if N_z==0 && N_e==0
        AgentDistPath(:,1)=AgentDist;
    elseif N_e==0
        AgentDistPath(:,1)=AgentDist;
    else
        AgentDistPath(:,:,1)=AgentDist;
    end

    for tt=1:T-1
        Policy_dsemiexo_tt=Policy_dsemiexo(:,:,tt);
        if simoptions.gridinterplayer==0
            Policy_aprime_tt=Policy_aprime(:,:,tt);
            if N_z==0 && N_e==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_SemiExo_noz_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,jequalOneDist);
                AgentDistPath(:,tt+1)=AgentDist;
            elseif N_e==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_SemiExo_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J_sim,jequalOneDist);
                AgentDistPath(:,tt+1)=AgentDist;
            elseif N_z==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_SemiExo_noz_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J_sim,jequalOneDist);
                AgentDistPath(:,:,tt+1)=AgentDist;
            else
                AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_SemiExo_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J_sim,pi_e_J_sim,jequalOneDist);
                AgentDistPath(:,:,tt+1)=AgentDist;
            end
        else % gridinterplayer==1
            PolicyProbs_tt=reshape(PolicyProbs(:,:,:,tt),[N_a*N_bothze,N_probs,N_j]);
            Policy_aprime_tt=reshape(Policy_aprime(:,:,:,tt),[N_a*N_bothze,N_probs,N_j]);
            if N_z==0 && N_e==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_SemiExo_noz_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,jequalOneDist);
                AgentDistPath(:,tt+1)=AgentDist;
            elseif N_e==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_SemiExo_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J_sim,jequalOneDist);
                AgentDistPath(:,tt+1)=AgentDist;
            elseif N_z==0
                AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_SemiExo_noz_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J_sim,jequalOneDist);
                AgentDistPath(:,:,tt+1)=AgentDist;
            else
                AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_SemiExo_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J_sim,pi_e_J_sim,jequalOneDist);
                AgentDistPath(:,:,tt+1)=AgentDist;
            end
        end
    end

    % Put the age weights back in, and reshape to output [n_a,n_bothze,N_j,T]
    if N_z==0 && N_e==0
        AgentDistPath=AgentDistPath.*repelem(AgeWeights_T,N_a*N_semiz,1); % [N_a*N_semiz*N_j,T]
        AgentDistPath=reshape(AgentDistPath,[N_a*N_semiz,N_j,T]); % (a,semiz,j,T)
        AgentDistPath=reshape(AgentDistPath,[n_a,n_bothze,N_j,T]);
    elseif N_e==0
        AgentDistPath=AgentDistPath.*repmat(repelem(AgeWeights_T,N_a*N_semiz,1),N_z,1); % [N_a*N_semiz*N_j*N_z,T]
        AgentDistPath=permute(reshape(AgentDistPath,[N_a*N_semiz,N_j,N_z,T]),[1,3,2,4]); % (a,semiz,z,j,T)
        AgentDistPath=reshape(AgentDistPath,[n_a,n_bothze,N_j,T]);
    elseif N_z==0
        AgentDistPath=AgentDistPath.*repelem(reshape(AgeWeights_T,[N_j,1,T]),N_a*N_semiz,1); % [N_a*N_semiz*N_j,N_e,T]
        AgentDistPath=permute(reshape(AgentDistPath,[N_a*N_semiz,N_j,N_e,T]),[1,3,2,4]); % (a,semiz,e,j,T)
        AgentDistPath=reshape(AgentDistPath,[n_a,n_bothze,N_j,T]);
    else
        AgentDistPath=AgentDistPath.*repmat(repelem(reshape(AgeWeights_T,[N_j,1,T]),N_a*N_semiz,1),N_z,1); % [N_a*N_semiz*N_j*N_z,N_e,T]
        AgentDistPath=permute(reshape(AgentDistPath,[N_a*N_semiz,N_j,N_z,N_e,T]),[1,3,4,2,5]); % (a,semiz,z,e,j,T)
        AgentDistPath=reshape(AgentDistPath,[n_a,n_bothze,N_j,T]);
    end
end

end
