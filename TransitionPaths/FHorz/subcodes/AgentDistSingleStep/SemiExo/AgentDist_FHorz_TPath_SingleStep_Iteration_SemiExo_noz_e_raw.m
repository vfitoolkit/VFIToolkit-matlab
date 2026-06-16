function AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_SemiExo_noz_e_raw(AgentDist,Policy_dsemiexo,Policy_aprime,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J,jequalOneDist)
% One time-step of the agent distribution iteration, with semi-exogenous state and iid e (no z).
% age weights are handled elsewhere, here all are normalized to one
% AgentDist is [N_a*N_semiz*N_e,N_j] (the time-t panel); output is the time-(t+1) panel
% Policy_dsemiexo and Policy_aprime are [N_a*N_semiz*N_e,N_j]
% semiz->semiz' (depends on d2) is folded into Gammatranspose (which also sums out e); e' is then added via kron(pi_e,.)

%% Sparsity trick on pi_semiz_J
N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
[pi_semiz_J_short, idx] = sort(pi_semiz_J,2);
pi_semiz_J_short=pi_semiz_J_short(:,end-N_semizshort+1:end,:,:);
idxshort=idx(:,end-N_semizshort+1:end,:,:);

Policy_dsemiexo=gather(Policy_dsemiexo);
Policy_aprime=gather(Policy_aprime);
pi_semiz_J_short=gather(pi_semiz_J_short);
idxshort=gather(idxshort);
semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_e,1)+N_semiz*(0:1:N_semizshort-1); % age-independent part of semizindex_short
% semizindex_short_jj (built per age below) is [N_a*N_semiz*N_e,N_semizshort]

AgentDist=gather(AgentDist);

II2=repelem((1:1:N_a*N_semiz*N_e)',1,N_semizshort);

for jjr=1:(N_j-1)
    jj=N_j-jjr; % It is important that this is in reverse order (due to just overwriting AgentDist)
    AgentDist_jj=sparse(AgentDist(:,jj));

    semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1);
    Policy_aprimesemiz_jj=repelem(Policy_aprime(:,jj),1,N_semizshort)+N_a*(idxshort(semizindex_short_jj)-1); % Note: add semiz' index following the semiz' dimension
    semiztransitions_jj=pi_semiz_J_short(semizindex_short_jj);

    Gammatranspose=sparse(Policy_aprimesemiz_jj,II2,semiztransitions_jj,N_a*N_semiz,N_a*N_semiz*N_e); % From (a,semiz,e) to (a',semiz')

    % No z, so just one step (e is summed out by Gammatranspose)
    AgentDist_jj=Gammatranspose*AgentDist_jj;

    % Add e back into the distribution
    pi_e=sparse(gather(pi_e_J(:,jj)));
    AgentDist_jj=kron(pi_e,AgentDist_jj);

    AgentDist(:,jj+1)=full(AgentDist_jj);
end

% Move the solution to the gpu
AgentDist=gpuArray(AgentDist);
AgentDist(:,1)=jequalOneDist; % age j=1 dist

end
