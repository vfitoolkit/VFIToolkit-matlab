function SimPanel=SimPanelIndexes_TransPath_Case1_lowmem(PolicyPathKron, T, AgentDist_initial, n_d, n_a, n_z, pi_z, simoptions)
% Intended for internal use, not by user.
%
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn AgentDist_initial. (If you use the
% newbirths option you will get more than 'numbersims', due to the extra births)
%
% AgentDist_initial can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an AgentDist_initial
% for time j=1. (So AgentDist_initial is n_a-by-n_z)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which simoptions have been declared, set all others to defaults 
% This command is for internal use only, so simoptions should already be appropraitely set up.

l_d=length(n_d); % l_d=0 is handled by a different code
l_a=length(n_a);
l_z=length(n_z);

%%
% Transition paths do not currently allow for the exogenous shock process to differ based on time period.
% fieldexists_ExogShockFn=0;

% Get seedpoints from InitialDist while on gpu
seedpoints=nan(simoptions.numbersims,2,'gpuArray'); % 2 as a,z (vectorized)
cumsumInitialDistVec=cumsum(reshape(AgentDist_initial,[N_a*N_z,1]));
[~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1,'gpuArray'));
for ii=1:simoptions.numbersims
    seedpoints(ii,:)=ind2sub_homemade_gpu([N_a,N_z],seedpointvec(ii));
end

seedpoints=floor(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

cumsumpi_z_T=gather(cumsum(pi_z,2)).*ones(1,1,T);
PolicyPathKron=gather(PolicyPathKron);
seedpoints=gather(seedpoints);
simoptions.simperiods=gather(simoptions.simperiods);
% Simulation on GPU is really slow. So instead, switch to CPU, and then switch
% back. For anything but ridiculously short simulations it is more than worth the overhead.
MoveOutputtoGPU=0;
if simoptions.parallel==2
    MoveOutputtoGPU=1;
end

SimPanel=nan(l_a+l_z,simoptions.simperiods,simoptions.numbersims); % (a,z)
if simoptions.parallel==0
    for ii=1:simoptions.numbersims
        seedpoint=seedpoints(ii,:);
        % Since a finite-horizon value fn problem and a transition path are much the same thing we can just piggy back on the codes for FHorz.
        SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyPathKron,N_d,N_a,N_z,T,cumsumpi_z_T, [seedpoint,1], simoptions.simperiods);
                
        SimPanel(l_daprime+1:l_daprime+l_a,:,ii)=ind2sub_vec_homemadet(n_a, SimLifeCycleKron(1,:)); % a
        SimPanel(l_daprime+l_a+1:l_daprime+l_a+l_z,:,ii)=ind2sub_vec_homemadet(n_z, SimLifeCycleKron(2,:)); % z
        % SimLifeCycleKron(3,:) is t, but we don't care

        % Following lines should be outside the loop, but I'm feeling lazy
        dind=PolicyPathKron(1,SimLifeCycleKron(1,:),SimLifeCycleKron(2,:),SimLifeCycleKron(3,:));
        SimPanel(1:l_d,:,ii)=ind2sub_vec_homemadet(n_d,dind);
        aprimeind=PolicyPathKron(2,SimLifeCycleKron(1,:),SimLifeCycleKron(2,:),SimLifeCycleKron(3,:));
        SimPanel(l_d+1:l_daprime,:,ii)=ind2sub_vec_homemadet(n_a,aprimeind);
    end
else
    simperiodstemp=simoptions.simperiods; % just to avoid overhead in parfor
    parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
        seedpoint=seedpoints(ii,:);
        SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyPathKron,N_d,N_a,N_z,T,cumsumpi_z_T, [seedpoint,1], simperiodstemp);
        
        SimPanel_ii=zeros(l_a+l_z,simperiodstemp);
        SimPanel_ii(1:l_a,:)=ind2sub_vec_homemadet(n_a, SimLifeCycleKron(1,:)); % a
        SimPanel_ii(l_a+1:l_a+l_z,:)=ind2sub_vec_homemadet(n_z, SimLifeCycleKron(2,:)); % z
        % SimLifeCycleKron(3,:) is t, but we don't care

        % Following lines should be outside the loop, but I'm feeling lazy
        dind=PolicyPathKron(1+2*(SimLifeCycleKron(1,:)-1)+2*N_a*(SimLifeCycleKron(2,:)-1)+2*N_a*N_z*(SimLifeCycleKron(3,:)-1));
        SimPanel_ii(1:l_d,:)=ind2sub_vec_homemadet(n_d,dind);
        aprimeind=PolicyPathKron(2+2*(SimLifeCycleKron(1,:)-1)+2*N_a*(SimLifeCycleKron(2,:)-1)+2*N_a*N_z*(SimLifeCycleKron(3,:)-1));
        SimPanel_ii(l_d+1:l_daprime,:)=ind2sub_vec_homemadet(n_a,aprimeind);

        SimPanel(:,:,ii)=SimPanel_ii;
    end
end

if MoveOutputtoGPU==1
    SimPanel=gpuArray(SimPanel);
end

end



