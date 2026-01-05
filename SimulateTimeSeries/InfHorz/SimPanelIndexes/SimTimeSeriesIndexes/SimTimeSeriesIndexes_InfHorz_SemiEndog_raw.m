function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_SemiEndog_raw(PolicyIndexesKron,N_d,N_a,N_z,cumsum_pi_z_semiendog,burnin,seedpoint,simperiods,parallel)
%
% burnin=simoptions.burnin;
% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;
%
%Simulates path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped). The burn in begins at point 'seedpoint' (this is not just left
%as being random since some random points may be ones that never 'exist' in
%eqm)

MoveSTSKtoGPU=0;
if parallel==2
    % Simulation on GPU is really slow.
    % So instead, switch to CPU, and then switch
    % back. For anything but ridiculously short simulations it is more than
    % worth the overhead.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    cumsum_pi_z_semiendog=gather(cumsum_pi_z_semiendog);
    MoveSTSKtoGPU=1;
end

% cumsum_pi_z_semiendog % cumulative some over zprime; has dimensions z-zprime-k

SimTimeSeriesKron=zeros(2,simperiods);

currstate=seedpoint;
% cumsum_pi_z=cumsum(pi_z,2);

if N_d==0
%     optaprime=1;
    if burnin>0
        for t=1:burnin
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z_semiendog(currstate(2),:,currstate(1))>rand(1,1));
        end
    end
    for t=1:simperiods
        SimTimeSeriesKron(1,t)=currstate(1); %a_c
        SimTimeSeriesKron(2,t)=currstate(2); %z_c
        
        currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
        [~,currstate(2)]=max(cumsum_pi_z_semiendog(currstate(2),:,currstate(1))>rand(1,1));
    end
else
%     optaprime=2;
    if burnin>0
        for t=1:burnin
            currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z_semiendog(currstate(2),:,currstate(1))>rand(1,1));
        end
    end
    for t=1:simperiods
        SimTimeSeriesKron(1,t)=currstate(1); %a_c
        SimTimeSeriesKron(2,t)=currstate(2); %z_c
        
        currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2));
        [~,currstate(2)]=max(cumsum_pi_z_semiendog(currstate(2),:,currstate(1))>rand(1,1));
    end
end

if MoveSTSKtoGPU==1
    SimTimeSeriesKron=gpuArray(SimTimeSeriesKron);
end

end
