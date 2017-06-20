function SimTimeSeriesKron=SimTimeSeriesIndexes_Case2_raw(PolicyIndexesKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_z,pi_z,burnin,seedpoint,simperiods,parallel)

% burnin=simoptions.burnin;
% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;

%Simulates path based on PolicyIndexes & Phi_aprimeKron of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped). The burn in begins at point 'seedpoint' (this is not just left
%as being random since some random points may be ones that never 'exist' in
%eqm)

MoveOutputToGPU=0;
if parallel==2
    % Simulation on GPU is really slow.
    % So instead, switch to CPU, and then switch
    % back. For anything but ridiculously short simulations it is more than
    % worth the overhead.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    Phi_aprimeKron=gather(Phi_aprimeKron);
    pi_z=gather(pi_z);
    MoveOutputToGPU=1;
end

SimTimeSeriesKron=zeros(2,simperiods);

currstate=seedpoint;
cumsum_pi_z=cumsum(pi_z,2);

if Case2_Type==1
    if burnin>0
        for t=1:burnin
            z_c=currstate(2);
            optd=PolicyIndexesKron(currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            currstate(1)=Phi_aprimeKron(optd,a_c,z_c,currstate(2));
        end
    end
    for t=1:simperiods
        SimTimeSeriesKron(1,t)=currstate(1); %a_c
        SimTimeSeriesKron(2,t)=currstate(2); %z_c
        
        z_c=currstate(2);
        optd=PolicyIndexesKron(currstate(1),currstate(2));
        [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        currstate(1)=Phi_aprimeKron(optd,a_c,z_c,currstate(2));
    end
elseif Case2_Type==2
    if burnin>0
        for t=1:burnin
            z_c=currstate(2);
            optd=PolicyIndexesKron(currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            currstate(1)=Phi_aprimeKron(optd,z_c,currstate(2));
        end
    end
    for t=1:simperiods
        SimTimeSeriesKron(1,t)=currstate(1); %a_c
        SimTimeSeriesKron(2,t)=currstate(2); %z_c
        
        z_c=currstate(2);
        optd=PolicyIndexesKron(currstate(1),currstate(2));
        [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        currstate(1)=Phi_aprimeKron(optd,z_c,currstate(2));
    end
elseif Case2_Type==3
    if burnin>0
        for t=1:burnin
            optd=PolicyIndexesKron(currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            currstate(1)=Phi_aprimeKron(optd);
        end
    end
    for t=1:simperiods
        SimTimeSeriesKron(1,t)=currstate(1); %a_c
        SimTimeSeriesKron(2,t)=currstate(2); %z_c
        
        optd=PolicyIndexesKron(currstate(1),currstate(2));
        [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        currstate(1)=Phi_aprimeKron(optd);
    end
end

if MoveOutputToGPU==1
    SimTimeSeriesKron=gpuArray(SimTimeSeriesKron);
end

%% OLD CODE
% %Simulates a path based on PolicyIndexes (and Phi_aprime) of length 'periods' after a burn
% %in of length 'burnin' (burn-in are the initial run of points that are then
% %dropped)
% %Note: N_d is not actually needed, it is just left in so inputs are more
% %like those for Case1
% 
% %First, generate the transition matrix P=phi of Q (in the notation of SLP)
% P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
% if Case2_Type==1
%     for z_c=1:N_z
%         for a_c=1:N_a
%             optd=PolicyIndexesKron(a_c,z_c);
%             for zprime_c=1:N_z
%                 optaprime=Phi_aprimeKron(optd,a_c,z_c,zprime_c);
%                 P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%             end
%         end
%     end
% elseif Case2_Type==2
%     for z_c=1:N_z
%         for a_c=1:N_a
%             optd=PolicyIndexesKron(a_c,z_c);
%             for zprime_c=1:N_z
%                 optaprime=Phi_aprimeKron(optd,z_c,zprime_c);
%                 P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%             end
%         end
%     end
% elseif Case2_Type==3
%     for z_c=1:N_z
%         for a_c=1:N_a
%             optd=PolicyIndexesKron(a_c,z_c);
%             optaprime=Phi_aprimeKron(optd);
%             for zprime_c=1:N_z
%                 P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%             end
%         end
%     end
% end
% P=reshape(P,[N_a*N_z,N_a*N_z]);
% %Now turn P into a cumulative distn
% P=cumsum(P,2);
% 
% SimTimeSeriesKron=zeros(2,periods);
% 
% currstate=sub2ind_homemade([N_a,N_z],seedpoint); 
% 
% for t=1:burnin
%     [~,currstate]=max(P(currstate,:)>rand(1,1));
% end
% for t=1:periods
%     temp=ind2sub_homemade([N_a,N_z], currstate);
%     SimTimeSeriesKron(1,t)=temp(1); %a_c
%     SimTimeSeriesKron(2,t)=temp(2); %z_c
%     [~,currstate]=max(P(currstate,:)>rand(1,1));
% end

end