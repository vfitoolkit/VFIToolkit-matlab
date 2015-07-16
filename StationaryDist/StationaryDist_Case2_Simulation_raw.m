function StationaryDistKron=StationaryDist_Case2_Simulation_raw(PolicyIndexesKron,Phi_aprimeKron, Case2_Type,N_d,N_a,N_z,pi_z, simoptions)
%Simulates a path based on PolicyIndexes (and Phi_aprime) of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)
%Note: N_d is not actually needed, it is just left in so inputs are more
%like those for Case1

% Options needed
%    simoptions.seedpoint
%    simoptions.simperiods
%    simoptions.burnin
%    simoptions.parallel
%    simoptions.verbose
%    simoptions.ncores

MoveSSDKtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow.
    % So instead, switch to CPU, use all available cores, and then switch
    % back. For anything but ridiculously short simulations it is more than
    % worth the overhead.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    Phi_aprimeKron=gather(Phi_aprimeKron);
    pi_z=gather(pi_z);
    simoptions.parallel=1;
    MoveSSDKtoGPU=1;
end

%%

if simoptions.parallel==1
    eachsimperiods=ceil(simoptions.simperiods/simoptions.ncores);
    %     disp('Create simoptions.ncores different steady state distns, then combine them')
    StationaryDistKron=zeros(N_a*N_z,simoptions.ncores);
    cumsum_pi_z=cumsum(pi_z,2);
    %Create NumOfSeedPoint different steady state distn's, then combine them.
    if Case2_Type==1
        parfor seed_c=1:simoptions.ncores
            SteadyStateDistKron_seed_c=zeros(N_a*N_z,1);
            currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
            for i=1:simoptions.burnin
                optd=PolicyIndexesKron(currstate(1),currstate(2));
                [trash,newcurrstate2]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
                currstate(1)=Phi_aprimeKron(optd,currstate(1),currstate(2),newcurrstate2);
                currstate(2)=newcurrstate2;
            end
            for i=1:eachsimperiods
                SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)+1;
                
                optd=PolicyIndexesKron(currstate(1),currstate(2));
                [trash,newcurrstate2]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
                currstate(1)=Phi_aprimeKron(optd,currstate(1),currstate(2),newcurrstate2);
                currstate(2)=newcurrstate2;
            end
            StationaryDistKron(:,seed_c)=SteadyStateDistKron_seed_c;
        end
    elseif Case2_Type==2
        parfor seed_c=1:simoptions.ncores
            SteadyStateDistKron_seed_c=zeros(N_a*N_z,1);
            currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
            for i=1:simoptions.burnin
                optd=PolicyIndexesKron(currstate(1),currstate(2));
                [trash,newcurrstate2]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
                currstate(1)=Phi_aprimeKron(optd,currstate(2),newcurrstate2);
                currstate(2)=newcurrstate2;
            end
            for i=1:eachsimperiods
                SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)+1;
                
                optd=PolicyIndexesKron(currstate(1),currstate(2));
                [trash,newcurrstate2]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
                currstate(1)=Phi_aprimeKron(optd,currstate(2),newcurrstate2);
                currstate(2)=newcurrstate2;
            end
            StationaryDistKron(:,seed_c)=SteadyStateDistKron_seed_c;
        end
    elseif Case2_Type==3
        parfor seed_c=1:simoptions.ncores
            SteadyStateDistKron_seed_c=zeros(N_a*N_z,1);
            currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
            for i=1:simoptions.burnin
                optd=PolicyIndexesKron(currstate(1),currstate(2));
                currstate(1)=Phi_aprimeKron(optd);
                [trash,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
            for i=1:eachsimperiods
                SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)+1;
                
                optd=PolicyIndexesKron(currstate(1),currstate(2));
                currstate(1)=Phi_aprimeKron(optd);
                [trash,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
            StationaryDistKron(:,seed_c)=SteadyStateDistKron_seed_c;
        end
    end
    StationaryDistKron=sum(StationaryDistKron,2)./(eachsimperiods*simoptions.ncores);
    
elseif simoptions.parallel==0
    StationaryDistKron=zeros(N_a*N_z);
    cumsum_pi_z=cumsum(pi_z,2);
    %Create NumOfSeedPoint different steady state distn's, then combine them.
    if Case2_Type==1
        currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
        for i=1:simoptions.burnin
            optd=PolicyIndexesKron(currstate(1),currstate(2));
            [trash,newcurrstate2]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            currstate(1)=Phi_aprimeKron(optd,currstate(1),currstate(2),newcurrstate2);
            currstate(2)=newcurrstate2;
        end
        for i=1:simoptions.simperiods
            StationaryDistKron(currstate(1)+(currstate(2)-1)*N_a)=StationaryDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
            
            optd=PolicyIndexesKron(currstate(1),currstate(2));
            [trash,newcurrstate2]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            currstate(1)=Phi_aprimeKron(optd,currstate(1),currstate(2),newcurrstate2);
            currstate(2)=newcurrstate2;
        end
    elseif Case2_Type==2
        currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
        for i=1:simoptions.burnin
            optd=PolicyIndexesKron(currstate(1),currstate(2));
            [trash,newcurrstate2]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            currstate(1)=Phi_aprimeKron(optd,currstate(2),newcurrstate2);
            currstate(2)=newcurrstate2;
        end
        for i=1:simoptions.simperiods
            StationaryDistKron(currstate(1)+(currstate(2)-1)*N_a)=StationaryDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
            
            optd=PolicyIndexesKron(currstate(1),currstate(2));
            [trash,newcurrstate2]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            currstate(1)=Phi_aprimeKron(optd,currstate(2),newcurrstate2);
            currstate(2)=newcurrstate2;
        end
    elseif Case2_Type==3
        currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
        for i=1:simoptions.burnin
            optd=PolicyIndexesKron(currstate(1),currstate(2));
            currstate(1)=Phi_aprimeKron(optd);
            [trash,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
        for i=1:simoptions.simperiods
            StationaryDistKron(currstate(1)+(currstate(2)-1)*N_a)=StationaryDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
            
            optd=PolicyIndexesKron(currstate(1),currstate(2));
            currstate(1)=Phi_aprimeKron(optd);
            [trash,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
    end
    StationaryDistKron=sum(StationaryDistKron,2)./(simoptions.simperiods);
end

if MoveSSDKtoGPU==1
    StationaryDistKron=gpuArray(StationaryDistKron);
end

end