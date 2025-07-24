function StationaryDistKron=StationaryDist_InfHorz_Simulation_raw(PolicyIndexesKron,N_d,N_a,N_z,pi_z, simoptions)
%Simulates path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)
% 
% Options needed
%    simoptions.seedpoint
%    simoptions.simperiods
%    simoptions.burnin
%    simoptions.parallel
%    simoptions.verbose
%    simoptions.ncores
%
%%

MoveSSDKtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow.
    % So instead, switch to CPU. If going to iterate then use all available cores for speed, but doing this is not 
    % in line with the mathematical theories on convergence, so if not iterating just use one core.
    % For anything but ridiculously short simulations it is more than worth the overhead to switch to CPU and back.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    pi_z=gather(pi_z);
    if simoptions.iterate==1
        simoptions.parallel=1; % multiple short time series, this is just creating an intial guess of the agent distribution for the iteration so perfectly good way to do things
    else
        simoptions.parallel=0; % For the asymptotic theory to work you have to just use one big time series. (Theory does not cover the 'multiple short time series' approach of parallel cpus with simoptions.parallel=1)
    end
    MoveSSDKtoGPU=1;
end

if simoptions.parallel==1 
    eachsimperiods=ceil(simoptions.simperiods/simoptions.ncores);
    %     disp('Create simoptions.ncores different steady state distns, then combine them')
    StationaryDistKron=zeros(N_a*N_z,simoptions.ncores);
    cumsum_pi_z=cumsum(pi_z,2);
    %Create NumOfSeedPoint different steady state distn's, then combine them.
    if N_d==0
        parfor seed_c=1:simoptions.ncores
            SteadyStateDistKron_seed_c=zeros(N_a*N_z,1);
            currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
            for i=1:simoptions.burnin
                currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
                [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
            for i=1:eachsimperiods
                SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)+1;
                
                currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
                [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
            StationaryDistKron(:,seed_c)=SteadyStateDistKron_seed_c;
        end
        StationaryDistKron=sum(StationaryDistKron,2);
        StationaryDistKron=StationaryDistKron./sum(sum(StationaryDistKron));
    else
        optaprime=2;
        parfor seed_c=1:simoptions.ncores
            SteadyStateDistKron_seed_c=zeros(N_a*N_z,1);
            currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
            for i=1:simoptions.burnin
                currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2));
                [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
            for i=1:eachsimperiods
                SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)+1;
                
                currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2));
                [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
            StationaryDistKron(:,seed_c)=SteadyStateDistKron_seed_c;
        end
        StationaryDistKron=sum(StationaryDistKron,2);
        StationaryDistKron=StationaryDistKron./sum(sum(StationaryDistKron));
    end
elseif simoptions.parallel==0
    if N_d==0
        cumsum_pi_z=cumsum(pi_z,2);
        StationaryDistKron=zeros(N_a*N_z,1);
        currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
        for i=1:simoptions.burnin
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
        for i=1:simoptions.simperiods
            StationaryDistKron(currstate(1)+(currstate(2)-1)*N_a)=StationaryDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
            
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            
        end
        StationaryDistKron=StationaryDistKron./sum(sum(StationaryDistKron));
    else
        optaprime=2;
        cumsum_pi_z=cumsum(pi_z,2);
        StationaryDistKron=zeros(N_a*N_z,1);
        currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
        for i=1:simoptions.burnin
            currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
        for i=1:simoptions.simperiods
            StationaryDistKron(currstate(1)+(currstate(2)-1)*N_a)=StationaryDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
            
            currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            
        end
        StationaryDistKron=StationaryDistKron./sum(sum(StationaryDistKron));
    end
    % I did once implement the simulations on gpu, but it is painfully slow. Much
    % better to switch to cpu, do the simluation, and then switch back.
end

if MoveSSDKtoGPU==1
    StationaryDistKron=gpuArray(StationaryDistKron);
end

end
