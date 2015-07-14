function SteadyStateDistKron=SteadyState_Case2_Simulation_raw(PolicyIndexesKron,Phi_aprimeKron, Case2_Type,N_d,N_a,N_z,pi_z, simoptions)
%Simulates a path based on PolicyIndexes (and Phi_aprime) of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)
%Note: N_d is not actually needed, it is just left in so inputs are more
%like those for Case1

if nargin<8
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10^4;
    simoptions.burnin=10^3;
    simoptions.parallel=0;
    simoptions.verbose=0;
    simoptions.ncores=1;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;simoptions.seedpoint;','fieldexists=0;')
    if fieldexists==0
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
    eval('fieldexists=1;simoptions.simperiods;','fieldexists=0;')
    if fieldexists==0
        simoptions.simperiods=10^4;
    end
    eval('fieldexists=1;simoptions.burnin;','fieldexists=0;')
    if fieldexists==0
        simoptions.burnin=10^3;
    end
    eval('fieldexists=1;simoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        simoptions.parallel=0;
    end
    eval('fieldexists=1;simoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        simoptions.verbose=0;
    end
    eval('fieldexists=1;simoptions.ncores;','fieldexists=0;')
    if fieldexists==0
        simoptions.ncores=1;
    end
end

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
    SteadyStateDistKron=zeros(N_a*N_z,simoptions.ncores);
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
            SteadyStateDistKron(:,seed_c)=SteadyStateDistKron_seed_c;
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
            SteadyStateDistKron(:,seed_c)=SteadyStateDistKron_seed_c;
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
            SteadyStateDistKron(:,seed_c)=SteadyStateDistKron_seed_c;
        end
    end
    SteadyStateDistKron=sum(SteadyStateDistKron,2)./(eachsimperiods*simoptions.ncores);
    
elseif simoptions.parallel==0
    SteadyStateDistKron=zeros(N_a*N_z);
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
            SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
            
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
            SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
            
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
            SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
            
            optd=PolicyIndexesKron(currstate(1),currstate(2));
            currstate(1)=Phi_aprimeKron(optd);
            [trash,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
    end
    SteadyStateDistKron=sum(SteadyStateDistKron,2)./(simoptions.simperiods);
end

if MoveSSDKtoGPU==1
    SteadyStateDistKron=gpuArray(SteadyStateDistKron);
end

end