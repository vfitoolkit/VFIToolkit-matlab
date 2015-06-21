function SteadyStateDistKron=SteadyState_Case1_Simulation_raw(PolicyIndexesKron,N_d,N_a,N_z,pi_z, simoptions)
%Simulates path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)

if nargin<6
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

%N_a=prod(n_a);
%N_z=prod(n_z);
%num_a_vars=length(n_a);

% %First, generate the transition matrix P=phi of Q (in the notation of SLP)
% P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
% for a_c=1:N_a
%     for z_c=1:N_z
%         if N_d==0 %length(n_d)==1 && n_d(1)==0
%                 optaprime=PolicyIndexesKron(a_c,z_c);
%             else
%                 optaprime=PolicyIndexesKron(2,a_c,z_c);
%         end
%         for zprime_c=1:N_z
% %             if N_d==0 %length(n_d)==1 && n_d(1)==0
% %                 optaprime=PolicyIndexesKron(1,a_c,z_c);
% %             else
% %                 optaprime=PolicyIndexesKron(2,a_c,z_c);
% %             end
%             P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%         end
%     end
% end
% P=reshape(P,[N_a*N_z,N_a*N_z]);
% %Now turn P into a cumulative distn
% P=cumsum(P,2);
%
% SteadyStateDistKron=zeros(N_a*N_z,1);
% currstate=ceil(rand(1,1)*N_a*N_z); %Pick a random start point on the (vectorized) (a,z) grid
% for i=1:burnin
%     [trash,currstate]=max(P(currstate,:)>rand(1,1));
% end
% for i=1:periods
%     SteadyStateDistKron(currstate)=SteadyStateDistKron(currstate)+1;
%     [trash,currstate]=max(P(currstate,:)>rand(1,1));
% end
% SteadyStateDistKron=SteadyStateDistKron./periods;


% if N_d==0 %length(n_d)==1 && n_d(1)==0
%     optaprime=1;
% else
%     optaprime=2;
% end

MoveSSDKtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow.
    % So instead, switch to CPU, use all available cores, and then switch
    % back. For anything but ridiculously short simulations it is more than
    % worth the overhead.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    pi_z=gather(pi_z);
    simoptions.parallel=1;
    MoveSSDKtoGPU=1;
end


if simoptions.parallel==1
    eachsimperiods=ceil(simoptions.simperiods/simoptions.ncores);
    %     disp('Create simoptions.ncores different steady state distns, then combine them')
    SteadyStateDistKron=zeros(N_a*N_z,simoptions.ncores);
    cumsum_pi_z=cumsum(pi_z,2);
    %Create NumOfSeedPoint different steady state distn's, then combine them.
    if N_d==0
        parfor seed_c=1:simoptions.ncores
            SteadyStateDistKron_seed_c=zeros(N_a*N_z,1);
            currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
            for i=1:simoptions.burnin
                currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
                currstate(2)=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
            for i=1:eachsimperiods
                SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)+1;
                
                currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
                currstate(2)=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
            SteadyStateDistKron(:,seed_c)=SteadyStateDistKron_seed_c;
        end
        SteadyStateDistKron=sum(SteadyStateDistKron,2);
        SteadyStateDistKron=SteadyStateDistKron./sum(sum(SteadyStateDistKron));
    else
        optaprime=2;
        parfor seed_c=1:simoptions.ncores
            SteadyStateDistKron_seed_c=zeros(N_a*N_z,1);
            currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
            for i=1:simoptions.burnin
                currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2));
                currstate(2)=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
            for i=1:eachsimperiods
                SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron_seed_c(currstate(1)+(currstate(2)-1)*N_a)+1;
                
                currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2));
                currstate(2)=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            end
            SteadyStateDistKron(:,seed_c)=SteadyStateDistKron_seed_c;
        end
        SteadyStateDistKron=sum(SteadyStateDistKron,2);
        SteadyStateDistKron=SteadyStateDistKron./sum(sum(SteadyStateDistKron));
    end
elseif simoptions.parallel==0
    if N_d==0
        cumsum_pi_z=cumsum(pi_z,2);
        SteadyStateDistKron=zeros(N_a*N_z,1);
        currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
        for i=1:simoptions.burnin
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
        for i=1:simoptions.simperiods
            SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
            
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            
        end
        SteadyStateDistKron=SteadyStateDistKron./sum(sum(SteadyStateDistKron));
    else
        optaprime=2;
        cumsum_pi_z=cumsum(pi_z,2);
        SteadyStateDistKron=zeros(N_a*N_z,1);
        currstate=simoptions.seedpoint; %Pick a random start point on the (vectorized) (a,z) grid
        for i=1:simoptions.burnin
            currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
        for i=1:simoptions.simperiods
            SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
            
            currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            
        end
        SteadyStateDistKron=SteadyStateDistKron./sum(sum(SteadyStateDistKron));
    end
%    SteadyStateDistKron(:,seed_c)=SteadyStateDistKron;
% % % elseif simoptions.parallel==2 %% WAY TOO SLOW. HAVE CHANGED TO SIMPLY MOVING TO CPU AND THEN MOVING BACK 
% % %     if N_d==0
% % %         v1=0
% % %         if v1==1
% % %             % THIS IS A BETTER VERSION, BUT WON'T RUN (ON GPU) UNTIL MATLAB
% % %             % R2014B AT LEAST
% % %             %Uses gpu to run simoptions.ncores simulations simultaneously
% % %             simoptions.ncores=100;
% % %             SteadyStateDistKron=zeros(N_a*N_z,1,'gpuArray');
% % %             cumsum_pi_z=cumsum(pi_z,2);
% % %             currstate=ones(simoptions.ncores,2,'gpuArray').*(ones(simoptions.ncores,1,'gpuArray')*simoptions.seedpoint); %Pick a random start point on the (vectorized) (a,z) grid
% % %             for i=1:simoptions.burnin
% % %                 currstate(:,1)=PolicyIndexesKron(currstate(:,1)+N_a*(currstate(:,2)-1));
% % %                 [~,currstate(:,2)]=max((cumsum_pi_z(currstate(:,2),:)>rand(simoptions.ncores,1,'gpuArray')*ones(1,N_z)),[],2);
% % % %                 [~,currstate(:,2)]=max(cumsum_pi_z(currstate(:,2),:)>(rand(simoptions.ncores,1,'gpuArray')*ones(1,4,'gpuArray')),[],2);
% % %             end
% % %             for i=1:simoptions.simperiods
% % %                 temp=currstate(:,1)+N_a*(currstate(:,2)-1);
% % % %                 % histcounts does not yet exist in Matlab 2014a, but will in
% % % %                 % Matlab 2014b (may still not yet exist for GPU)
% % % %                 temp=gather(temp);
% % %                 [uniquecurrstate,~,AAA]=unique(temp);
% % %                 [Ncurrstate,~] = histcounts(AAA,length(uniquecurrstate));
% % % %                 [Ncurrstate,uniquecurrstate] = histcounts(temp,length(unique(temp)));
% % %                 % may need to run ceil() or similar on uniquecurrstate
% % %                 SteadyStateDistKron(uniquecurrstate)=SteadyStateDistKron(uniquecurrstate)+Ncurrstate';
% % %                 
% % %                 currstate(:,1)=PolicyIndexesKron(currstate(:,1)+N_a*(currstate(:,2)-1));
% % %                 [~,currstate(:,2)]=max((cumsum_pi_z(currstate(:,2),:)>rand(simoptions.ncores,1,'gpuArray')*ones(1,N_z)),[],2);
% % % %                 [~,currstate(:,2)]=max(cumsum_pi_z(currstate(:,2),:)>(rand(simoptions.ncores,1,'gpuArray')*ones(1,4,'gpuArray')),[],2);
% % %             end
% % %             SteadyStateDistKron=SteadyStateDistKron./(simoptions.simperiods*simoptions.ncores);
% % %         else
% % %             % TEMPORARILY STICKING WITH THIS
% % %             cumsum_pi_z=cumsum(pi_z,2);
% % %             SteadyStateDistKron=zeros(N_a*N_z,1,'gpuArray');
% % %             currstate=gpuArray(simoptions.seedpoint); %Pick a random start point on the (vectorized) (a,z) grid
% % %             for i=1:simoptions.burnin
% % %                 currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
% % %                 [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
% % %             end
% % %             for i=1:simoptions.simperiods
% % %                 SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
% % %                 
% % %                 currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
% % %                 [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
% % %                 
% % %             end
% % %             SteadyStateDistKron=SteadyStateDistKron./sum(sum(SteadyStateDistKron));
% % %         end
% % %     else
% % %         optaprime=2;
% % %         cumsum_pi_z=cumsum(pi_z,2);
% % %         SteadyStateDistKron=zeros(N_a*N_z,1,'gpuArray');
% % %         currstate=gpuArray(simoptions.seedpoint); %Pick a random start point on the (vectorized) (a,z) grid
% % %         for i=1:simoptions.burnin
% % %             currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2));
% % %             [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
% % %         end
% % %         for i=1:simoptions.simperiods
% % %             SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)=SteadyStateDistKron(currstate(1)+(currstate(2)-1)*N_a)+1;
% % %             
% % %             currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2));
% % %             [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
% % %             
% % %         end
% % %         SteadyStateDistKron=SteadyStateDistKron./sum(sum(SteadyStateDistKron));
% % %     end
    
end

if MoveSSDKtoGPU==1
    SteadyStateDistKron=gpuArray(SteadyStateDistKron);
end

end
