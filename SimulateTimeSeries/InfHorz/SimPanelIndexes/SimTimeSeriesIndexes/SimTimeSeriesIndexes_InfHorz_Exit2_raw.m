function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_Exit2_raw(PolicyIndexesKron, CondlProbOfSurvivalKron,N_d,N_a,N_z,cumsum_pi_z,burnin,seedpoint,simperiods,exitprobabilities,parallel)
% Exit2 is same as Exit, but allowing for 'mix' of endogenous and exogenous exit.

% burnin=simoptions.burnin;
% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;

%Simulates path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped). The burn in begins at point 'seedpoint' (this is not just left
%as being random since some random points may be ones that never 'exist' in
%eqm)

% NEED TO USE exitprobabilities

MoveSTSKtoGPU=0;
if parallel==2
    % Simulation on GPU is really slow.
    % So instead, switch to CPU, and then switch
    % back. For anything but ridiculously short simulations it is more than
    % worth the overhead.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    cumsum_pi_z=gather(cumsum_pi_z);
    MoveSTSKtoGPU=1;
end

SimTimeSeriesKron=nan(2,simperiods); % Will just leave the nan on exit/death
% The kind of exit that occurs at time t is recorded in the time t+1 exogenous state as a value of 0 for endog exit.
% (Note: so exogenous just leaves nan from then on, endog exit leaves 0 in
% next period exogenous state and otherwise just leaves nan from then on. Notice that a zero value will throw an error if just treated as a standard index.)

currstate=seedpoint;
% cumsum_pi_z=cumsum(pi_z,2);

if N_d==0
%     optaprime=1;
    if burnin>0
        for t=1:burnin
            
            temp=rand(1,1);
            if temp>exitprobabilities(1)+exitprobabilities(2) % exog exit
                currstate(1)=0;
                break
            elseif temp>exitprobabilities(1) % endog exit (maybe, agent has to make decision))
                if CondlProbOfSurvivalKron(currstate(1),currstate(2))>0 % agent chooses Death/Exit (Note that CondlProbOfSurvivalKron contains 1-ExitPolicy, so really checking for ExitPolicy==1)
                    currstate(1)=0;
                    break
                end
            end
            
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
    end
    if currstate(1)>0 % Still alive
        for t=1:simperiods
            SimTimeSeriesKron(1,t)=currstate(1); %a_c
            SimTimeSeriesKron(2,t)=currstate(2); %z_c
            
            temp=rand(1,1);
            if temp>exitprobabilities(1)+exitprobabilities(2) % exog exit
                break
            elseif temp>exitprobabilities(1) % endog exit (maybe, agent has to make decision))
                if CondlProbOfSurvivalKron(currstate(1),currstate(2))>0 % agent chooses Death/Exit (Note that CondlProbOfSurvivalKron contains 1-ExitPolicy, so really checking for ExitPolicy==1)
                    SimTimeSeriesKron(2,t+1)=0;
                    break
                end
            end
            
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
            
        end
    end
else
%     optaprime=2;
    if burnin>0
        for t=1:burnin

            temp=rand(1,1);
            if temp>exitprobabilities(1)+exitprobabilities(2) % exog exit
                currstate(1)=0;
                break
            elseif temp>exitprobabilities(1) % endog exit (maybe, agent has to make decision))
                if CondlProbOfSurvivalKron(currstate(1),currstate(2))>0 % agent chooses Death/Exit (Note that CondlProbOfSurvivalKron contains 1-ExitPolicy, so really checking for ExitPolicy==1)
                    currstate(1)=0;
                    break
                end
            end
            
            currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
    end
    if currstate(1)>0 % Still alive
        for t=1:simperiods
            SimTimeSeriesKron(1,t)=currstate(1); %a_c
            SimTimeSeriesKron(2,t)=currstate(2); %z_c
            
            temp=rand(1,1);
            if temp>exitprobabilities(1)+exitprobabilities(2) % exog exit
                break
            elseif temp>exitprobabilities(1) % endog exit (maybe, agent has to make decision))
                if CondlProbOfSurvivalKron(currstate(1),currstate(2))>0 % agent chooses Death/Exit (Note that CondlProbOfSurvivalKron contains 1-ExitPolicy, so really checking for ExitPolicy==1)
                    SimTimeSeriesKron(2,t+1)=0;
                    break
                end
            end
            
            currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2));
            [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
        end
    end
end

if MoveSTSKtoGPU==1
    SimTimeSeriesKron=gpuArray(SimTimeSeriesKron);
end

end
