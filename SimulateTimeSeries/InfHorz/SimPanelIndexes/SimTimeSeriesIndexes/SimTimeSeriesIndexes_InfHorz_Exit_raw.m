function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_Exit_raw(Policy_aprime, CondlProbOfSurvivalKron,cumsum_pi_z,burnin,seedpoint,simperiods)

% burnin=simoptions.burnin;
% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;

%Simulates path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped). The burn in begins at point 'seedpoint' (this is not just left
%as being random since some random points may be ones that never 'exist' in
%eqm)

SimTimeSeriesKron=nan(2,simperiods); % Will just leave the nan on exit/death

currstate=seedpoint;
% cumsum_pi_z=cumsum(pi_z,2);

if burnin>0
    for t=1:burnin
        if rand(1,1)>CondlProbOfSurvivalKron(currstate(1),currstate(2)) % Death/Exit
            currstate(1)=0;
            break
        end
        currstate(1)=Policy_aprime(currstate(1),currstate(2));
        [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
    end
end
if currstate(1)>0 % Still alive
    for t=1:simperiods
        SimTimeSeriesKron(1,t)=currstate(1); %a_c
        SimTimeSeriesKron(2,t)=currstate(2); %z_c

        if rand(1,1)>CondlProbOfSurvivalKron(currstate(1),currstate(2)) % Death/Exit
            currstate(1)=0;
            break
        end

        currstate(1)=Policy_aprime(currstate(1),currstate(2));
        [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));

    end
end

end
