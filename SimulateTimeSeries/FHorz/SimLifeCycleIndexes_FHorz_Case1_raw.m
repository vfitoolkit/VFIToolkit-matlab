function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,N_j,pi_z,seedpoint,simperiods)
% All inputs must be on the CPU
%
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm)
%
% Outputs the indexes for (a,z) for every period j. This is for period 1 to
% J. Since most simulations will not start at period 1, the first entries
% are typically 'NaN'.

% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;

SimLifeCycleKron=nan(2,N_j);

% if N_d==0
%     optaprime=1;
% else
%     optaprime=2;
% end
currstate=seedpoint;
cumsum_pi_z=cumsum(pi_z,2);

% seedpoint is (a,z,j)

% Simulation is simperiods, or up to 'end of finite horizon'.
periods=min(simperiods,N_j+1-seedpoint(3));

if N_d==0
    for jj=1:periods
        SimLifeCycleKron(1,jj+seedpoint(3)-1)=currstate(1); %a_c
        SimLifeCycleKron(2,jj+seedpoint(3)-1)=currstate(2); %z_c
        
        currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),jj+seedpoint(3)-1);
        [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
    end
else
    for jj=1:periods
        SimLifeCycleKron(1,jj+seedpoint(3)-1)=currstate(1); %a_c
        SimLifeCycleKron(2,jj+seedpoint(3)-1)=currstate(2); %z_c
        
        currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),jj+seedpoint(3)-1);
        [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
    end
end
% for jj=1:periods
%     SimLifeCycleKron(1,jj+seedpoint(3)-1)=currstate(1); %a_c
%     SimLifeCycleKron(2,jj+seedpoint(3)-1)=currstate(2); %z_c
%     
%     currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2),jj+seedpoint(3)-1);
%     [~,currstate(2)]=max(cumsum_pi_z(currstate(2),:)>rand(1,1));
% end
end
