function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_semiz_e_raw(PolicyIndexesKron,N_d1,N_j,cumsumpi_z_J,cumsumpi_semiz_J,cumsumpi_e_J,seedpoint,simperiods)
% All inputs must be on the CPU
%
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm)
%
% Outputs the indexes for (a,z,semiz,e,j) for every period j. This is for period 1 to J. 
% Since most simulations will not start at period 1, the first entries are typically 'NaN'.

% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;

currstate=seedpoint;

% seedpoint is (a,z,semiz,e,j)

initialage=seedpoint(5); % j in (a,z,semiz,e,j)

% Simulation is simperiods, or up to 'end of finite horizon'.
periods=min(simperiods,N_j-initialage);

if N_d1==0
    SimLifeCycleKron=nan(5,N_j);
    for jj=0:periods
        SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
        SimLifeCycleKron(2,jj+initialage)=currstate(2); % semiz_c
        SimLifeCycleKron(3,jj+initialage)=currstate(3); % z_c
        SimLifeCycleKron(4,jj+initialage)=currstate(4); % e

        temp=PolicyIndexesKron(:,currstate(1),currstate(2),currstate(3),currstate(4),jj+initialage);
        % d2=temp(1);
        currstate(1)=temp(2);

        [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj+initialage)>rand(1,1));
        [~,currstate(3)]=max(cumsumpi_semiz_J(currstate(3),:,temp(1),jj+initialage)>rand(1,1));
        [~,currstate(4)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
    end
    SimLifeCycleKron(5,initialage:(initialage+periods))=initialage:1:(initialage+periods);
else
    SimLifeCycleKron=nan(5,N_j);
    for jj=0:periods
        SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
        SimLifeCycleKron(2,jj+initialage)=currstate(2); % semiz_c
        SimLifeCycleKron(3,jj+initialage)=currstate(3); % z_c
        SimLifeCycleKron(4,jj+initialage)=currstate(4); % e_c

        temp=PolicyIndexesKron(:,currstate(1),currstate(2),currstate(3),currstate(4),jj+initialage);
        currstate(1)=temp(3);

        [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj+initialage)>rand(1,1));
        [~,currstate(3)]=max(cumsumpi_semiz_J(currstate(3),:,temp(2),jj+initialage)>rand(1,1));
        [~,currstate(4)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
    end
    SimLifeCycleKron(5,initialage:(initialage+periods))=initialage:1:(initialage+periods);
end
