function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyIndexesKron,N_d,N_j,cumsumpi_z_J,seedpoint,simperiods,include_daprime)
% All inputs must be on the CPU
%
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm)
%
% Outputs the indexes for (a,z,j) for every period j. This is for period 1 to J. 
% Since most simulations will not start at period 1, the first entries are typically 'NaN'.

% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;

currstate=seedpoint;

% seedpoint is (a,z,j)

initialage=seedpoint(3); % j in (a,z,j)

% Simulation is simperiods, or up to 'end of finite horizon'.
periods=min(simperiods,N_j+1-initialage);

if N_d==0
    SimLifeCycleKron=nan(3,N_j);
    for jj=1:periods
        SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
        SimLifeCycleKron(2,jj+initialage-1)=currstate(2); % z_c

        currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),jj+initialage-1);
        [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj)>rand(1,1));
    end
    SimLifeCycleKron(3,seedpoint(3):end)=seedpoint(3):1:N_j;
else
    if ~exist('include_daprime','var')
        include_daprime=0;
    end
    if include_daprime==0
        SimLifeCycleKron=nan(3,N_j);
        for jj=1:periods
            SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+initialage-1)=currstate(2); % z_c

            currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),jj+initialage-1);
            [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj)>rand(1,1));
        end
        SimLifeCycleKron(3,seedpoint(3):end)=seedpoint(3):1:N_j;
    else
        SimLifeCycleKron=nan(5,N_j);
        for jj=1:periods
            SimLifeCycleKron(3,jj+initialage-1)=currstate(1); % a
            SimLifeCycleKron(4,jj+initialage-1)=currstate(2); % z

            curr_d=PolicyIndexesKron(1,currstate(1),currstate(2),jj+initialage-1);
            SimLifeCycleKron(1,jj+initialage-1)=curr_d; % d

            currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),jj+initialage-1);
            [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj)>rand(1,1));

            SimLifeCycleKron(2,jj+initialage-1)=currstate(1); % aprime
        end
        SimLifeCycleKron(5,seedpoint(3):end)=seedpoint(3):1:N_j;
    end
end


end