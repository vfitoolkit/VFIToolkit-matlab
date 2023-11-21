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
periods=min(simperiods,N_j-initialage);

if N_d==0
    SimLifeCycleKron=nan(3,N_j);
    for jj=0:periods
        SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
        SimLifeCycleKron(2,jj+initialage)=currstate(2); % z_c

        currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),jj+initialage);
        [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj+initialage)>rand(1,1));
    end
    SimLifeCycleKron(3,initialage:(initialage+periods))=initialage:1:(initialage+periods);
else
    if ~exist('include_daprime','var')
        include_daprime=0;
    end    
    if include_daprime==0
        SimLifeCycleKron=nan(3,N_j);
        for jj=0:periods
            SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+initialage)=currstate(2); % z_c

            currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),jj+initialage);
            [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj+initialage)>rand(1,1));
        end
        SimLifeCycleKron(3,initialage:(initialage+periods))=initialage:1:(initialage+periods);
    else
        SimLifeCycleKron=nan(5,N_j);
        for jj=0:periods
            SimLifeCycleKron(3,jj+initialage)=currstate(1); % a
            SimLifeCycleKron(4,jj+initialage)=currstate(2); % z

            curr_d=PolicyIndexesKron(1,currstate(1),currstate(2),jj+initialage);
            SimLifeCycleKron(1,jj+initialage)=curr_d; % d

            currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),jj+initialage);
            [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj+initialage)>rand(1,1));

            SimLifeCycleKron(2,jj+initialage)=currstate(1); % aprime
        end
        SimLifeCycleKron(5,initialage:(initialage+periods))=initialage:1:(initialage+periods);
    end
end


end