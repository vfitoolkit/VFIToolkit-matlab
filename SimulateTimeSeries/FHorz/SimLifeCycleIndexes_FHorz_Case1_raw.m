function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,N_j,cumsumpi_z,seedpoint,simperiods,fieldexists_pi_z_J,include_daprime)
% All inputs must be on the CPU
%
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm)
%
% Outputs the indexes for (a,z) for every period j. This is for period 1 to J. 
% Since most simulations will not start at period 1, the first entries are typically 'NaN'.

% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;

currstate=seedpoint;

% seedpoint is (a,z,j)

% Simulation is simperiods, or up to 'end of finite horizon'.
periods=min(simperiods,N_j+1-seedpoint(3));

if fieldexists_pi_z_J==0
    if N_d==0
        SimLifeCycleKron=nan(2,N_j);
        for jj=1:periods
            SimLifeCycleKron(1,jj+seedpoint(3)-1)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+seedpoint(3)-1)=currstate(2); % z_c
            
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),jj+seedpoint(3)-1);
            [~,currstate(2)]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
        end
    else
        if ~exist('include_daprime','var')
            include_daprime=0;
        end
        if include_daprime==0
            SimLifeCycleKron=nan(2,N_j);
            for jj=1:periods
                SimLifeCycleKron(1,jj+seedpoint(3)-1)=currstate(1); % a_c
                SimLifeCycleKron(2,jj+seedpoint(3)-1)=currstate(2); % z_c
                
                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),jj+seedpoint(3)-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
            end
        else
            SimLifeCycleKron=nan(4,N_j);
            for jj=1:periods
                SimLifeCycleKron(3,jj+seedpoint(3)-1)=currstate(1); % a_c
                SimLifeCycleKron(4,jj+seedpoint(3)-1)=currstate(2); % z_c
                
                curr_d=PolicyIndexesKron(1,currstate(1),currstate(2),jj+seedpoint(3)-1);
                SimLifeCycleKron(1,jj+seedpoint(3)-1)=curr_d; % d_c

                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),jj+seedpoint(3)-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
                
                SimLifeCycleKron(2,jj+seedpoint(3)-1)=currstate(1); % aprime_c
            end
        end
    end
else
    if N_d==0
        SimLifeCycleKron=nan(2,N_j);
        for jj=1:periods
            SimLifeCycleKron(1,jj+seedpoint(3)-1)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+seedpoint(3)-1)=currstate(2); % z_c
            
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),jj+seedpoint(3)-1);
            [~,currstate(2)]=max(cumsumpi_z(currstate(2),:,jj)>rand(1,1));
        end
    else
        if ~exist('include_daprime','var')
            include_daprime=0;
        end
        if include_daprime==0
            SimLifeCycleKron=nan(2,N_j);
            for jj=1:periods
                SimLifeCycleKron(1,jj+seedpoint(3)-1)=currstate(1); % a_c
                SimLifeCycleKron(2,jj+seedpoint(3)-1)=currstate(2); % z_c
                
                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),jj+seedpoint(3)-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:,jj)>rand(1,1));
            end
        else
            SimLifeCycleKron=nan(4,N_j);
            for jj=1:periods
                SimLifeCycleKron(3,jj+seedpoint(3)-1)=currstate(1); % a
                SimLifeCycleKron(4,jj+seedpoint(3)-1)=currstate(2); % z
                
                curr_d=PolicyIndexesKron(1,currstate(1),currstate(2),jj+seedpoint(3)-1);
                SimLifeCycleKron(1,jj+seedpoint(3)-1)=curr_d; % d

                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),jj+seedpoint(3)-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:,jj)>rand(1,1));
                
                SimLifeCycleKron(2,jj+seedpoint(3)-1)=currstate(1); % aprime
            end            
        end
    end
end
