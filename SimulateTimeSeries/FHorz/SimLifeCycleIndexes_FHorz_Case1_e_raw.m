function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_e_raw(PolicyIndexesKron,N_d,N_j,cumsumpi_z,cumsumpi_e,seedpoint,simperiods,fieldexists_pi_z_J,fieldexists_pi_e_J,include_daprime)
% All inputs must be on the CPU
%
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm)
%
% Outputs the indexes for (a,z,e) for every period j. This is for period 1 to J. 
% Since most simulations will not start at period 1, the first entries are typically 'NaN'.

% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;

currstate=seedpoint;

% seedpoint is (a,z,e,j)

initialage=seedpoint(4); % j in (a,z,e,j)

% Simulation is simperiods, or up to 'end of finite horizon'.
periods=min(simperiods,N_j+1-initialage);

if fieldexists_pi_z_J==0 && fieldexists_pi_e_J==0
    if N_d==0
        SimLifeCycleKron=nan(4,N_j);
        for jj=1:periods
            SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+initialage-1)=currstate(2); % z_c
            SimLifeCycleKron(3,jj+initialage-1)=currstate(3); % e_c
            
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),currstate(3),jj+initialage-1);
            [~,currstate(2)]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
            [~,currstate(3)]=max(cumsumpi_e>rand(1,1));
        end
        SimLifeCycleKron(4,:)=seedpoint(4):1:N_j;
    else
        if ~exist('include_daprime','var')
            include_daprime=0;
        end
        if include_daprime==0
            SimLifeCycleKron=nan(4,N_j);
            for jj=1:periods
                SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
                SimLifeCycleKron(2,jj+initialage-1)=currstate(2); % z_c
                SimLifeCycleKron(3,jj+initialage-1)=currstate(3); % e_c
                
                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),currstate(3),jj+initialage-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
                [~,currstate(3)]=max(cumsumpi_e>rand(1,1));
            end
            SimLifeCycleKron(4,:)=seedpoint(4):1:N_j;
        else
            SimLifeCycleKron=nan(6,N_j); %d,aprime,a,z,e
            for jj=1:periods
                SimLifeCycleKron(3,jj+initialage-1)=currstate(1); % a_c
                SimLifeCycleKron(4,jj+initialage-1)=currstate(2); % z_c
                SimLifeCycleKron(5,jj+initialage-1)=currstate(3); % z_c
                
                curr_d=PolicyIndexesKron(1,currstate(1),currstate(2),jj+initialage-1);
                SimLifeCycleKron(1,jj+initialage-1)=curr_d; % d_c

                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),currstate(3),jj+initialage-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
                [~,currstate(3)]=max(cumsumpi_e>rand(1,1));
                
                SimLifeCycleKron(2,jj+initialage-1)=currstate(1); % aprime_c
            end
            SimLifeCycleKron(6,:)=seedpoint(4):1:N_j;
        end
    end
elseif fieldexists_pi_z_J==0 && fieldexists_pi_e_J==1
    if N_d==0
        SimLifeCycleKron=nan(4,N_j);
        for jj=1:periods
            SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+initialage-1)=currstate(2); % z_c
            SimLifeCycleKron(3,jj+initialage-1)=currstate(3); % e_c
            
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),currstate(3),jj+initialage-1);
            [~,currstate(2)]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
            [~,currstate(3)]=max(cumsumpi_e(:,jj)>rand(1,1));
        end
        SimLifeCycleKron(4,:)=seedpoint(4):1:N_j;
    else
        if ~exist('include_daprime','var')
            include_daprime=0;
        end
        if include_daprime==0
            SimLifeCycleKron=nan(4,N_j);
            for jj=1:periods
                SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
                SimLifeCycleKron(2,jj+initialage-1)=currstate(2); % z_c
                SimLifeCycleKron(3,jj+initialage-1)=currstate(3); % e_c
                
                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),currstate(3),jj+initialage-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
                [~,currstate(3)]=max(cumsumpi_e(:,jj)>rand(1,1));
            end
            SimLifeCycleKron(4,:)=seedpoint(4):1:N_j;
        else
            SimLifeCycleKron=nan(6,N_j);
            for jj=1:periods
                SimLifeCycleKron(3,jj+initialage-1)=currstate(1); % a_c
                SimLifeCycleKron(4,jj+initialage-1)=currstate(2); % z_c
                SimLifeCycleKron(5,jj+initialage-1)=currstate(3); % e_c
                
                curr_d=PolicyIndexesKron(1,currstate(1),currstate(2),currstate(3),jj+initialage-1);
                SimLifeCycleKron(1,jj+initialage-1)=curr_d; % d_c
                
                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),currstate(3),jj+initialage-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
                [~,currstate(3)]=max(cumsumpi_e(:,jj)>rand(1,1));
                
                SimLifeCycleKron(2,jj+initialage-1)=currstate(1); % aprime_c
            end
            SimLifeCycleKron(6,:)=seedpoint(4):1:N_j;
        end
    end
elseif fieldexists_pi_z_J==1 && fieldexists_pi_e_J==0
    if N_d==0
        SimLifeCycleKron=nan(4,N_j);
        for jj=1:periods
            SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+initialage-1)=currstate(2); % z_c
            SimLifeCycleKron(3,jj+initialage-1)=currstate(3); % e_c
            
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),currstate(3),jj+initialage-1);
            [~,currstate(2)]=max(cumsumpi_z(currstate(2),:,jj)>rand(1,1));
            [~,currstate(3)]=max(cumsumpi_e>rand(1,1));
        end
        SimLifeCycleKron(4,:)=seedpoint(4):1:N_j;
    else
        if ~exist('include_daprime','var')
            include_daprime=0;
        end
        if include_daprime==0
            SimLifeCycleKron=nan(4,N_j);
            for jj=1:periods
                SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
                SimLifeCycleKron(2,jj+initialage-1)=currstate(2); % z_c
                SimLifeCycleKron(3,jj+initialage-1)=currstate(3); % e_c

                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),currstate(3),jj+initialage-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:,jj)>rand(1,1));            
                [~,currstate(3)]=max(cumsumpi_e>rand(1,1));
            end
            SimLifeCycleKron(4,:)=seedpoint(4):1:N_j;
        else
            SimLifeCycleKron=nan(6,N_j);
            for jj=1:periods
                SimLifeCycleKron(3,jj+initialage-1)=currstate(1); % a
                SimLifeCycleKron(4,jj+initialage-1)=currstate(2); % z
                SimLifeCycleKron(5,jj+initialage-1)=currstate(3); % e
                
                curr_d=PolicyIndexesKron(1,currstate(1),currstate(2),currstate(3),jj+initialage-1);
                SimLifeCycleKron(1,jj+initialage-1)=curr_d; % d

                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),currstate(3),jj+initialage-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:,jj)>rand(1,1));
                [~,currstate(3)]=max(cumsumpi_e>rand(1,1));
 
                SimLifeCycleKron(2,jj+initialage-1)=currstate(1); % aprime
            end   
            SimLifeCycleKron(6,:)=seedpoint(4):1:N_j;
        end
    end
elseif fieldexists_pi_z_J==1 && fieldexists_pi_e_J==1
    if N_d==0
        SimLifeCycleKron=nan(4,N_j);
        for jj=1:periods
            SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+initialage-1)=currstate(2); % z_c
            SimLifeCycleKron(3,jj+initialage-1)=currstate(3); % e_c

            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),currstate(3),jj+initialage-1);
            [~,currstate(2)]=max(cumsumpi_z(currstate(2),:,jj)>rand(1,1));
            [~,currstate(3)]=max(cumsumpi_e(:,jj)>rand(1,1));
        end
        SimLifeCycleKron(4,:)=seedpoint(4):1:N_j;
    else
        if ~exist('include_daprime','var')
            include_daprime=0;
        end
        if include_daprime==0
            SimLifeCycleKron=nan(4,N_j);
            for jj=1:periods
                SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
                SimLifeCycleKron(2,jj+initialage-1)=currstate(2); % z_c
                SimLifeCycleKron(3,jj+initialage-1)=currstate(3); % e_c

                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),currstate(3),jj+initialage-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:,jj)>rand(1,1));
                [~,currstate(3)]=max(cumsumpi_e(:,jj)>rand(1,1));
            end
            SimLifeCycleKron(4,:)=seedpoint(4):1:N_j;
        else
            SimLifeCycleKron=nan(6,N_j);
            for jj=1:periods
                SimLifeCycleKron(3,jj+initialage-1)=currstate(1); % a
                SimLifeCycleKron(4,jj+initialage-1)=currstate(2); % z
                SimLifeCycleKron(5,jj+initialage-1)=currstate(3); % e_c

                curr_d=PolicyIndexesKron(1,currstate(1),currstate(2),currstate(3),jj+initialage-1);
                SimLifeCycleKron(1,jj+initialage-1)=curr_d; % d

                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),currstate(3),jj+initialage-1);
                [~,currstate(2)]=max(cumsumpi_z(currstate(2),:,jj)>rand(1,1));
                [~,currstate(3)]=max(cumsumpi_e(:,jj)>rand(1,1));
                
                SimLifeCycleKron(2,jj+initialage-1)=currstate(1); % aprime
            end     
            SimLifeCycleKron(6,:)=seedpoint(4):1:N_j;
        end
    end
end
