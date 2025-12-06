function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_e_raw(PolicyIndexesKron,N_d,N_j,cumsumpi_z_J,cumsumpi_e_J,simoptions,seedpoint)
% All inputs must be on the CPU
%
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm)
%
% Outputs the indexes for (a,z,e) for every period j. This is for period 1 to J. 
% Since most simulations will not start at period 1, the first entries are typically 'NaN'.

currstate=seedpoint;

% seedpoint is (a,z,e,j)

initialage=seedpoint(4); % j in (a,z,e,j)

% Simulation is simoptions.simperiods, or up to 'end of finite horizon'.
periods=min(simoptions.simperiods,N_j-initialage);

if simoptions.gridinterplayer==0
    if N_d==0
        SimLifeCycleKron=nan(4,N_j);
        for jj=0:periods
            SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+initialage)=currstate(2); % z_c
            SimLifeCycleKron(3,jj+initialage)=currstate(3); % e_c

            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),currstate(3),jj+initialage);
            [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj+initialage)>rand(1,1));
            [~,currstate(3)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
        end
        SimLifeCycleKron(4,initialage:(initialage+periods))=initialage:1:(initialage+periods);
    else
        SimLifeCycleKron=nan(4,N_j);
        for jj=0:periods
            SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+initialage)=currstate(2); % z_c
            SimLifeCycleKron(3,jj+initialage)=currstate(3); % e_c

            currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2),currstate(3),jj+initialage);
            [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj+initialage)>rand(1,1));
            [~,currstate(3)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
        end
        SimLifeCycleKron(4,initialage:(initialage+periods))=initialage:1:(initialage+periods);
    end
elseif simoptions.gridinterplayer==1
    if N_d==0
        SimLifeCycleKron=nan(4,N_j);
        for jj=0:periods
            SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+initialage)=currstate(2); % z_c
            SimLifeCycleKron(3,jj+initialage)=currstate(3); % e_c

            temp=PolicyIndexesKron(:,currstate(1),currstate(2),currstate(3),jj+initialage);
            loweraprimdind=temp(1);
            aprimeprob=(temp(2)-1)/(1+simoptions.ngridinterp);
            currstate(1)=loweraprimdind+(rand(1)<aprimeprob);

            [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj+initialage)>rand(1,1));
            [~,currstate(3)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
        end
        SimLifeCycleKron(4,initialage:(initialage+periods))=initialage:1:(initialage+periods);
    else
        SimLifeCycleKron=nan(4,N_j);
        for jj=0:periods
            SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
            SimLifeCycleKron(2,jj+initialage)=currstate(2); % z_c
            SimLifeCycleKron(3,jj+initialage)=currstate(3); % e_c

            temp=PolicyIndexesKron(:,currstate(1),currstate(2),currstate(3),jj+initialage);
            loweraprimdind=temp(2);
            aprimeprob=(temp(3)-1)/(1+simoptions.ngridinterp);
            currstate(1)=loweraprimdind+(rand(1)<aprimeprob);

            [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj+initialage)>rand(1,1));
            [~,currstate(3)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
        end
        SimLifeCycleKron(4,initialage:(initialage+periods))=initialage:1:(initialage+periods);
    end
end




end
