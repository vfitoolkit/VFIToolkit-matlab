function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_semiz_e_raw(PolicyKron,N_semiz,N_z,N_j,cumsumpi_z_J,cumsumpi_semiz_J,cumsumpi_e_J,simoptions,seedpoint)
% All inputs must be on the CPU
%
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm)
%
% Outputs the indexes for (a,z,semiz,e,j) for every period j. This is for period 1 to J. 
% Since most simulations will not start at period 1, the first entries are typically 'NaN'.

currstate=seedpoint;

% seedpoint is (a,z,semiz,e,j)

initialage=seedpoint(5); % j in (a,z,semiz,e,j)

% Simulation is simoptions.simperiods, or up to 'end of finite horizon'.
periods=min(simoptions.simperiods,N_j-initialage);

% If N_d1>0, you should eliminate d1 from Policy so that instead of
% (d,aprime) only contains (d2,aprime), where d2 is the decision variable
% that is for the semi-exo state.

SimLifeCycleKron=nan(5,N_j);
if simoptions.gridinterplayer==0
    for jj=0:periods
        SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
        SimLifeCycleKron(2,jj+initialage)=currstate(2); % semiz_c
        SimLifeCycleKron(3,jj+initialage)=currstate(3); % z_c
        SimLifeCycleKron(4,jj+initialage)=currstate(4); % e

        temp=PolicyKron(:,currstate(1),currstate(2)+N_semiz*(currstate(3)-1)+N_semiz*N_z*(currstate(4)-1),jj+initialage);
        currstate(1)=temp(2);

        [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj+initialage)>rand(1,1));
        [~,currstate(3)]=max(cumsumpi_semiz_J(currstate(3),:,temp(1),jj+initialage)>rand(1,1));
        [~,currstate(4)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
    end
elseif simoptions.gridinterplayer==1
    for jj=0:periods
        SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
        SimLifeCycleKron(2,jj+initialage)=currstate(2); % semiz_c
        SimLifeCycleKron(3,jj+initialage)=currstate(3); % z_c
        SimLifeCycleKron(4,jj+initialage)=currstate(4); % e

        temp=PolicyKron(:,currstate(1),currstate(2)+N_semiz*(currstate(3)-1)+N_semiz*N_z*(currstate(4)-1),jj+initialage);
        loweraprimdind=temp(2);
        aprimeprob=(temp(3)-1)/(1+simoptions.ngridinterp);
        currstate(1)=loweraprimdind+(rand(1)<aprimeprob);

        [~,currstate(2)]=max(cumsumpi_z_J(currstate(2),:,jj+initialage)>rand(1,1));
        [~,currstate(3)]=max(cumsumpi_semiz_J(currstate(3),:,temp(1),jj+initialage)>rand(1,1));
        [~,currstate(4)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
    end
end
SimLifeCycleKron(5,initialage:(initialage+periods))=initialage:1:(initialage+periods);



end