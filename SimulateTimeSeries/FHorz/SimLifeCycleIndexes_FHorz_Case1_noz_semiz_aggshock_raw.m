function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_aggshock_raw(PolicyIndexesKron,AggShockIndex,N_d1,N_semiz_idio,N_j,cumsumpi_semiz_idio_J,seedpoint,simperiods)
% All inputs must be on the CPU
%
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm)
%
% Outputs the indexes for (a,semiz,j) for every period j. This is for period 1 to J. 
% Since most simulations will not start at period 1, the first entries are typically 'NaN'.

% seedpoint=simoptions.seedpoint;
% simperiods=simoptions.simperiods;

currstate=seedpoint;

% seedpoint is (a,semiz,j)

initialage=seedpoint(3); % j in (a,semiz,j)

% Simulation is simperiods, or up to 'end of finite horizon'.
periods=min(simperiods,N_j-initialage);

aggz_c=AggShockIndex(initialage);

if N_d1==0
    SimLifeCycleKron=nan(3,N_j);
    for jj=0:periods-1
        SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
        SimLifeCycleKron(2,jj+initialage)=currstate(2)+N_semiz_idio*(aggz_c-1); % semiz_c
        % Note: We want the same structure as we would have for semiz variables, hence put aggregate back into semiz

        temp=PolicyIndexesKron(:,currstate(1),currstate(2),jj+initialage); % (d2,aprime)
        currstate(1)=temp(2);

        aggzprime_c=AggShockIndex(initialage+jj+1);
        [~,currstate(2)]=max(cumsumpi_semiz_idio_J(currstate(2),:,aggz_c,aggzprime_c,temp(1),jj+initialage)>rand(1,1));
        aggz_c=aggzprime_c;

    end
    jj=periods; % Do the final period
    SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
    SimLifeCycleKron(2,jj+initialage)=currstate(2)+N_semiz_idio*(aggz_c-1); % semiz_c

    SimLifeCycleKron(3,initialage:(initialage+periods))=initialage:1:(initialage+periods);
else
    SimLifeCycleKron=nan(3,N_j);
    for jj=0:periods-1
        SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
        SimLifeCycleKron(2,jj+initialage-1)=currstate(2)+N_semiz_idio*(aggz_c-1); % semiz_c
        % Note: We want the same structure as we would have for semiz variables, hence put aggregate back into semiz

        temp=PolicyIndexesKron(:,currstate(1),currstate(2),jj+initialage); % (d1,d2,aprime)
        currstate(1)=temp(3);

        aggzprime_c=AggShockIndex(initialage+jj+1);
        [~,currstate(2)]=max(cumsumpi_semiz_idio_J(currstate(2),:,aggz_c,aggzprime_c,temp(2),jj+initialage)>rand(1,1));
        aggz_c=aggzprime_c;
    end
    jj=periods; % Do the final period
    SimLifeCycleKron(1,jj+initialage-1)=currstate(1); % a_c
    SimLifeCycleKron(2,jj+initialage-1)=currstate(2)+N_semiz_idio*(aggz_c-1); % semiz_c

    SimLifeCycleKron(3,initialage:(initialage+periods))=initialage:1:(initialage+periods);
end
