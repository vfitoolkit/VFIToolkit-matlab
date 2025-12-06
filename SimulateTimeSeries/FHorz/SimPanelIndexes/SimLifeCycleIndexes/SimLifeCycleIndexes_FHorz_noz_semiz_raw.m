function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_noz_semiz_raw(PolicyKron,N_j,cumsumpi_semiz_J,simoptions,seedpoint)
% All inputs must be on the CPU
%
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist' in eqm)
%
% Outputs the indexes for (a,semiz,j) for every period j. This is for period 1 to J. 
% Since most simulations will not start at period 1, the first entries are typically 'NaN'.

currstate=seedpoint;

% seedpoint is (a,semiz,j)

initialage=seedpoint(3); % j in (a,semiz,j)

% Simulation is simoptions.simperiods, or up to 'end of finite horizon'.
periods=min(simoptions.simperiods,N_j-initialage);

% If N_d1>0, you should eliminate d1 from Policy so that instead of
% (d,aprime) only contains (d2,aprime), where d2 is the decision variable
% that is for the semi-exo state.

SimLifeCycleKron=nan(3,N_j);
if simoptions.gridinterplayer==0
    for jj=0:periods
        SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
        SimLifeCycleKron(2,jj+initialage)=currstate(2); % semiz_c

        temp=PolicyKron(:,currstate(1),currstate(2),jj+initialage); % (d2,aprime)
        currstate(1)=temp(2);

        [~,currstate(2)]=max(cumsumpi_semiz_J(currstate(2),:,temp(1),jj+initialage)>rand(1,1));
    end
elseif simoptions.gridinterplayer==1
    for jj=0:periods
        SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
        SimLifeCycleKron(2,jj+initialage)=currstate(2); % semiz_c

        temp=PolicyKron(:,currstate(1),currstate(2),jj+initialage); % (d2,aprime)
        loweraprimdind=temp(2);
        aprimeprob=(temp(3)-1)/(1+simoptions.ngridinterp);
        currstate(1)=loweraprimdind+(rand(1)<aprimeprob);

        [~,currstate(2)]=max(cumsumpi_semiz_J(currstate(2),:,temp(1),jj+initialage)>rand(1,1));
    end
end
SimLifeCycleKron(3,initialage:(initialage+periods))=initialage:1:(initialage+periods);

end