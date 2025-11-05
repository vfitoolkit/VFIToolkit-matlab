function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_e_raw(PolicyKron,N_semiz,N_j,cumsumpi_semiz_J,cumsumpi_e_J,seedpoint,simperiods)
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

% seedpoint is (a,semiz,e,j)

initialage=seedpoint(4); % j in (a,semiz,e,j)

% Simulation is simperiods, or up to 'end of finite horizon'.
periods=min(simperiods,N_j-initialage);

% If N_d1>0, you should eliminate d1 from Policy so that instead of
% (d,aprime) only contains (d2,aprime), where d2 is the decision variable
% that is for the semi-exo state.

SimLifeCycleKron=nan(4,N_j);
for jj=0:periods
    SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
    SimLifeCycleKron(2,jj+initialage)=currstate(2); % semiz_c
    SimLifeCycleKron(3,jj+initialage)=currstate(3); % e_c

    temp=PolicyKron(:,currstate(1),currstate(2)+N_semiz*(currstate(3)-1),jj+initialage);
    currstate(1)=temp(2);

    [~,currstate(2)]=max(cumsumpi_semiz_J(currstate(2),:,temp(1),jj+initialage)>rand(1,1));
    [~,currstate(3)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
end
SimLifeCycleKron(4,initialage:(initialage+periods))=initialage:1:(initialage+periods);

end