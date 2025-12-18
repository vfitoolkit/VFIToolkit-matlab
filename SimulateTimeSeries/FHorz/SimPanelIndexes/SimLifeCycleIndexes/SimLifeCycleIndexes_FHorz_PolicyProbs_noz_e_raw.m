function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_noz_e_raw(Policy_aprime,CumPolicyProbs,N_j,cumsumpi_e_J,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a,e,j)

initialage=seedpoint(3); % j in (a,e,j)

% Simulation is simoptions.simperiods, or up to 'end of finite horizon'.
periods=min(simoptions.simperiods,N_j-initialage);

SimLifeCycleKron=nan(3,N_j);
for jj=0:periods
    SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
    SimLifeCycleKron(2,jj+initialage)=currstate(2); % e_c

    alowerProbs=CumPolicyProbs(currstate(1),currstate(2),:,jj+initialage);
    [~,probindex]=max(alowerProbs>rand(1));
    currstate(1)=Policy_aprime(currstate(1),currstate(2),probindex,jj+initialage);

    [~,currstate(2)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
end
SimLifeCycleKron(3,initialage:(initialage+periods))=initialage:1:(initialage+periods);


end
