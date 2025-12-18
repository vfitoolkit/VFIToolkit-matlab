function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_noz_raw(Policy_aprime,CumPolicyProbs,N_j,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a,j)

initialage=seedpoint(2); % j in (a,j)

% Simulation is simoptions.simperiods, or up to 'end of finite horizon'.
periods=min(simoptions.simperiods,N_j-initialage);

SimLifeCycleKron=nan(2,N_j);
for jj=0:periods
    SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c

    alowerProbs=CumPolicyProbs(currstate(1),:,jj+initialage);
    [~,probindex]=max(alowerProbs>rand(1));
    currstate(1)=Policy_aprime(currstate(1),probindex,jj+initialage);
end
SimLifeCycleKron(2,initialage:(initialage+periods))=initialage:1:(initialage+periods);


end
