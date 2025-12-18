function SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_noz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_semiz_J,cumsumpi_e_J,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a,semiz,e,j)

initialage=seedpoint(4); % j in (a,semiz,e,j)

% Simulation is simoptions.simperiods, or up to 'end of finite horizon'.
periods=min(simoptions.simperiods,N_j-initialage);

SimLifeCycleKron=nan(4,N_j);
for jj=0:periods
    SimLifeCycleKron(1,jj+initialage)=currstate(1); % a_c
    SimLifeCycleKron(2,jj+initialage)=currstate(2); % semiz_c
    SimLifeCycleKron(3,jj+initialage)=currstate(3); % e

    alowerProbs=CumPolicyProbs(currstate(1),currstate(2),currstate(3),:,jj+initialage);
    [~,probindex]=max(alowerProbs>rand(1));
    d2ind=Policy_dsemiexo(currstate(1),currstate(2),currstate(3),jj+initialage);
    currstate(1)=Policy_aprime(currstate(1),currstate(2),currstate(3),probindex,jj+initialage);

    [~,currstate(2)]=max(cumsumpi_semiz_J(currstate(2),:,d2ind,jj+initialage)>rand(1,1));
    [~,currstate(3)]=max(cumsumpi_e_J(:,jj+initialage)>rand(1,1));
end
SimLifeCycleKron(4,initialage:(initialage+periods))=initialage:1:(initialage+periods);



end
