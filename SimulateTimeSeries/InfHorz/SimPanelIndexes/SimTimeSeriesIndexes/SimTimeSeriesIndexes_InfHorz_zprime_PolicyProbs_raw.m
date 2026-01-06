function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_zprime_PolicyProbs_raw(Policy_aprime,CumPolicyProbs,cumsumpi_z,simoptions,seedpoint)
% All inputs must be on the CPU

% zprime: so
%   Policy_aprime and CumPolicyProbs
% are of size [N_a,N_z,N_zprime,N_probs]

currstate=seedpoint; % seedpoint is (a,z)

SimTimeSeriesKron=nan(2,simoptions.simperiods);
for tt=1:simoptions.simperiods
    SimTimeSeriesKron(1,tt)=currstate(1); % a_c
    SimTimeSeriesKron(2,tt)=currstate(2); % z_c
    
    [~,zprime_c]=max(cumsumpi_z(currstate(2),:)>rand(1,1)); % zprime

    alowerProbs=CumPolicyProbs(currstate(1),currstate(2),zprime_c,:);
    [~,probindex]=max(alowerProbs>rand(1));
    currstate(1)=Policy_aprime(currstate(1),currstate(2),zprime_c,probindex);

    currstate(2)=zprime_c;
end

end
