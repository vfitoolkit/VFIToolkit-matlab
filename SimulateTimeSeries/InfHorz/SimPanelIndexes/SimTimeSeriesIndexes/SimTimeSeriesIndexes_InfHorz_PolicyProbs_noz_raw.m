function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_PolicyProbs_noz_raw(Policy_aprime,CumPolicyProbs,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a)

SimTimeSeriesKron=nan(1,simoptoins.simperiods);
for tt=1:simoptions.simperiods
    SimTimeSeriesKron(1,tt)=currstate(1); % a_c

    alowerProbs=CumPolicyProbs(currstate(1),:);
    [~,probindex]=max(alowerProbs>rand(1));
    currstate(1)=Policy_aprime(currstate(1),probindex);
end

end
