function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_PolicyProbs_raw(Policy_aprime,CumPolicyProbs,cumsumpi_z,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a,z)

SimTimeSeriesKron=nan(2,simoptions.simperiods);
for tt=1:simoptions.simperiods
    SimTimeSeriesKron(1,tt)=currstate(1); % a_c
    SimTimeSeriesKron(2,tt)=currstate(2); % z_c

    alowerProbs=CumPolicyProbs(currstate(1),currstate(2),:);
    [~,probindex]=max(alowerProbs>rand(1));
    currstate(1)=Policy_aprime(currstate(1),currstate(2),probindex);

    [~,currstate(2)]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
end

end
