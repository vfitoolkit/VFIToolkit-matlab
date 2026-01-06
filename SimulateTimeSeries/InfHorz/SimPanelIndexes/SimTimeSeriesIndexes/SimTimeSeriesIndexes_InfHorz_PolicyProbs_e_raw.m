function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_PolicyProbs_e_raw(Policy_aprime,CumPolicyProbs,cumsumpi_z,cumsumpi_e,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a,z,e)

SimTimeSeriesKron=nan(3,simoptions.simperiods);
for tt=1:simoptions.simperiods
    SimTimeSeriesKron(1,tt)=currstate(1); % a_c
    SimTimeSeriesKron(2,tt)=currstate(2); % z_c
    SimTimeSeriesKron(3,tt)=currstate(3); % e_c

    alowerProbs=CumPolicyProbs(currstate(1),currstate(2),currstate(3),:);
    [~,probindex]=max(alowerProbs>rand(1));
    currstate(1)=Policy_aprime(currstate(1),currstate(2),currstate(3),probindex);

    [~,currstate(2)]=max(cumsumpi_z(currstate(2),:)>rand(1,1));
    [~,currstate(3)]=max(cumsumpi_e>rand(1,1));
end

end
