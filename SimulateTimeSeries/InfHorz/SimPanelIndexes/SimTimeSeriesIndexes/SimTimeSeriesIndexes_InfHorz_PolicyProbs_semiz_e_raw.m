function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_PolicyProbs_semiz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,cumsumpi_z,cumsumpi_semiz,cumsumpi_e,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a,semiz,z,e)

SimTimeSeriesKron=nan(4,simoptions.simperiods);
for tt=1:simoptions.simperiods
    SimTimeSeriesKron(1,tt)=currstate(1); % a_c
    SimTimeSeriesKron(2,tt)=currstate(2); % semiz_c
    SimTimeSeriesKron(3,tt)=currstate(3); % z_c
    SimTimeSeriesKron(4,tt)=currstate(4); % e

    alowerProbs=CumPolicyProbs(currstate(1),currstate(2),currstate(3),currstate(4),:);
    [~,probindex]=max(alowerProbs>rand(1));
    d2ind=Policy_dsemiexo(currstate(1),currstate(2),currstate(3),currstate(4));
    currstate(1)=Policy_aprime(currstate(1),currstate(2),currstate(3),currstate(4),probindex);

    [~,currstate(2)]=max(cumsumpi_semiz(currstate(2),:,d2ind)>rand(1,1));
    [~,currstate(3)]=max(cumsumpi_z(currstate(3),:)>rand(1,1));
    [~,currstate(4)]=max(cumsumpi_e>rand(1,1));
end

end
