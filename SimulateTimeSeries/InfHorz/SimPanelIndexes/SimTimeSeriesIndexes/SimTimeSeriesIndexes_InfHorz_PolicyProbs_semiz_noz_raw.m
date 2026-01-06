function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_PolicyProbs_semiz_noz_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,cumsumpi_semiz,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a,semiz)

SimTimeSeriesKron=nan(2,simoptions.simperiods);
for tt=1:simoptions.simperiods
    SimTimeSeriesKron(1,tt)=currstate(1); % a_c
    SimTimeSeriesKron(2,tt)=currstate(2); % semiz_c

    alowerProbs=CumPolicyProbs(currstate(1),currstate(2),:);
    [~,probindex]=max(alowerProbs>rand(1));
    d2ind=Policy_dsemiexo(currstate(1),currstate(2));
    currstate(1)=Policy_aprime(currstate(1),currstate(2),probindex);

    [~,currstate(2)]=max(cumsumpi_semiz(currstate(2),:,d2ind)>rand(1,1));
end

end