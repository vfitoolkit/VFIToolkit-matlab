function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_semiz_noz_raw(PolicyKron,Policy_dsemiexo,cumsumpi_semiz,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a,semiz,j)

SimTimeSeriesKron=nan(2,simoptions.simperiods);
for tt=1:simoptions.simperiods
    SimTimeSeriesKron(1,tt)=currstate(1); % a_c
    SimTimeSeriesKron(2,tt)=currstate(2); % semiz_c

    d2ind=Policy_dsemiexo(currstate(1),currstate(2));
    currstate(1)=PolicyKron(currstate(1),currstate(2)); % (d2,aprime)
    
    [~,currstate(2)]=max(cumsumpi_semiz(currstate(2),:,d2ind)>rand(1,1));
end

end