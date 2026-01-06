function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_noz_e_raw(Policy_aprime,cumsumpi_e,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a,e)

SimTimeSeriesKron=nan(2,simoptions.simperiods);
for tt=1:simoptions.simperiods
    SimTimeSeriesKron(1,tt)=currstate(1); % a_c
    SimTimeSeriesKron(2,tt)=currstate(2); % e_c

    currstate(1)=Policy_aprime(currstate(1),currstate(2));
    
    [~,currstate(2)]=max(cumsumpi_e>rand(1,1));
end

end
