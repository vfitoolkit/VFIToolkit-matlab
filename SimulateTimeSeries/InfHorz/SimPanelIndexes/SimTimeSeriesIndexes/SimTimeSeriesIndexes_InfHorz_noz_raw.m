function SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_noz_raw(Policy_aprime,simoptions,seedpoint)
% All inputs must be on the CPU

currstate=seedpoint; % seedpoint is (a)

SimTimeSeriesKron=nan(1,simoptions.simperiods);
for tt=1:simoptions.simperiods
    SimTimeSeriesKron(1,tt)=currstate(1); % a_c

    currstate(1)=Policy_aprime(currstate(1));
end

end