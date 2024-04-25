function [DateVector, DataMatrix, CountryCodes]=getOECDFredData(code1, code2, justcountries, observation_start, observation_end, units, frequency, aggregation_method)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Will get data for all of the OECD countries (including Eurozone, EU,
% all OECD, and G7) for the data series corresponding to code1 and
% code2. I returns this as a big matrix.
%
% inputs: code1 and code2 tell it which data series to get.
%         if justcountries==1 then it only reports the countries (no EU, G7, or Luxembourg etc.).
%         if justcountries==2 then also drop Switzerland too.
%         all other inputs are same as for 'getFredData.m'
%
% 
% It simply calls 'getFredData.m' for each of the OECD member countries. IF
% you do not have getFredData.m you can get it from robertdkirkby.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Running GetOECDDataFromFRED()')

%f = fred('http://research.stlouisfed.org/fred2/')

CountryCodes=['AT'; 'AU'; 'BE'; 'CA'; 'CZ'; 'DK'; 'EE'; 'FI'; 'FR'; 'DE'; 'GR'; 'HU'; 'IS'; 'IE'; 'IL'; 'IT'; 'JP'; 'KR'; 'LU'; 'MX'; 'NL'; 'NZ'; 'NO'; 'PL'; 'PT'; 'SK'; 'SI'; 'ES'; 'SE'; 'CH'; 'TR'; 'GB'; 'US'; 'EZ'; 'EU'; 'O1'; 'G7'];
% Austria, Australia, Belgium, Canada, Czech Republic,
% Denmark, Estonia, Finland, France, Denmark,
% Greece, Hungary, Iceland, Ireland, Israel,
% Italy, Japan, South Korea, Luxembourg, Mexico,
% Netherlands, New Zealand, Norway, Poland, Portugal,
% Slovakia, Slovenia, Spain, Sweden, Switzerland
% Turkey, United Kingdom, United States, Eurozone, European Union
% All OECD, G7

if justcountries==1
    % Gets rid of Luxembourg, Eurozone, European Union, All OECD, and G7
    CountryCodes=['AT'; 'AU'; 'BE'; 'CA'; 'CZ'; 'DK'; 'EE'; 'FI'; 'FR'; 'DE'; 'GR'; 'HU'; 'IS'; 'IE'; 'IL'; 'IT'; 'JP'; 'KR'; 'MX'; 'NL'; 'NZ'; 'NO'; 'PL'; 'PT'; 'SK'; 'SI'; 'ES'; 'SE'; 'CH'; 'TR'; 'GB'; 'US'];
elseif justcountries==2
    % Drop Switzerland too.
    CountryCodes=['AT'; 'AU'; 'BE'; 'CA'; 'CZ'; 'DK'; 'EE'; 'FI'; 'FR'; 'DE'; 'GR'; 'HU'; 'IS'; 'IE'; 'IL'; 'IT'; 'JP'; 'KR'; 'MX'; 'NL'; 'NZ'; 'NO'; 'PL'; 'PT'; 'SK'; 'SI'; 'ES'; 'SE'; 'TR'; 'GB'; 'US'];
end

tempseries_id = [code1, 'US', code2];
if exist('units','var') && exist('frequency','var') && exist('aggregation_method','var') 
    tempfred=getFredData(tempseries_id, observation_start, observation_end, units, frequency, aggregation_method) %, ondate, realtime_end)
elseif exist('units','var') && exist('frequency','var')
    tempfred=getFredData(tempseries_id, observation_start, observation_end, units, frequency)
elseif exist('units','var')
    tempfred=getFredData(tempseries_id, observation_start, observation_end,units)
else
    tempfred=getFredData(tempseries_id, observation_start, observation_end)
end
DateVector=tempfred.Dates;

DataMatrix=zeros(length(DateVector), length(CountryCodes));
for ii=1:length(CountryCodes)
    currentcode=[code1, CountryCodes(ii,:), code2];
    currentfred = getFredData(currentcode, observation_start, observation_end)
    DataMatrix(:,ii)=currentfred.Data;
end

CountryCodes=CountryCodes'; %For purpose of output

end
