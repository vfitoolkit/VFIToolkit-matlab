function [output] = getIMFData(database_id, series_id, countrycode2L, frequency, observation_start, observation_end)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% database_id: currently supports 'BOP', 'IFS', 'DOT'
% [IMF datasets Balance of Payments (BOP), International Fiscal Statistics (IFS), and Direction of Trade Statistics (DOT)]
%
% If the only input is the database_id, then output will return a
% dictionary of 'series_id' and their names. This can then be searched to
% find the series_id for a variable when you only have some idea what the
% name might be.
%
% series_id: the code for the variable you want
%
% frequency: for example monthly M, quarterly Q, or annually A;
% 
% I have deliberately made it so that the actual output is similar to that
% from getFredData(). Especially that the dates and data are in output.Data
%
% 2019
% Robert Kirkby
% robertdkirkby.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     

%%%%%
%
% Format of the request sent to the api:
%   BOP: e.g., 'NZ. ' New Zealand 
%   IFS: e.g., '
%   DOT: e.g., 'IT. .FR'; Italy-France, 
%
% Connects to IMF database and retrieves the data series identified by series_id. 
% See https://www.bd-econ.com/imfapi1.html
% 
% The IMF's CompactData method, combined with codes for the series, frequency, area, and indicator, returns a JSON structured dataset. The codes and method are explained in more detail as follows:
%
% Area: The country, region, or set of countries, for example GB for the U.K., or GB+US for the U.K. and the U.S.;
% Indicator: The code for the indicator of interest--IFS includes more than 2,500. In the example above, the code of interest is PMP_IX; and
% Date Range (Optional): Use this to limit the data range returned, for example ?startPeriod=2010&endPeriod=2017 otherwise the full set of data is returned.
%
% http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/DS-BOP/A.NZ.?startPeriod={start date}&endPeriod={end date}
% http://dataservices.imf.org/REST/SDMX_JSON.svc/CodeList/{dimension_list[2]}
%
% Method: CompactData retrieves data, DataStructure retrieves series information, and GenericMetadata returns the metadata;
%
% Three websites I found useful to figure out the 'codes' for making a request
% http://datahelp.imf.org/knowledgebase/articles/630877-api
% http://datahelp.imf.org/knowledgebase/articles/667681-using-json-restful-web-service
% http://dataservices.imf.org/REST/SDMX_JSON.svc/help
%
% For Balance of Payments data, the database ID is 'BOP' (database 81, see full list of databases at
% http://dataservices.imf.org/REST/SDMX_JSON.svc/Dataflow 
% Use regular IMF website to find the name of the database, and then use above link to find the relevant database ID code.
%
%%%%%

if nargin==1 % Just return a dictionary for that database
    % /DataStructure/
    JSONdata2 = webread(['http://dataservices.imf.org/REST/SDMX_JSON.svc/DataStructure/',database_id]);
    temp=JSONdata2.Structure.CodeLists.CodeList{1}.Code;
    for ii=1:length(temp)
        output.Scale{ii,1}=temp(ii).x_value;
        output.Scale{ii,2}=temp(ii).Description.x_text;
    end
    temp=JSONdata2.Structure.CodeLists.CodeList{2}.Code;
    for ii=1:length(temp)
        output.Frequency{ii,1}=temp(ii).x_value;
        output.Frequency{ii,2}=temp(ii).Description.x_text;
    end
    temp=JSONdata2.Structure.CodeLists.CodeList{3}.Code;
    for ii=1:length(temp)
        output.CountryCodes{ii,1}=temp(ii).x_value;
        output.CountryCodes{ii,2}=temp(ii).Description.x_text;
    end
    temp=JSONdata2.Structure.CodeLists.CodeList{4}.Code;
    for ii=1:length(temp)
        output.Variables{ii,1}=temp(ii).x_value;
        output.Variables{ii,2}=temp(ii).Description.x_text;
    end
%     temp=JSONdata2.Structure.CodeLists.CodeList{5}.Code;
%     for ii=1:length(temp) % No idea why, but this just seems to be a second copy of Frequency.
%         output.IMFcodes.Frequency2{ii,1}=temp(ii).x_value;
%         output.IMFcodes.Frequency2{ii,2}=temp(ii).Description.x_text;
%     end
    return
end


% Not just database_id, so an actual data 'series_id' has been requested

optionstring=[frequency,'.',countrycode2L,'.',series_id];

if nargin>4 % if specific start and end dates are given
    if ~isempty(observation_start)
        longoptionstring=[optionstring,'?startPeriod=',observation_start,'&endPeriod=',observation_end];
    end
end

% /CompactData/
JSONdata = webread(['http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/',database_id,'/',longoptionstring]);
output.Data=nan(length(JSONdata.CompactData.DataSet.Series.Obs),2);
temp=JSONdata.CompactData.DataSet.Series.Obs;
% for unknown reason 'temp' is sometimes a cell and sometimes a structure
if iscell(temp)
    for ii=1:length(JSONdata.CompactData.DataSet.Series.Obs)
        output.Data(ii,1)=str2num(JSONdata.CompactData.DataSet.Series.Obs{ii}.x_TIME_PERIOD); % I have deliberately chosen a format for the .Data that is like the output of getFREDdata()
        output.Data(ii,2)=str2num(JSONdata.CompactData.DataSet.Series.Obs{ii}.x_OBS_VALUE);
    end
else % otherwise it is a structure
    for ii=1:length(JSONdata.CompactData.DataSet.Series.Obs)
        output.Data(ii,1)=str2num(JSONdata.CompactData.DataSet.Series.Obs(ii).x_TIME_PERIOD); % I have deliberately chosen a format for the .Data that is like the output of getFREDdata()
        output.Data(ii,2)=str2num(JSONdata.CompactData.DataSet.Series.Obs(ii).x_OBS_VALUE);
    end
end
output.IMFcodes.info='These codes are how IMF decribe the data series'; % To do this for more than just the BOP & IFS I really need to just do a read of the field names and then index through them
output.IMFcodes.x_FREQ=JSONdata.CompactData.DataSet.Series.x_FREQ;
output.IMFcodes.x_REF_AREA=JSONdata.CompactData.DataSet.Series.x_REF_AREA;
output.IMFcodes.x_INDICATOR=JSONdata.CompactData.DataSet.Series.x_INDICATOR;
output.IMFcodes.x_UNIT_MULT=JSONdata.CompactData.DataSet.Series.x_UNIT_MULT;
output.IMFcodes.x_TIME_FORMAT=JSONdata.CompactData.DataSet.Series.x_TIME_FORMAT;

% %GenericMetadata/
JSONdata3 = webread(['http://dataservices.imf.org/REST/SDMX_JSON.svc/GenericMetadata/',database_id,'/',longoptionstring]);
output.country=JSONdata3.GenericMetadata.MetadataSet.AttributeValueSet(2).ReportedAttribute(1).ReportedAttribute(1).Value.x_text;
output.series_id=JSONdata3.GenericMetadata.MetadataSet.AttributeValueSet(3).ReportedAttribute(1).ReportedAttribute(1).Value.x_text;
output.description=JSONdata3.GenericMetadata.MetadataSet.AttributeValueSet(3).ReportedAttribute(2).ReportedAttribute(1).Value.x_text;

end
