function [output] = getIMFData(database_id, series_id, countrycode2L, frequency, observation_start, observation_end, counterpartycountrycode2L, sector_id, counterpartysector_id, vintage_id)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% database_id: currently supports 'BOP', 'IFS', 'DOT', 'CPIS'
% [IMF datasets Balance of Payments (BOP), International Fiscal Statistics (IFS), 
% Direction of Trade Statistics (DOT), and Coordinated Portfolio Investment Survey (CPIS).]
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
% If you do not input observation_start and observation_end you will be
% given data for all available dates. Equivalently, set
% observation_start=[], and similarly for observation_end=[].
%
% For BOP and IFS no further inputs are required. 
%
% Using CPIS you will need three further inputs:
%   sector_id: see CPIS database for an explanation of Sectors
%   counterpartycountrycode2L: same format as countrycode2L, is the counterpart country
%   counterpartysector_id: same format as sector_id, is the counterpart sector
%
% To access vintage data you need to include a 'vintage_id'. This will be a
% string that contains the year and, where relevant, either quarter or month for the vintage
% you want, e.g., '2017', '2017Q1' or '2017M07'. Note, some databases are just
% annual, some quarterly, and some monthly. 
% 
% I have deliberately made it so that the actual output is similar to that
% from getFredData(). Especially that the dates and data are in output.Data
%
% Some examples of usage can be found on my website, or on my github.
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

if exist('vintage_id', 'var')==1
   database_id_full=[database_id,'_',vintage_id]
end

if nargin==1 % Just return a dictionary for that database
    % /DataStructure/
    JSONdata2 = webread(['http://dataservices.imf.org/REST/SDMX_JSON.svc/DataStructure/',database_id_full]);
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
    % The above first four appear to always be the same, at least for BOP,
    % IFS and CPIS. After the first four things are specific to the database being queried.
    if strcmp(database_id,'IFS')
        temp=JSONdata2.Structure.CodeLists.CodeList{5}.Code;
        for ii=1:length(temp)
            output.TimeFormat{ii,1}=temp(ii).x_value;
            output.TimeFormat{ii,2}=temp(ii).Description.x_text;
        end
    end
    if strcmp(database_id,'CPIS')
        temp=JSONdata2.Structure.CodeLists.CodeList{5}.Code;
        for ii=1:length(temp)
            output.Sector{ii,1}=temp(ii).x_value;
            output.Sector{ii,2}=temp(ii).Description.x_text;
        end
        temp=JSONdata2.Structure.CodeLists.CodeList{6}.Code;
        for ii=1:length(temp)
            output.TimeFormat{ii,1}=temp(ii).x_value;
            output.TimeFormat{ii,2}=temp(ii).Description.x_text;
        end
    end
    if strcmp(database_id,'DOT')
        temp=JSONdata2.Structure.CodeLists.CodeList{5}.Code;
        for ii=1:length(temp)
            output.CouterpartCountry{ii,1}=temp(ii).x_value;
            output.CouterpartCountry{ii,2}=temp(ii).Description.x_text;
        end
        temp=JSONdata2.Structure.CodeLists.CodeList{6}.Code;
        for ii=1:length(temp)
            output.TimeFormat{ii,1}=temp(ii).x_value;
            output.TimeFormat{ii,2}=temp(ii).Description.x_text;
        end
    end
    return
end


% Not just database_id, so an actual data 'series_id' has been requested
if strcmp(database_id,'IFS') || strcmp(database_id, 'BOP')
    optionstring=[frequency,'.',countrycode2L,'.',series_id];
    if nargin==7
        optionstring=[frequency,'.',countrycode2L,'.',series_id,'.',timeformat];
    end
elseif strcmp(database_id,'CPIS')
    optionstring=[frequency,'.',countrycode2L,'.',series_id,'.',sector_id,'.',counterpartysector_id,'.',counterpartycountrycode2L];
elseif strcmp(database_id,'DOT')
    optionstring=[frequency,'.',countrycode2L,'.',series_id,'.',counterpartycountrycode2L];
end

if exist('observation_start','var')==1 && exist('observation_end','var')==1 % if specific start and end dates are given
    if ~isempty(observation_start)
        longoptionstring=[optionstring,'?startPeriod=',observation_start,'&endPeriod=',observation_end];
    end
else
    longoptionstring=optionstring;
end


% /CompactData/
JSONdata = webread(['http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/',database_id_full,'/',longoptionstring]);
output.Data=nan(length(JSONdata.CompactData.DataSet.Series.Obs),2);
temp=JSONdata.CompactData.DataSet.Series.Obs;
% for unknown reason 'temp' is sometimes a cell and sometimes a structure
if iscell(temp)
    for ii=1:length(JSONdata.CompactData.DataSet.Series.Obs)
        output.Data(ii,2)=str2num(JSONdata.CompactData.DataSet.Series.Obs{ii}.x_OBS_VALUE);
        tempstr=JSONdata.CompactData.DataSet.Series.Obs{ii}.x_TIME_PERIOD; % I have deliberately chosen a format for the .Data that is like the output of getFREDdata()
    end
else % otherwise it is a structure
    for ii=1:length(JSONdata.CompactData.DataSet.Series.Obs)
        output.Data(ii,2)=str2num(JSONdata.CompactData.DataSet.Series.Obs(ii).x_OBS_VALUE);
        tempstr=JSONdata.CompactData.DataSet.Series.Obs(ii).x_TIME_PERIOD; % I have deliberately chosen a format for the .Data that is like the output of getFREDdata()
    end
end
for ii=1:length(JSONdata.CompactData.DataSet.Series.Obs)
    if strcmp(frequency,'A')
        output.Data(ii,1)=datenum(str2num(tempstr),1,1);
    elseif strcmp(frequency,'B')
        output.Data(ii,1)=datenum(str2num(tempstr(1:4)),1+6*(str2num(tempstr(7))-1),1);
    elseif strcmp(frequency,'Q')
        output.Data(ii,1)=datenum(str2num(tempstr(1:4)),1+3*(str2num(tempstr(7))-1),1);
    elseif strcmp(frequency,'M')
        output.Data(ii,1)=datenum(str2num(tempstr(1:4)),str2num(tempstr(7)),1);
    else
        fprinf('ERROR: I HAVE NOT CODED getIMFData FOR ANYTHING HIGHER THAN MONTHLY FREQUENCY, IF YOU NEED THIS THEN PLEASE JUST SEND ME AN EMAIL robertdkirkby@gmail.com AND I WILL IMPLEMENT IT')
    end
end
output.IMFcodes.info='These codes are how IMF decribe the data series. (Contents of Obs are just another copy of the data)'; % To do this for more than just the BOP & IFS I really need to just do a read of the field names and then index through them
FullInfoNames=fieldnames(JSONdata.CompactData.DataSet.Series);
nFields=length(FullInfoNames);
for ii=1:nFields
    output.IMFcodes.(FullInfoNames{ii})=JSONdata.CompactData.DataSet.Series.(FullInfoNames{ii});
end



% %GenericMetadata/
% For some reason known only to IMF, it appears you can only request
% metadata for the 'annual frequency' code. So next line just imposes this
% on the metadata request (the frequency is not part of the metadata anyway).
longoptionstring(1)='A';
% Now request the metadata
JSONdata3 = webread(['http://dataservices.imf.org/REST/SDMX_JSON.svc/GenericMetadata/',database_id_full,'/',longoptionstring]);
output.country=JSONdata3.GenericMetadata.MetadataSet.AttributeValueSet(2).ReportedAttribute(1).ReportedAttribute(1).Value.x_text;
output.series_id=JSONdata3.GenericMetadata.MetadataSet.AttributeValueSet(3).ReportedAttribute(1).ReportedAttribute(1).Value.x_text;
output.description=JSONdata3.GenericMetadata.MetadataSet.AttributeValueSet(3).ReportedAttribute(2).ReportedAttribute(1).Value.x_text;
if strcmp(database_id,'CPIS')
    output.sector=JSONdata3.GenericMetadata.MetadataSet.AttributeValueSet(4).ReportedAttribute(2).ReportedAttribute(1).Value.x_text;
    output.counterpartsector=JSONdata3.GenericMetadata.MetadataSet.AttributeValueSet(5).ReportedAttribute(2).ReportedAttribute(1).Value.x_text;
    output.counterpartcountry=JSONdata3.GenericMetadata.MetadataSet.AttributeValueSet(6).ReportedAttribute(2).ReportedAttribute(1).Value.x_text;
end
if strcmp(database_id,'DOT')
    output.counterpartcountry=JSONdata3.GenericMetadata.MetadataSet.AttributeValueSet(4).ReportedAttribute(2).ReportedAttribute(1).Value.x_text;
end

end
