function [output] = getFredData(series_id, observation_start, observation_end, units, frequency, aggregation_method, ondate, realtime_end)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Connects to FRED database and retrieves the data series identified by series_id. 
%
% Examples based on getting M1 money (FRED: M1SL): 
%       M1 = getFredData(‘M1SL’, ‘1960-01-01′, ‘2000-12-31′);
%       M1 = getFredData(‘M1SL’, ‘1960-01-01′, ‘2000-12-31′,’pca’,’q’);
% More at: robertdkirkby.com/academic/matlab-tips/
%
% All dates should be formatted as YYYY-MM-DD
%
% observation_start: Observations start (as soon as possible) from this date
% observation_end:   Observations end (as close as possible) at this date
% units:             Choose levels, percent change on year ago, etc.
% frequency:         If you want lower frequency than raw data
% aggregation_method:If you change frequency from raw data, this sets the aggreation method (default is mean)
% ondate:            If you want only the data that was available at a
%                    certain date (uses ALFRED).
% realtime_end:      If you want to use ALFRED's realtime_start and
%                    realtime_end options. If you give realtime_end then
%                    the entry for ondate is used as realtime_start. Note
%                    however that I do not really parse the returned data
%                    properly if you use this option.
%
% The formats and effect of all the input parameters is exactly as described in FRED
% API: http://api.stlouisfed.org/docs/fred/series_observations.html
% Only exception is ondate which works as FRED would if you set both realtime_start
% and realtime_end both equal to this same date.
%
% Unwanted inputs can be left empty, [].
%
% Returns the same format as you normally get accessing FRED using Matlab's 'fetch': 
% Namely a structure with the following,
%   'Title'
%   'SeriesID'
%   'Source'
%   'Release'
%   'SeasonalAdjustment'
%   'Frequency'
%   'Units'
%   'DateRange'
%   'LastUpdated'
%   'Notes'             
%   'Data'
%
%
% Formats for inputs: https://api.stlouisfed.org/docs/fred/series_observations.html
%   units:  'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca', 'log' 
%   frequency: 'd', 'w', 'bw', 'm', 'q', 'sa', 'a', 'wef', 'weth', 'wew', 'wetu', 'wem', 'wesu', 'wesa', 'bwew', 'bwem' 
%   aggregation_method: 'avg', 'sum', 'eop' 
%
% 
% If you only set series_id, observations_start, and observation_end,  then 
% this code just does the same as using Matlab's inbuilt 'fetch' function 
% for accessing FRED (sole exception is different formatting of the 'Source').
% My 'Source' is the 'link'.
%
% 2014
% Robert Kirkby
% robertdkirkby.com
%
% Created based off of getFredData.m by Kim J. Ruhl (http://www.kimjruhl.com/computing/)
%
% This product uses the FRED® API but is not endorsed or certified by 
% the Federal Reserve Bank of St. Louis.  The terms of use can be found at
% http://api.stlouisfed.org/terms_of_use.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     

% FRED API documentation
% http://api.stlouisfed.org/docs/fred/

% This API is associated with my FRED account. If you are just running this function then 
% this API key is fine, but if you could change it to the API key associated with
% your own FRED login so much the better. If you plan to modify the function for your own use
% please do change the API key.
api_key='3521cf17600b6b9c26b4e024da47231f';

optionstring=[];
if nargin>=2
    if ~isempty(observation_start)
        if isdatetime(observation_start)
            observation_start=char(observation_start,'yyyy-MM-dd'); % which to string of appropriate format
        end
        optionstring=[optionstring,'&observation_start=',observation_start];
    end
end
if nargin>=3
    if ~isempty(observation_end)
        if isdatetime(observation_end)
            observation_end=char(observation_end,'yyyy-MM-dd'); % which to string of appropriate format
        end
        optionstring=[optionstring,'&observation_end=',observation_end];
    end
end
if nargin>=4
    if ~isempty(units)
        optionstring=[optionstring,'&units=',units];
    end
end
if nargin>=5
    if ~isempty(frequency)
        optionstring=[optionstring,'&frequency=',frequency];
    end
end
if nargin>=6
    if ~isempty(aggregation_method)
        optionstring=[optionstring,'&aggregation_method=',aggregation_method];
    end
end    
optionstringshort=optionstring;
if nargin>=7
    optionstring=[optionstringshort,'&realtime_start=',ondate,'&realtime_end=',ondate];
end
if nargin>=8
    realtime_start=ondate;
    optionstring=[optionstringshort,'&realtime_start=',realtime_start,'&realtime_end=',realtime_end];
end

% /fred/series
xDoc=xmlread(['https://api.stlouisfed.org/fred/series?series_id=',series_id,optionstring,'&api_key=',api_key]);
info=xDoc.getElementsByTagName('series');
output.Title=cellstr(char(getValue(info,'title')));
output.SeriesID=cellstr(char(getValue(info,'id')));
output.SeasonalAdjustment=cellstr(char(getValue(info,'seasonal_adjustment')));
output.Frequency=cellstr(char(getValue(info,'frequency')));
output.Units=cellstr(char(getValue(info,'units')));
output.DataRange=cellstr([char(getValue(info,'observation_start')),' to ',char(getValue(info,'observation_end'))]);
output.LastUpdated=cellstr(char(getValue(info,'last_updated')));
% Some variables, eg. 'JTSJOR', do not have notes
% field and this was causing errors. I have dealt with this by creating
% an empty Notes whenever there are none in FRED.
try
    output.Notes=cellstr(char(getValue(info,'notes')));
catch
    output.Notes={' '};
end

% /fred/release
xDoc1=xmlread(['https://api.stlouisfed.org/fred/series/release?series_id=',series_id,optionstring,'&api_key=',api_key]);
release=xDoc1.getElementsByTagName('release');
try
    output.Source=cellstr(char(getValue(release,'link')));
catch
    fprintf(['Source unavailable from FRED for series ',series_id],' \n')
end
output.Release=cellstr(char(getValue(release,'name')));

%['http://api.stlouisfed.org/fred/series/observations?series_id=',series_id,optionstring,'&api_key=',api_key]
% /fred/observations
xDoc2=xmlread(['https://api.stlouisfed.org/fred/series/observations?series_id=',series_id,optionstring,'&api_key=',api_key]);
data=xDoc2.getElementsByTagName('observation');
nn=data.getLength;
output.Dates=NaT(nn,1);
output.Data=NaN(nn,1);

%An observation contains 4 attributes: realtime_start, realtime_end, date,
%and value.  It's not clear the attributes are always in the same order, so
%this bit of code finds the indices that correspond to the date and value
value_idx= getIndex(data,'value');
date_idx= getIndex(data,'date');

temp1old='fakestartdate';
%read the data into a matrix
for i=1:nn
    temp1=data.item(i-1).getAttributes.item(date_idx).getValue;

    if strcmp(temp1,temp1old)
        
    else
        output.Dates(i)=datetime(char(temp1));
        temp2=data.item(i-1).getAttributes.item(value_idx).getValue;
        if ~strcmp(temp2,'.')
            output.Data(i)=str2num(temp2);
        end
    end
    temp1old=temp1;
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [idx] = getIndex(dd, pattern)

n_att = dd.item(0).getAttributes.getLength;

for j=1:n_att
    name=dd.item(0).getAttributes.item(j-1).getName;
    if( strcmp(char(name),pattern) )
        idx=j-1;
    end
end

end %getIndex

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [val] = getValue(dd, pattern)

n_att = dd.item(0).getAttributes.getLength;

for j=1:n_att
    name=dd.item(0).getAttributes.item(j-1).getName;
    if( strcmp(char(name),pattern) )
        val=dd.item(0).getAttributes.item(j-1).getValue;
    end
end

end %getValue

