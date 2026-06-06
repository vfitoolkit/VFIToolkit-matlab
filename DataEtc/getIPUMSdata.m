function [output, files] = getIPUMSdata(api_key, collection, varList, yearStart, yearEnd, ...
                                          sampleType, keepFiles, dataDir, jobname)
% Submit, poll, download and parse an IPUMS microdata extract via the
% IPUMS Extract API (v2). Blocking call: returns when the parsed data is
% in `output`.
%
% Currently supports collection 'cps'; other IPUMS collections (e.g.
% 'usa', 'ipumsi', 'atus') are stubs that will need their own sample-ID
% expander and parser.
%
% Inputs
%   api_key      IPUMS API key (required). Generate one at
%                  https://account.ipums.org/api_keys after creating a
%                  (free) IPUMS account. If empty, the function prints
%                  instructions and errors out.
%   collection   string, IPUMS collection name. Today only 'cps'.
%   varList      cellstr of IPUMS variable mnemonics, e.g.
%                  {'YEAR','SEX','AGE','EDUC','WKSWORK1','UHRSWORKLY', ...
%                   'ASECWT','ASECFLAG'}
%   yearStart    integer survey-year lower bound, e.g. 1976
%   yearEnd      integer survey-year upper bound, e.g. 2008
%   sampleType   (optional) selects which CPS samples to include:
%                  'asec' (default) -> cps<yyyy>_03s
%                  'basic'          -> reserved for future
%   keepFiles    (optional) 0 (default) deletes the downloaded data and
%                metadata files after parsing; 1 leaves them on disk
%   dataDir      (optional) folder to download into; defaults to pwd
%   jobname      (optional) filename stem under dataDir. Defaults to
%                  ipums_<collection>_<yyyymmdd>_<HHMMSS>
%
% Outputs
%   output       struct with one field per requested IPUMS variable, each
%                holding .value (Nx1 double), .label, .valuelabel. Matches
%                the layout returned by ImportCPSdata, so it is a drop-in
%                replacement at the call site.
%   files        struct with .dat, .sps_txt, and (if kept) .dat_gz paths,
%                so the user can re-parse later without re-pulling.
%
% Notes
%   The poll interval defaults to 5 minutes and the maximum wait to 3
%   hours. Override by editing the constants below if your extract is
%   tiny or huge.

%% ------------------------------------------------------------------
%  Constants
%% ------------------------------------------------------------------
apiBase          = 'https://api.ipums.org';
apiVersion       = '2';
pollIntervalSec  = 300;                                   % 5 minutes
maxWaitMin       = 180;                                   % 3 hours
httpTimeoutSec   = 60;                                    % per API call
downloadTimeoutSec = 600;                                 % per file download

%% ------------------------------------------------------------------
%  Input defaults and validation
%% ------------------------------------------------------------------
if nargin<5
    error('getIPUMSdata: requires at least api_key, collection, varList, yearStart, yearEnd');
end
if nargin<6 || isempty(sampleType), sampleType = 'asec';    end
if nargin<7 || isempty(keepFiles),  keepFiles  = 0;         end
if nargin<8 || isempty(dataDir),    dataDir    = pwd;       end
if nargin<9 || isempty(jobname)
    jobname = sprintf('ipums_%s_%s', lower(collection), datestr(now,'yyyymmdd_HHMMSS'));
end

% Resolve the API key as the very first thing. If empty, print
% instructions on how to get one before erroring out.
if isstring(api_key), api_key = char(api_key); end
if isempty(api_key)
    fprintf('\n');
    fprintf('getIPUMSdata: no IPUMS API key was provided.\n');
    fprintf('\n');
    fprintf('  To use this function you need a (free) IPUMS API key:\n');
    fprintf('    1. Create or sign in to an IPUMS account at\n');
    fprintf('         https://uma.pop.umn.edu/usa/user/new\n');
    fprintf('       (the same account works for IPUMS CPS, USA, etc.)\n');
    fprintf('    2. Generate a key at\n');
    fprintf('         https://account.ipums.org/api_keys\n');
    fprintf('    3. Pass the key string as the first argument to\n');
    fprintf('         getIPUMSdata(api_key, collection, varList, ...)\n');
    fprintf('\n');
    error('getIPUMSdata: api_key is required (see instructions above).');
end

if ~ischar(collection) && ~isstring(collection)
    error('getIPUMSdata: collection must be a string');
end
collection = lower(char(collection));

if ~iscellstr(varList) && ~isstring(varList) %#ok<ISCLSTR>
    error('getIPUMSdata: varList must be a cellstr of IPUMS variable mnemonics');
end
varList = cellstr(varList);

if ~isnumeric(yearStart) || ~isnumeric(yearEnd) || yearStart>yearEnd
    error('getIPUMSdata: yearStart and yearEnd must be integers with yearStart<=yearEnd');
end

if ~strcmp(collection,'cps')
    error(['getIPUMSdata: collection ''%s'' is not yet supported ', ...
           '(only ''cps'' is wired up). Add a branch in expandSampleRange ', ...
           'and parseDownload to support a new IPUMS collection.'], collection);
end

if exist(dataDir,'dir')~=7
    mkdir(dataDir);
end

%% ------------------------------------------------------------------
%  Phase 1: build and submit the extract request
%% ------------------------------------------------------------------
sampleIds = expandSampleRange(collection, yearStart, yearEnd, sampleType);
if isempty(sampleIds)
    error('getIPUMSdata: no IPUMS samples in range %d-%d for collection %s / sampleType %s', ...
          yearStart, yearEnd, collection, sampleType);
end

% Build the JSON body. We use jsonencode rather than letting webwrite
% serialise a struct so that empty objects render as {} (which IPUMS
% needs) and not as null.
body = struct();
body.description    = sprintf('getIPUMSdata %s %d-%d', collection, yearStart, yearEnd);
body.dataStructure.rectangular.on = 'P';
body.dataFormat     = 'fixed_width';
body.samples        = struct();
for ii=1:numel(sampleIds)
    body.samples.(sampleIds{ii}) = struct();
end
body.variables      = struct();
for ii=1:numel(varList)
    body.variables.(upper(varList{ii})) = struct();
end
jsonBody = jsonencode(body);

submitUrl = sprintf('%s/extracts?collection=%s&version=%s', apiBase, collection, apiVersion);

fprintf('\n');
fprintf('getIPUMSdata: requesting an IPUMS %s extract\n', upper(collection));
fprintf('  samples:   %d (survey years %d-%d, %s)\n', numel(sampleIds), yearStart, yearEnd, sampleType);
fprintf('  variables: %d (%s)\n', numel(varList), strjoin(varList, ', '));
fprintf('  jobname:   %s (saving to %s)\n', jobname, dataDir);
fprintf('Sending request to IPUMS...\n');
submitResp = httpSendJson(submitUrl, 'POST', jsonBody, api_key, httpTimeoutSec);

extractId = [];
if isfield(submitResp,'number'),  extractId = submitResp.number;
elseif isfield(submitResp,'id'),  extractId = submitResp.id;
end
if isempty(extractId)
    error('getIPUMSdata: submit response did not contain an extract id. Full response: %s', ...
          jsonencode(submitResp));
end
fprintf('Request has been sent to IPUMS (extract #%d). Waiting for the\n', extractId);
fprintf('dataset to be built on the IPUMS servers before retrieval; this\n');
fprintf('typically takes 5-30 minutes for a CPS extract. Will poll every\n');
fprintf('%d minutes for up to %d minutes total.\n', round(pollIntervalSec/60), maxWaitMin);

%% ------------------------------------------------------------------
%  Phase 2: poll until ready (blocking)
%% ------------------------------------------------------------------
statusUrl = sprintf('%s/extracts/%d?collection=%s&version=%s', ...
                    apiBase, extractId, collection, apiVersion);
info = waitForExtract(statusUrl, api_key, pollIntervalSec, maxWaitMin, httpTimeoutSec);

%% ------------------------------------------------------------------
%  Phase 3: download the data and metadata files
%% ------------------------------------------------------------------
[dataUrl, metaUrl] = pickDownloadLinks(info);

datGzPath  = fullfile(dataDir, [jobname '.dat.gz']);
spsPath    = fullfile(dataDir, [jobname '.sps']);
spsTxtPath = fullfile(dataDir, [jobname '_sps.txt']);
datPath    = fullfile(dataDir, [jobname '.dat']);

fprintf('Downloading data file (.dat.gz) from IPUMS...\n');
downloadOne(dataUrl, datGzPath, api_key, downloadTimeoutSec);

fprintf('Downloading SPSS metadata file (.sps) from IPUMS...\n');
downloadOne(metaUrl, spsPath,   api_key, downloadTimeoutSec);

fprintf('Decompressing data file...\n');
gunzip(datGzPath, dataDir);                  % writes <jobname>.dat alongside .dat.gz
movefile(spsPath, spsTxtPath);               % rename .sps to _sps.txt so parser treats it as text

%% ------------------------------------------------------------------
%  Phase 4: parse into MATLAB
%% ------------------------------------------------------------------
fprintf('Parsing fixed-width data into MATLAB workspace...\n');
output = parseDownload(collection, fullfile(dataDir, jobname));

%% ------------------------------------------------------------------
%  Phase 5: cleanup unless keepFiles==1
%% ------------------------------------------------------------------
files = struct();
if keepFiles==1
    fprintf('Keeping downloaded files (keepFiles==1):\n');
    fprintf('  %s\n', datGzPath);
    fprintf('  %s\n', datPath);
    fprintf('  %s\n', spsTxtPath);
    files.dat     = datPath;
    files.sps_txt = spsTxtPath;
    files.dat_gz  = datGzPath;
else
    fprintf('Cleaning up downloaded files (set keepFiles=1 to retain them).\n');
    delete(datGzPath);
    delete(datPath);
    delete(spsTxtPath);
end

fprintf('getIPUMSdata: done. Returned struct has %d variables.\n', numel(fieldnames(output)));

end


% ====================================================================
%  Local helpers
% ====================================================================

function sampleIds = expandSampleRange(collection, yearStart, yearEnd, sampleType)
% Convert a (collection, year range, sampleType) into a cell array of
% IPUMS sample-ID strings. Each collection gets one branch.

sampleIds = {};
switch collection
    case 'cps'
        switch lower(sampleType)
            case 'asec'
                % ASEC samples are named cps<yyyy>_03s
                for yy = yearStart:yearEnd
                    sampleIds{end+1} = sprintf('cps%d_03s', yy); %#ok<AGROW>
                end
            case 'basic'
                error('getIPUMSdata: sampleType ''basic'' not yet implemented for CPS');
            otherwise
                error('getIPUMSdata: unknown sampleType ''%s'' for CPS', sampleType);
        end
    otherwise
        error('expandSampleRange: collection ''%s'' not implemented', collection);
end
end


function resp = httpSendJson(url, method, jsonBody, api_key, timeoutSec)
% POST/PUT a JSON body to the IPUMS API. Uses matlab.net.http rather
% than webwrite because the older webwrite/webread family silently
% strips the Authorization header on some MATLAB versions, which
% manifests as a server-side 401.
authH = matlab.net.http.HeaderField('Authorization', api_key);
ctH   = matlab.net.http.HeaderField('Content-Type',  'application/json');
accH  = matlab.net.http.HeaderField('Accept',        'application/json');
req   = matlab.net.http.RequestMessage(method, [authH ctH accH], jsonBody);
opts  = matlab.net.http.HTTPOptions('ConnectTimeout', timeoutSec);
respMsg = req.send(matlab.net.URI(url), opts);
resp = parseHttpResponse(respMsg, method, url);
end


function resp = httpGetJson(url, api_key, timeoutSec)
% GET a JSON resource from the IPUMS API. matlab.net.http path, same
% reasoning as in httpSendJson.
authH = matlab.net.http.HeaderField('Authorization', api_key);
accH  = matlab.net.http.HeaderField('Accept',        'application/json');
req   = matlab.net.http.RequestMessage('GET', [authH accH]);
opts  = matlab.net.http.HTTPOptions('ConnectTimeout', timeoutSec);
respMsg = req.send(matlab.net.URI(url), opts);
resp = parseHttpResponse(respMsg, 'GET', url);
end


function resp = parseHttpResponse(respMsg, method, url)
% Common response handler. Errors out with the URL, HTTP status, and
% any response body so 401s/422s from IPUMS show an actionable message.
statusCode = double(respMsg.StatusCode);

% Stringify the body once for both error reporting and (when needed)
% JSON decoding.
bodyStr = '';
if ~isempty(respMsg.Body) && ~isempty(respMsg.Body.Data)
    raw = respMsg.Body.Data;
    if ischar(raw) || isstring(raw)
        bodyStr = char(raw);
    elseif isnumeric(raw)
        bodyStr = native2unicode(uint8(raw(:)'), 'UTF-8');
    elseif isstruct(raw) || iscell(raw)
        bodyStr = jsonencode(raw);
    end
end

if statusCode < 200 || statusCode >= 300
    error('getIPUMSdata: %s %s -> HTTP %d %s\nResponse body: %s', ...
          method, url, statusCode, char(respMsg.StatusCode), bodyStr);
end

% Return MATLAB's already-parsed struct if Content-Type was JSON,
% otherwise decode the captured string.
if ~isempty(respMsg.Body) && isstruct(respMsg.Body.Data)
    resp = respMsg.Body.Data;
elseif ~isempty(bodyStr)
    resp = jsondecode(bodyStr);
else
    resp = struct();
end
end


function info = waitForExtract(statusUrl, api_key, pollIntervalSec, maxWaitMin, timeoutSec)
% Poll the status endpoint until the extract reaches 'completed', or
% error out on 'failed'/'canceled', or time out after maxWaitMin.

tStart = tic;
maxSeconds = maxWaitMin * 60;
firstPoll = true;
while true
    info = httpGetJson(statusUrl, api_key, timeoutSec);
    elapsedSec = round(toc(tStart));
    if ~isfield(info,'status')
        error('waitForExtract: response missing ''status'' field. Body: %s', jsonencode(info));
    end
    fprintf('  [%s] elapsed %s, IPUMS extract status: %s\n', ...
            datestr(now,'yyyy-mm-dd HH:MM:SS'), formatElapsed(elapsedSec), info.status);

    switch lower(info.status)
        case 'completed'
            fprintf('Extract is ready on IPUMS, downloading now.\n');
            return
        case {'failed','canceled','cancelled'}
            msg = '';
            if isfield(info,'errors'), msg = jsonencode(info.errors); end
            error('getIPUMSdata: extract ended with status ''%s''. Errors: %s', info.status, msg);
        case {'queued','started','produced'}
            % keep polling
        otherwise
            warning('getIPUMSdata: unrecognised status ''%s''; continuing to poll', info.status);
    end

    if elapsedSec > maxSeconds
        error('getIPUMSdata: extract still %s after %d minutes; giving up', ...
              info.status, maxWaitMin);
    end

    % First poll fires immediately so the user sees the initial state;
    % subsequent polls wait pollIntervalSec.
    if firstPoll
        firstPoll = false;
    end
    pause(pollIntervalSec);
end
end


function [dataUrl, metaUrl] = pickDownloadLinks(info)
% Read the data URL and the SPSS-metadata URL out of the extract info
% struct. IPUMS v2 puts these under info.downloadLinks; field names are
% camelCase but the exact spellings have moved around historically, so
% we try a couple of plausible names for each.
if ~isfield(info,'downloadLinks')
    error('pickDownloadLinks: response missing ''downloadLinks''');
end
links = info.downloadLinks;

dataUrl = pickField(links, {'data','dataFile','dataFiles'});
metaUrl = pickField(links, {'spssCommandFile','spss','sps','spssSyntax'});

if isempty(dataUrl)
    error('pickDownloadLinks: could not find a data download link. Available fields: %s', ...
          strjoin(fieldnames(links), ', '));
end
if isempty(metaUrl)
    error('pickDownloadLinks: could not find an SPSS-metadata link. Available fields: %s', ...
          strjoin(fieldnames(links), ', '));
end

dataUrl = extractUrl(dataUrl);
metaUrl = extractUrl(metaUrl);
end


function v = pickField(s, candidates)
% Return s.(first candidate that exists), or '' if none.
v = '';
fn = fieldnames(s);
for ii=1:numel(candidates)
    hit = find(strcmpi(fn, candidates{ii}), 1);
    if ~isempty(hit)
        v = s.(fn{hit});
        return
    end
end
end


function s = formatElapsed(sec)
% Pretty-print an elapsed-seconds count as Hh MMm SSs (or shorter).
h = floor(sec/3600);
m = floor(mod(sec,3600)/60);
s_ = mod(sec,60);
if h>0
    s = sprintf('%dh %02dm %02ds', h, m, s_);
elseif m>0
    s = sprintf('%dm %02ds', m, s_);
else
    s = sprintf('%ds', s_);
end
end


function u = extractUrl(linkField)
% Each entry under downloadLinks is sometimes a string and sometimes a
% struct of the form struct('url', '...'). Normalise.
if ischar(linkField) || isstring(linkField)
    u = char(linkField);
elseif isstruct(linkField) && isfield(linkField,'url')
    u = char(linkField.url);
else
    error('extractUrl: cannot extract a URL from %s', class(linkField));
end
end


function downloadOne(url, localPath, api_key, timeoutSec)
% Stream a binary response straight to disk. We use matlab.net.http
% with a FileConsumer rather than websave so the Authorization header
% is preserved on the request (websave shares webwrite's header-
% stripping issue on some MATLAB versions).
authH    = matlab.net.http.HeaderField('Authorization', api_key);
req      = matlab.net.http.RequestMessage('GET', authH);
opts     = matlab.net.http.HTTPOptions('ConnectTimeout', timeoutSec);
consumer = matlab.net.http.io.FileConsumer(localPath);
respMsg  = req.send(matlab.net.URI(url), opts, consumer);
statusCode = double(respMsg.StatusCode);
if statusCode < 200 || statusCode >= 300
    error('getIPUMSdata: download %s -> HTTP %d %s', ...
          url, statusCode, char(respMsg.StatusCode));
end
end


function output = parseDownload(collection, jobnamePath)
% Hand off to the collection-specific parser. jobnamePath is the path
% stem (no extension); the parser appends .dat / _sps.txt etc.
switch collection
    case 'cps'
        output = ImportCPSdata(jobnamePath);
    otherwise
        error('parseDownload: collection ''%s'' not implemented', collection);
end
end
