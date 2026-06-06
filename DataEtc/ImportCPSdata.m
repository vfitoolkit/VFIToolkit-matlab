function output=ImportCPSdata(jobname)
% Imports an IPUMS CPS extract (fixed-width .dat plus SPSS .sps metadata)
% into a MATLAB struct.
%
% Usage:
%   CPSdata = ImportCPSdata('./Data/CPS/cps_00001')
%
% Requires two files to sit next to each other, sharing a common stem:
%   <jobname>.dat       - the fixed-width data file produced by IPUMS
%   <jobname>_sps.txt   - a copy of the IPUMS-supplied <jobname>.sps file
%                         renamed so that MATLAB will read it as text
%
% Output is a struct with one field per CPS variable. Each variable has:
%   .value       (Nx1 double, NaN where the source field was blank)
%   .label       (string from the SPSS 'variable labels' block)
%   .valuelabel  (cell array of {numericvalue, labelstring} pairs from the
%                 SPSS 'value labels' block, when present)
%
% IPUMS records implied decimal places on real-valued fields by writing
% "(d)" after the column range in the .sps file (e.g. "ASECWT 55-65 (4)").
% This function honours the indicator and divides the raw integer by 10^d.
%
% Implementation note: bulk-reads the whole .dat file in one textscan call
% using a width-specified format. For the typical IPUMS-CPS extract sizes
% (~700MB, ~10M rows, ~10 variables) this needs roughly 1-2 GB peak RAM
% and runs in seconds-to-minutes rather than the hours an
% str2double-per-cell loop would take.

filename_dat = [jobname,'.dat'];
filename_sps = [jobname,'_sps.txt'];

if exist(filename_sps,'file')~=2
    error(['ImportCPSdata: cannot find ', filename_sps, ...
           '. IPUMS ships the metadata as <jobname>.sps; rename a copy to ', ...
           '<jobname>_sps.txt so MATLAB will open it as text.']);
end
if exist(filename_dat,'file')~=2
    error(['ImportCPSdata: cannot find ', filename_dat]);
end

%% Parse the .sps file
[varnames, fromcol, tocol, decimals, labels, valuelabels] = parseSps(filename_sps);
nvars = numel(varnames);

%% Bulk-read the .dat file as fixed-width
fprintf('ImportCPSdata: reading %s (%d variables)\n', filename_dat, nvars);

fmtparts = cell(1, nvars);
for vv=1:nvars
    width = tocol(vv) - fromcol(vv) + 1;
    fmtparts{vv} = sprintf('%%%df', width);
end
fmt = strjoin(fmtparts, '');

fid = fopen(filename_dat,'r');
% Delimiter='' + WhiteSpace='' makes textscan honour the literal field
% widths in fmt; EmptyValue=NaN turns all-blank fields into NaN.
data = textscan(fid, fmt, ...
                'Delimiter', '', ...
                'WhiteSpace', '', ...
                'EmptyValue', NaN);
fclose(fid);

%% Assemble output, applying implied-decimal divisors
output = struct();
for vv=1:nvars
    v = data{vv};
    if decimals(vv) > 0
        v = v / 10^decimals(vv);
    end
    output.(varnames{vv}).value = v;
    if isfield(labels, varnames{vv})
        output.(varnames{vv}).label = labels.(varnames{vv});
    end
    if isfield(valuelabels, varnames{vv})
        output.(varnames{vv}).valuelabel = valuelabels.(varnames{vv});
    end
end

end


% ====================================================================
%  Local SPS parser
% ====================================================================

function [varnames, fromcol, tocol, decimals, labels, valuelabels] = parseSps(filename_sps)
% Pulls variable definitions, labels, and value labels out of an IPUMS .sps
% file. Recognises the optional "(d)" implied-decimal indicator after a
% column range.

varnames    = {};
fromcol     = [];
tocol       = [];
decimals    = [];
labels      = struct();
valuelabels = struct();

fid = fopen(filename_sps,'r');
cleaner = onCleanup(@() fclose(fid));

% --- skip to the "data list file" block ----------------------------
tline = fgetl(fid);
while ischar(tline)
    if length(tline)>=14 && strcmpi(strtrim(tline(1:min(14,end))), 'data list file')
        break
    end
    tline = fgetl(fid);
end

% --- read variable definitions until a line starting with '.' -------
tline = fgetl(fid);
while ischar(tline)
    raw = strtrim(tline);
    if isempty(raw) || raw(1)=='.'
        break
    end
    % Format of each line: "  VARNAME   from-to" optionally followed by " (d)"
    tok = regexp(raw, '^(\S+)\s+(\d+)\s*-\s*(\d+)(?:\s*\((\d+)\))?', 'tokens', 'once');
    if ~isempty(tok)
        varnames{end+1} = tok{1};           %#ok<AGROW>
        fromcol(end+1)  = str2double(tok{2});%#ok<AGROW>
        tocol(end+1)    = str2double(tok{3});%#ok<AGROW>
        if numel(tok)>=4 && ~isempty(tok{4})
            decimals(end+1) = str2double(tok{4});%#ok<AGROW>
        else
            decimals(end+1) = 0;%#ok<AGROW>
        end
    end
    tline = fgetl(fid);
end

% --- variable labels --------------------------------------------------
while ischar(tline)
    if length(tline)>=15 && strcmpi(strtrim(tline(1:min(15,end))), 'variable labels')
        break
    end
    tline = fgetl(fid);
end
tline = fgetl(fid);
while ischar(tline)
    raw = strtrim(tline);
    if isempty(raw) || raw(1)=='.'
        break
    end
    tok = regexp(raw, '^(\S+)\s+"(.*)"\s*$', 'tokens', 'once');
    if ~isempty(tok)
        labels.(tok{1}) = tok{2};
    end
    tline = fgetl(fid);
end

% --- value labels -----------------------------------------------------
while ischar(tline)
    if length(tline)>=12 && strcmpi(strtrim(tline(1:min(12,end))), 'value labels')
        break
    end
    tline = fgetl(fid);
end
currvarname = '';
tline = fgetl(fid);
while ischar(tline)
    raw = strtrim(tline);
    if isempty(raw) || raw(1)=='.'
        break
    end
    if raw(1)=='/'
        currvarname = strtrim(raw(2:end));
        valuelabels.(currvarname) = {};
    else
        tok = regexp(raw, '^(\S+)\s+"(.*)"\s*$', 'tokens', 'once');
        if ~isempty(tok) && ~isempty(currvarname)
            valuelabels.(currvarname){end+1,1} = ...
                {str2double(tok{1}), tok{2}};
        end
    end
    tline = fgetl(fid);
end

end
