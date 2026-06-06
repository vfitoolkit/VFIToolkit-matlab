function output=ImportCEXdata(directory,yearfromstr,yeartostr)
% Imports the CEX PUMD (Consumer Expenditure Survey, Public Use Microdata).
% Loads the Interview Survey files (FMLI, MTBI, MEMI, ITBI, plus ITII from
% 2004 and NTAXI from 2013), the Detailed Expenditure files (expn/),
% the Paradata files (para/, 2009 onward), and the Diary Survey files
% (diary/).
%
% Inputs:
%   directory     - path to the folder where the various intrvwYY, expnYY,
%                   paraYY, diaryYY subfolders have been unzipped
%   yearfromstr   - first year to load, as a 4-digit string (e.g. '1980')
%   yeartostr     - last year to load,  as a 4-digit string (e.g. '2003')
%
% Output is a struct with fields:
%   .FMLI / .MTBI / .MEMI / .ITBI / .ITII / .NTAXI : interview files
%       each with subfields Y<yyyy>Q<q> (table per quarter)
%   .EXPN  : detailed expenditure, .EXPN.<code>.Y<yyyy> (table per year)
%   .PARA  : paradata, .PARA.fpar.Y<yyyy>, .PARA.mchi.Y<yyyy>
%   .DIARY : .DIARY.<code>.Y<yyyy>Q<q> from 1990 on (table per quarter)
%            .DIARY.<code>.Y<yyyy>   for the 1980/1981 annual files
%
% Notes on data availability (BLS PUMD):
%   - 1982 and 1983 microdata are not released by BLS
%   - The 1992 interview folder ships as "Intrvw92" (capital I); this loader
%     locates it case-insensitively
%   - 1984-1989 interview folders contain only FMLI files (MTBI/MEMI/ITBI
%     not released); the loader skips missing files with a warning
%   - 2017 and 2019 wrap their contents inside a same-named subfolder

output=struct();

yearfrom=str2double(yearfromstr);
yearto=str2double(yeartostr);

for ii_year=yearfrom:yearto

    ii_yearstr=num2str(ii_year);
    yy=ii_yearstr(3:4);
    yy_prev=sprintf('%02d',mod(ii_year-1,100));

    fprintf('Import CEX: year %s\n', ii_yearstr);

    %% ---------------------------------------------------------------
    %  Interview files
    %% ---------------------------------------------------------------
    interviewfolder         = locateFolder(directory, ['intrvw', yy]);
    interviewfolder_tminus1 = locateFolder(directory, ['intrvw', yy_prev]);

    % Some years (2017, 2019) put the actual data inside a same-named
    % subfolder. Descend into it when present.
    interviewfolder         = descendIfNested(interviewfolder,         ['intrvw', yy]);
    interviewfolder_tminus1 = descendIfNested(interviewfolder_tminus1, ['intrvw', yy_prev]);

    if isempty(interviewfolder)
        warning('ImportCEXdata: interview folder for %s not found; skipping year', ii_yearstr);
    else
        % File set depends on year (BLS added ITII in 2004 and NTAXI in 2013 Q2)
        if ii_year>=2013
            interviewfiles      ={'fmli','mtbi','memi','itbi','itii','ntaxi'};
            interviewfilesupper ={'FMLI','MTBI','MEMI','ITBI','ITII','NTAXI'};
        elseif ii_year>=2004
            interviewfiles      ={'fmli','mtbi','memi','itbi','itii'};
            interviewfilesupper ={'FMLI','MTBI','MEMI','ITBI','ITII'};
        else
            interviewfiles      ={'fmli','mtbi','memi','itbi'};
            interviewfilesupper ={'FMLI','MTBI','MEMI','ITBI'};
        end

        for ff=1:length(interviewfiles)
            stub      = interviewfiles{ff};
            fieldname = interviewfilesupper{ff};

            % Q2, Q3, Q4 live in this year's folder
            for qq=2:4
                fpath = locateCsvFile(interviewfolder, [stub, yy, num2str(qq)]);
                if ~isempty(fpath)
                    output.(fieldname).(['Y',ii_yearstr,'Q',num2str(qq)]) = importCEX_interviewfile(fpath);
                end
            end

            % Q1 lives in the previous year's folder (except for the very
            % first year on disk: 1980)
            qq=1;
            if ii_year==1980
                fpath = locateCsvFile(interviewfolder, [stub, yy, num2str(qq)]);
            elseif ~isempty(interviewfolder_tminus1)
                fpath = locateCsvFile(interviewfolder_tminus1, [stub, yy, num2str(qq)]);
            else
                fpath = '';
            end
            if ~isempty(fpath)
                output.(fieldname).(['Y',ii_yearstr,'Q',num2str(qq)]) = importCEX_interviewfile(fpath);
            end
        end
    end

    %% ---------------------------------------------------------------
    %  Detailed Expenditure (expn)
    %% ---------------------------------------------------------------
    % expn ships as its own top-level folder for 1996-2016 (except 2001),
    % and as a subfolder of intrvw for 2001 and 2017+.
    expnfolder = locateFolder(directory, ['expn', yy]);
    if isempty(expnfolder) && ~isempty(interviewfolder)
        expnfolder = locateFolder(interviewfolder, ['expn', yy]);
    end

    if ~isempty(expnfolder)
        csvs = dir(fullfile(expnfolder, '*.csv'));
        for kk=1:length(csvs)
            fname = csvs(kk).name;
            [~, base, ~] = fileparts(fname);
            % filenames look like 'apa96.csv' / 'cla10.csv' — strip the
            % 2-digit year suffix to recover the topic code
            if length(base)>=3 && strcmp(base(end-1:end), yy)
                code = base(1:end-2);
            else
                code = base;
            end
            % MATLAB struct fields must start with a letter
            code = matlab.lang.makeValidName(code);
            output.EXPN.(upper(code)).(['Y',ii_yearstr]) = ...
                importCEX_interviewfile(fullfile(expnfolder, fname));
        end
    end

    %% ---------------------------------------------------------------
    %  Paradata (para)
    %% ---------------------------------------------------------------
    % para exists from 2009 on. Ships as own top-level folder through 2016,
    % then as subfolder of intrvw from 2017+.
    parafolder = locateFolder(directory, ['para', yy]);
    if isempty(parafolder) && ~isempty(interviewfolder)
        parafolder = locateFolder(interviewfolder, ['para', yy]);
    end

    if ~isempty(parafolder)
        csvs = dir(fullfile(parafolder, '*.csv'));
        for kk=1:length(csvs)
            fname = csvs(kk).name;
            [~, base, ~] = fileparts(fname);
            % Filenames are 'fparPPYY.csv' / 'mchiPPYY.csv' where PP is the
            % previous year's 2-digit suffix. Recover the bare stub.
            if length(base)>=4 && strcmp(base(end-3:end), [yy_prev, yy])
                code = base(1:end-4);
            else
                code = base;
            end
            code = matlab.lang.makeValidName(code);
            output.PARA.(code).(['Y',ii_yearstr]) = ...
                importCEX_interviewfile(fullfile(parafolder, fname));
        end
    end

    %% ---------------------------------------------------------------
    %  Diary files
    %% ---------------------------------------------------------------
    % Diary structure:
    %   1980-1981: annual files (fmld80.csv, expd80.csv, memd80.csv)
    %   1990-1999: quarterly with fmld/dtbd/expd/memd
    %   2000-2001: nested in a same-named subfolder
    %   2002+    : flat quarterly; from 2017 a dtid file is added
    diaryfolder = locateFolder(directory, ['diary', yy]);
    diaryfolder = descendIfNested(diaryfolder, ['diary', yy]);

    if ~isempty(diaryfolder)
        csvs = dir(fullfile(diaryfolder, '*.csv'));
        for kk=1:length(csvs)
            fname = csvs(kk).name;
            [~, base, ~] = fileparts(fname);

            % Try to parse <stub><yy><q> first (quarterly), else <stub><yy>
            % (annual, used in 1980/1981).
            qsuffix = '';
            if length(base)>=4 && strcmp(base(end-2:end-1), yy) ...
                                    && any(base(end)=='1234')
                code = base(1:end-3);
                qsuffix = ['Q', base(end)];
            elseif length(base)>=3 && strcmp(base(end-1:end), yy)
                code = base(1:end-2);
            else
                code = base;
            end
            code = matlab.lang.makeValidName(code);

            yfield = ['Y', ii_yearstr];
            if isempty(qsuffix)
                output.DIARY.(upper(code)).(yfield) = ...
                    importCEX_interviewfile(fullfile(diaryfolder, fname));
            else
                output.DIARY.(upper(code)).([yfield, qsuffix]) = ...
                    importCEX_interviewfile(fullfile(diaryfolder, fname));
            end
        end
    end

end

end


% ====================================================================
%  Local helpers
% ====================================================================

function found = locateFolder(parent, target)
% Case-insensitive lookup of a subfolder named `target` inside `parent`.
% Returns the full path if found, or '' otherwise.
found = '';
if isempty(parent), return; end
candidates = dir(parent);
candidates = candidates([candidates.isdir]);
for ii=1:numel(candidates)
    if strcmpi(candidates(ii).name, target)
        found = fullfile(parent, candidates(ii).name);
        return;
    end
end
end


function folder = descendIfNested(folder, innername)
% If `folder` contains a subfolder of the same name (BLS does this for 2017
% and 2019), descend into it.
if isempty(folder), return; end
nested = locateFolder(folder, innername);
if ~isempty(nested)
    folder = nested;
end
end


function fpath = locateCsvFile(folder, basename)
% Look for either basename.csv or basenamex.csv (the 'x' variant denotes
% the BLS-revised Q1 file used from 2002 onward in the same-year folder).
% Case-insensitive match. Returns '' if neither exists.
fpath = '';
if isempty(folder), return; end
candidates = dir(fullfile(folder, '*.csv'));
target_a = [basename,  '.csv'];
target_b = [basename, 'x.csv'];
for ii=1:numel(candidates)
    if strcmpi(candidates(ii).name, target_a) || ...
       strcmpi(candidates(ii).name, target_b)
        fpath = fullfile(folder, candidates(ii).name);
        return;
    end
end
end
