function output1 = ImportPSIDdata(jobname)
% Import the PSID data (and some of metadata) into matlab
%
% So, for example, I downloaded some PSID data and the data was called
% J226269.txt. To import it to matlab I then run ImportPSIDdata('J226269')
%
% To use this you must download your PSID data with 'xml' cookbook
% and with the data for SPSS.
% You must rename the two files ending in '.sps' to instead end in '_sps.txt'.
% 
%
% For the inputed 'jobname' --- the filename/code of your PSID dataset this
% code reads the metadata from the '.xml' file, finds out the fixed widths for
% each variable from the '.sps' file, and then imports the data itself from
% the '.txt' file.
%
% Output:
%  PSIDexplore: a structure that contains all the data and metadata

output1=struct();

%%
disp(['Starting import of PSID data for jobname ', jobname])

% jobname='J226274'

filename_xml = [jobname,'_codebook.xml']; % the .xml cookbook containing the metadata
filename_txt = [jobname,'.txt']; % the .txt file containing the data itself
filename_sps = [jobname,'_sps.txt']; % contains the formatting info needed to get the data from the .txt file
                                     % Note: you need to rename the .sps as _sps.txt (code checks for this and asks you to if you haven't)
filename_listcodes = [jobname, '_formats_sps.txt'];                                     

%% Read the .xml file to find out all of the details of the variables
disp(['Import of PSID data for jobname ', jobname, ' currently at step 1 of 4'])
PSIDdata=struct;
PSIDsummary=struct;
xDoc=xmlread(filename_xml);
allvariables=xDoc.getElementsByTagName('VARIABLE'); 
nPSIDvariables=allvariables.getLength;

strnamevec={'name','year','type_id','label','qtext','etext'}; %,'list_code'}; %It is important that 'name' is the first of these to be retrieved
tagnamevec={'NAME','YEAR','TYPE_ID','LABEL','QTEXT','ETEXT'}; %,'LIST_CODE'};

for ii=1:nPSIDvariables 
    thisvariable = allvariables.item(ii-1); % xml are indexed from 0

    for jj=1:length(strnamevec)
        strname=strnamevec{jj};
        tagname=tagnamevec{jj};
        
        thisfield = thisvariable.getElementsByTagName(tagname);
        field = thisfield.item(0);
        temp =cellstr(char(field.getFirstChild.getData));
        PSIDsummary.(strname){ii}=temp; 
        name=PSIDsummary.name{ii};
        PSIDdata.(name{:}).(strname)=temp;
    end
end

% list_codes are stored in the .xml, but I can't seem to get them out of
% this. So I get them from '_format.sps' instead (which has to be renamed
% '_format_sps.txt' to enable Matlab to access it).
if exist([jobname,'_formats_sps.txt'],'file')~=2
    error(['Matlab cannot access the data inside ',jobname,'_formats.sps directly. Please rename this file to ',jobname,'formats_sps.txt (or create a duplicate copy renamed in this manner)'])
end
fid = fopen(filename_listcodes,'r');
tline = fgetl(fid);
ii=1;
while ii<=nPSIDvariables
    name=PSIDsummary.name{ii};
    % Go through each line until you find the variable name
    if ~isempty(strfind(tline,PSIDsummary.name{ii}))
        jj=1;
        while jj<Inf
            tline = fgetl(fid);
            if strcmp(strtrim(tline),'.')
                jj=Inf;
            else
                PSIDdata.(name{:}).list_code(jj,1)={tline};
                jj=jj+1;
        	end
        end
        ii=ii+1;
        frewind(fid); % PSID xml file contains the variables 'out of order', so have to reset to beginning each time.
    else
        tline = fgetl(fid);
    end

    if ~ischar(tline) % Has reached end of the file without finding list_code for that variable, so set to empty and just move on to next variable
        ii=ii+1;
        frewind(fid); % PSID xml file contains the variables 'out of order', so have to reset to beginning each time.
        tline = fgetl(fid);
    end
end
fclose(fid);    

%% Now, read the .sps file to find out all of the variable locations.
disp(['Import of PSID data for jobname ', jobname, ' currently at step 2 of 4'])
% To do this you must rename the .sps as _sps.txt as otherwise matlab cannot read it.
% Following two lines do this for you by creating a copy of the .sps and
% then renaming it.

if exist([jobname,'_sps.txt'],'file')~=2
    error(['Matlab cannot access the data inside ',jobname,'.sps directly. Please rename this file to ',jobname,'_sps.txt (or create a duplicate copy renamed in this manner)'])
end


% April 2017: PSID sps file layout has changed. Old version is commented out below the following new version.
% April 2018: some PSID variable names now have an A2 or A3 on the end. Have had to make some minor changes to allow for this.
% June 2018: Correction: realised large data sets were used they were creating an error when the fixed locations reached four-digits: 
%                   Just had to make it 'temp+7+25' instead of 20 when creating 'tempstr'.
PSIDfixedlengths=zeros(nPSIDvariables,1);
fid = fopen(filename_sps,'r');
tline = fgetl(fid);
ii=1;
while ii<=nPSIDvariables
    if ~isempty(strfind(tline,PSIDsummary.name{ii}))
        temp=strfind(tline,PSIDsummary.name{ii}); % Find the name of the variable
        tempstr=tline(temp+7:min(temp+7+25,length(tline)));  % Grab a bunch of the characters that come after the variable name
        temp2=strfind(tempstr,'-'); % Find the '-' in the middle of tempstr
        % Figure out the length of this variable (in number of characters)
        % [This is a bit of a silly way to do it but is how the old PSID
        % sps formatting files gave the info and saves me rewriting Step 3
        % of 4 in this command.]
        firstpart=strtrim(tempstr(1:temp2-1)); % Start of April 2018 modification: grab this, then if it contains any spaces remove everything prior to first space. This gets rid of 'A2' etc at end of variable names.
        if max(isspace(firstpart))==1
            % Remove characters one by one until get to a space
            firstspace=0;
            while firstspace==0
                if ~isspace(firstpart(1))
                    firstpart=firstpart(2:end);
                else
                    firstspace=1;
                end
            end
        end % End of April 2018 modification.
        PSIDfixedlengths(ii)=1+str2num(strtrim(tempstr(temp2+1:end)))-str2num(strtrim(firstpart)); %str2num(strtrim(tempstr(1:temp2-1)));
        ii=ii+1;
        frewind(fid); % PSID xml file contains the variables 'out of order', so have to reset to beginning each time.
    else
        tline = fgetl(fid);
    end

    if ~ischar(tline)
        break
    end
end
fclose(fid);    
% % PSIDfixedlengths=zeros(nPSIDvariables,2);
% % fid = fopen(filename_sps,'r');
% % tline = fgetl(fid);
% % % First, go find the 'FORMATS' information. This is the only info we will need.
% % foundFORMATS=0;
% % while foundFORMATS==0
% %     if length(tline)>6
% % %         if strcmp('FORMATS',tline(1:7))==1
% %         if strcmp('DATA LIST',tline(1:9))==1
% %             foundFORMATS=1;
% %         end
% %     end
% %     tline = fgetl(fid);
% %     if tline==-1
% %         break
% %     end
% % end
% % ii=0;
% % % Now, just keep getting the formatting info until hit '.'
% % while ~strcmp('.',strtrim(tline))
% %     for tlinescan=1:length(tline)-1
% %         if strcmp(tline(tlinescan:tlinescan+1),'(F')
% %             ii=ii+1;
% %             PSIDfixedlengths(ii,1)=str2num(tline(tlinescan+2));
% %             PSIDfixedlengths(ii,2)=str2num(tline(tlinescan+4));
% %         end
% %     end
% %     tline = fgetl(fid);
% %     if tline==-1
% %         break
% %     end
% % end
if ii~=nPSIDvariables+1
    disp('ERROR: ImportPSIDdata does not find number of variables expected (A)')
end

%% Now we do the actual importing of the data
disp(['Import of PSID data for jobname ', jobname, ' currently at step 3 of 4'])

jj=0;
PSIDdataset=cell(nPSIDvariables,1);
PSIDcumfixedlengths=cumsum(PSIDfixedlengths(:,1));
fid = fopen(filename_txt,'r');
tline = fgetl(fid);
while ischar(tline)
    jj=jj+1;
    PSIDdataset{1,jj}=tline(1:PSIDcumfixedlengths(1));
    for ii=2:nPSIDvariables
        PSIDdataset{ii,jj}=tline(PSIDcumfixedlengths(ii-1)+1:PSIDcumfixedlengths(ii));
    end
    %read a line of data from file
    tline = fgetl(fid);
end
fclose(fid);

nPSIDhouseholds=jj;

PSIDdataset2=nan(nPSIDvariables,nPSIDhouseholds);
fid = fopen(filename_txt,'r');
tline = fgetl(fid);
for jj=1:nPSIDhouseholds
    PSIDdataset2(1,jj)=str2double(tline(1:PSIDcumfixedlengths(1)));
    for ii=2:nPSIDvariables
        PSIDdataset2(ii,jj)=str2double(tline(PSIDcumfixedlengths(ii-1)+1:PSIDcumfixedlengths(ii)));
    end
    %read a line of data from file
    tline = fgetl(fid);
    if ~ischar(tline)
        break
    end
end
fclose(fid);

%% Create structure that contain everything: metadata and the data itself
disp(['Import of PSID data for jobname ', jobname, ' currently at step 4 of 4'])

PSIDexplore = struct;
% Following loops are poorly constructed, but does the job. Could be sped up.
for ii=1:nPSIDvariables
    name=PSIDsummary.name{ii}; 
    for jj=1:nPSIDhouseholds    
        PSIDexplore.(name{:}).value(jj)=PSIDdataset(ii,jj);
    end
    %PSIDexplore.var(ii).name=name;
    for kk=2:length(strnamevec)
        strname=strnamevec{kk};
        temp=PSIDsummary.(strname){ii};
        PSIDexplore.(name{:}).(strname)=temp;
    end
end
% Given that PSIDexplore is largely just a combination of PSIDsummary of PSIDdata I should really
% take advantage of this to speed up these loops.

%% Currently 'value' contains the data. But it is all cells of strings while much of the actual data is numerical.
% I therefore have decided to also create 'value2' which converts it all
% into numerical data (and puts NaN where there are missing numbers), as
% well as converting to matrix (instead of cell)

disp('The field called value is the actual data, where there is a value2 it is an attempt to automatically convert to a numerical array (it is advised to compare value and value2 before using value2)')
for ii=1:nPSIDvariables
    name=PSIDsummary.name{ii};
    if isfield(PSIDexplore.(name{:}),'value')
        % First, check out the first 3 entries, to guess if this is actually numerical data
        seems_like_a_number=1;
        try % try the first
            str2double(PSIDexplore.(name{:}).value(1));
        catch
            seems_like_a_number=0; % if failed, probably not a number
        end
        try % try the second
            str2double(PSIDexplore.(name{:}).value(2));
        catch
            seems_like_a_number=0; % if failed, probably not a number
        end
        try % try the third
            str2double(PSIDexplore.(name{:}).value(3));
        catch
            seems_like_a_number=0; % if failed, probably not a number
        end

        if seems_like_a_number==1 % If it seems that it is numerical data
            % Currently MyPSIDdataset.V3.value are strings, so we need to convert them to numbers
            temp=cellfun(@str2num,PSIDexplore.(name{:}).value,'un',0); % not sure what ,'un',0) does here, copy-pasted it from internet
            % Before converting to a matrix, we have to replace the empty cells with nan
            empties = cellfun('isempty',temp);
            % Now change all the empty cells in temp from empty strings '' to double NaN
            temp(empties) = {NaN};
            PSIDexplore.(name{:}).value2=cell2mat(temp);
        end
    end
end

%% 
output1=PSIDexplore;

disp('Finished import of PSID data')









