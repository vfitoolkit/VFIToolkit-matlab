function output1 = ImportPSIDdata(jobname, ExtraOutput)

% Import the PSID data (and some of metadata) into matlab
%
% So, for example, I downloaded some PSID data and the data was called
% J179504.txt. To import it to matlab I then run ImportPSIDdata('J179504')
%
% To use this you must download your PSID data with 'xml' cookbook
% and with the data for SPSS.
%
% For the inputed 'jobname' --- the filename/code of your PSID dataset this
% code reads the metadata from the '.xml' file, finds out the fixed widths for
% each variable from the '.sps' file, and then imports the data itself from
% the '.txt' file.
%
% In basic mode it returns:
%  PSIDexplore: a structure that contains all the data and metadata
%
% If you give ExtraOutput=1, it also returns the objects: (DISABLED)
%  PSIDsummary: a structure containing the metadata
%  PSIDdataset: a cell array containing all of the data as strings
%  PSIDdataset2: same as PSIDdataset, except it is an array, missing observations show as NaN
% These extra objects just repeat the contents of PSIDexplore, but maybe
% they are more useful forms for certain purposes.

%%
disp(['Starting import of PSID data for jobname ', jobname])

% jobname='J179504'

filename_xml = [jobname,'.xml']; % the .xml cookbook containing the metadata
filename_txt = [jobname,'.txt']; % the .txt file containing the data itself
filename_sps = [jobname,'.sps']; % contains the formatting info needed to get the data from the .txt file

%% Read the .xml file to find out all of the details of the variables
PSIDdata=struct;
PSIDsummary=struct;
xDoc=xmlread(filename_xml);
allvariables=xDoc.getElementsByTagName('VARIABLE'); 
nPSIDvariables=allvariables.getLength;

strnamevec={'name','year','type_id','label','qtext','etext'}; %It is important that 'name' is the first of these to be retrieved
tagnamevec={'NAME','YEAR','TYPE_ID','LABEL','QTEXT','ETEXT'};

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

%% Now, read the .sps file to find out all of the variable locations.
% To do this you must rename the .sps as _sps.txt as otherwise matlab cannot read it.

PSIDfixedlengths=zeros(nPSIDvariables,2);

fid = fopen(filename_sps,'r');
tline = fgetl(fid);
% First, go find the 'FORMATS' information. This is the only info we will need.
foundFORMATS=0;
while foundFORMATS==0
    if length(tline)>6
        if strcmp('FORMATS',tline(1:7))==1
            foundFORMATS=1;
        end
    end
    tline = fgetl(fid);
end
ii=0;
% Now, just keep getting the formatting info until hit '.'
while ~strcmp('.',tline(1))
    for tlinescan=1:length(tline)-1
        if strcmp(tline(tlinescan:tlinescan+1),'(F')
            ii=ii+1;
            PSIDfixedlengths(ii,1)=str2num(tline(tlinescan+2));
            PSIDfixedlengths(ii,2)=str2num(tline(tlinescan+4));
        end
    end
    tline = fgetl(fid);
end
if ii~=nPSIDvariables
    disp('ERROR: ImportPSIDdata does not find number of variables expected (A)')
end

% PSIDlineformat=['%',num2str(PSIDfixedlengths(1)),'u']; %This didn't work.
% %     Had problems with the decimal points in the floating point numbers.
% for ii=2:nPSIDvariables
%     if PSIDfixedlengths(ii,2)==0
%         PSIDlineformat=[PSIDlineformat,' %',num2str(PSIDfixedlengths(ii,1)),'u'];
%     else
%         PSIDlineformat=[PSIDlineformat,' %',num2str(PSIDfixedlengths(ii,1)),'.',num2str(PSIDfixedlengths(ii,2)),'f'];
%     end
% end

%% Now we do the actual importing of the data

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
end
fclose(fid);

%% Create two structures that contain everything: metadata and the data itself
% First just calls the variables var(ii), second uses the actual names of the PSID
% variables

% PSIDexplore = struct;
% % Following loops are poorly constructed, but does the job. Could be sped up.
% for ii=1:nPSIDvariables
%     for jj=1:nPSIDhouseholds    
%         PSIDexplore.var(ii).value(jj)=PSIDdataset(ii,jj);
%     end
%     name=PSIDsummary.name{ii};
%     PSIDexplore.var(ii).name=name;
%     for kk=2:length(strnamevec)
%         strname=strnamevec{kk};
%         temp=PSIDsummary.(strname){ii};
%         PSIDexplore.var(ii).(strname)=temp;
%     end
% end

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
% take advantage of this



%% 
output1=PSIDexplore;
if nargin>1 %This is currently DISABLED
    if ExtraOutput==1
        output2=PSIDsummary;
        output3=PSIDdataset;
        output4=PSIDdataset2;
    end
end

disp('Finished import of PSID data')









