function CellOfParamValues=CreateCellFromParams(Parameters,ParamNames,index1,index2)
%
% CellOfParamValues=CreateCellFromParams(Parameters,ParamNames)
% CellOfParamValues=CreateCellFromParams(Parameters,ParamNames,index1)
% CellOfParamValues=CreateCellFromParams(Parameters,ParamNames,index1,index2)
%
% CreateCellFromParams looks in structure called 'Parameters' and 
% then creates a cell containing the values of it's fields that
% correspond to those field names in ParamNames (and in the order
% given by CalibParamNames)
%
% Some parameters are stored in the Parameters structure as vectors or
% matrices (eg., because the parameter values depends on age). In these 
% cases 'index1' (and 'index2') can be used to specify which is the relevant element.

if isempty(ParamNames)
    CellOfParamValues=cell(0);
    return
end

nCalibParams=length(ParamNames);
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);

CellOfParamValues=cell(1,nCalibParams);
if nargin==2
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                CellOfParamValues(iCalibParam)={Parameters.(FullParamNames{iField})};
                found=1;
            end
        end
        if found==0 % Have added this check so that user can see if they are missing a parameter
            dbstack
            error(['Failed to find parameter: ',ParamNames{iCalibParam}])
        end
    end
elseif nargin==3
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                temp=gather(Parameters.(FullParamNames{iField}));
                if length(temp)>1 % Some parameters will depend on the index, some will not.
                    CellOfParamValues(iCalibParam)={temp(index1)}; 
                else
                    CellOfParamValues(iCalibParam)={temp};
                end
                found=1;
            end
        end
        if found==0 % Have added this check so that user can see if they are missing a parameter
            dbstack
            error(['Failed to find parameter: ',ParamNames{iCalibParam}])
        end
    end
elseif nargin==4
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                temp=gather(Parameters.(FullParamNames{iField}));
                if numel(temp)>length(temp) % Some parameters will depend on both index1 and index2
                    CellOfParamValues(iCalibParam)={temp(index1,index2)};
                elseif size(temp,1)==length(temp) % Some parameters will depend only on index1.
                    CellOfParamValues(iCalibParam)={temp(index1)};
                elseif size(temp,2)==length(temp) % Some parameters will depend only on index2.
                    CellOfParamValues(iCalibParam)={temp(1,index2)}; 
                else
                    CellOfParamValues(iCalibParam)={temp};
                end
                found=1;
            end
        end
        if found==0 % Have added this check so that user can see if they are missing a parameter
            dbstack
            error(['Failed to find parameter: ',ParamNames{iCalibParam}])
        end
    end
end


end