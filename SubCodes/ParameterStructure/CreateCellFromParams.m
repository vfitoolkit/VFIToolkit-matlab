function CellOfParamValues=CreateCellFromParams(Parameters,ParamNames,index1,index2,precision)
%
% CellOfParamValues=CreateCellFromParams(Parameters,ParamNames)
% CellOfParamValues=CreateCellFromParams(Parameters,ParamNames,precision)
% CellOfParamValues=CreateCellFromParams(Parameters,ParamNames,index1)
% CellOfParamValues=CreateCellFromParams(Parameters,ParamNames,index1,precision)
% CellOfParamValues=CreateCellFromParams(Parameters,ParamNames,index1,index2)
% CellOfParamValues=CreateCellFromParams(Parameters,ParamNames,index1,index2,precision)
%
% CreateCellFromParams looks in structure called 'Parameters' and
% then creates a cell containing the values of it's fields that
% correspond to those field names in ParamNames (and in the order
% given by CalibParamNames)
%
% Some parameters are stored in the Parameters structure as vectors or
% matrices (eg., because the parameter values depends on age). In these
% cases 'index1' (and 'index2') can be used to specify which is the relevant element.

nargin_temp=nargin;
if exist('index2','var') && ischar(index2)
    precision=index2;
    clear index2
    % Don't confuse `precision` with index 1 or 2
    nargin_temp=nargin_temp-1;
elseif exist('index1','var') && ischar(index1)
    precision=index1;
    clear index1
    % Don't confuse `precision` with index 1 or 2
    nargin_temp=nargin_temp-1;
elseif ~exist('precision','var')
    precision='double';
else
    % Don't confuse `precision` with index 1 or 2
    nargin_temp=nargin_temp-1;
end
if strcmp(precision,'single')
    precision_cast=@(x) single(x);
else
    precision_cast=@(x) x;
end

if isempty(ParamNames)
    CellOfParamValues=cell(precision_cast(0));
    return
end

nCalibParams=length(ParamNames);
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);

CellOfParamValues=cell(1,nCalibParams);
if nargin_temp==2
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                CellOfParamValues(iCalibParam)={precision_cast(Parameters.(FullParamNames{iField}))};
                found=1;
                break
            end
        end
        if found==0 % Have added this check so that user can see if they are missing a parameter
            dbstack
            error(['Failed to find parameter: ',ParamNames{iCalibParam}])
        end
    end
elseif nargin_temp==3
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                temp=precision_cast(gather(Parameters.(FullParamNames{iField})));
                if isscalar(temp) % Some parameters will depend on the index, some will not.
                    CellOfParamValues(iCalibParam)={temp};
                else
                    CellOfParamValues(iCalibParam)={temp(index1)};
                end
                found=1;
                break
            end
        end
        if found==0 % Have added this check so that user can see if they are missing a parameter
            dbstack
            error(['Failed to find parameter: ',ParamNames{iCalibParam}])
        end
    end
elseif nargin_temp==4
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                temp=precision_cast(gather(Parameters.(FullParamNames{iField})));
                if isscalar(temp) % parameter is scalar, so just store it
                    CellOfParamValues(iCalibParam)={temp};
                elseif numel(temp)>length(temp) % Some parameters will depend on both index1 and index2
                    CellOfParamValues(iCalibParam)={temp(index1,index2)};
                elseif size(temp,1)==length(temp) % Some parameters will depend only on index1.
                    CellOfParamValues(iCalibParam)={temp(index1)};
                elseif size(temp,2)==length(temp) % Some parameters will depend only on index2.
                    CellOfParamValues(iCalibParam)={temp(1,index2)};
                end
                found=1;
                break
            end
        end
        if found==0 % Have added this check so that user can see if they are missing a parameter
            dbstack
            error(['Failed to find parameter: ',ParamNames{iCalibParam}])
        end
    end
end


end
