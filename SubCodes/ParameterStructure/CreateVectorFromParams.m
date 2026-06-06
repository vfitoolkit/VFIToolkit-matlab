function VectorOfParamValues=CreateVectorFromParams(Parameters,ParamNames,index1,index2,precision)
%
% VectorOfParamValues=CreateVectorFromParams(Parameters,ParamNames)
% VectorOfParamValues=CreateVectorFromParams(Parameters,ParamNames,precision)
% VectorOfParamValues=CreateVectorFromParams(Parameters,ParamNames,index1)
% VectorOfParamValues=CreateVectorFromParams(Parameters,ParamNames,index1,precision)
% VectorOfParamValues=CreateVectorFromParams(Parameters,ParamNames,index1,index2)
% VectorOfParamValues=CreateVectorFromParams(Parameters,ParamNames,index1,index2,precision)
%
% CreateVectorFromParams looks in structure called 'Parameters' and
% then creates a row vector containing the values of it's fields that
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

nCalibParams=length(ParamNames);
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);

VectorOfParamValues=zeros(1,nCalibParams,precision);
if nargin_temp==2
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                VectorOfParamValues(iCalibParam)=precision_cast(gather(Parameters.(FullParamNames{iField})));
                found=1;
                break
            end
        end
        if found==0 % Have added this check so that user can see if they are missing a parameter
            warning(['FAILED TO FIND PARAMETER ',ParamNames{iCalibParam}])
        end
    end
elseif nargin_temp==3
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                temp=precision_cast(gather(Parameters.(FullParamNames{iField})));
                if isscalar(temp) % Some parameters will depend on the index, some will not.
                    VectorOfParamValues(iCalibParam)=temp;
                else
                    VectorOfParamValues(iCalibParam)=temp(index1);
                end
                found=1;
                break
            end
        end
        if found==0 % Have added this check so that user can see if they are missing a parameter
            warning(['FAILED TO FIND PARAMETER ',ParamNames{iCalibParam}])
        end
    end
elseif nargin_temp==4
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                temp=precision_cast(gather(Parameters.(FullParamNames{iField})));
                if isscalar(temp) % parameter is scalar, so just store it
                    VectorOfParamValues(iCalibParam)=temp;
                elseif numel(temp)>length(temp) % Some parameters will depend on both index1 and index2
                    VectorOfParamValues(iCalibParam)=temp(index1,index2);
                elseif size(temp,1)==length(temp) % Some parameters will depend only on index1.
                    VectorOfParamValues(iCalibParam)=temp(index1);
                elseif size(temp,2)==length(temp) % Some parameters will depend only on index2.
                    VectorOfParamValues(iCalibParam)=temp(1,index2);
                end
                found=1;
                break
            end
        end
        if found==0 % Have added this check so that user can see if they are missing a parameter
            warning(['FAILED TO FIND PARAMETER ',ParamNames{iCalibParam}])
        end
    end
end

% if nCalibParams==0
%     VectorOfParamValues=[]; % Have to treat this specially so that length(VectorOfParamValues) evaluates to zero
% end

end
