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

% 1. Intercept precision and determine the EFFECTIVE number of arguments
nargin_temp = nargin;
if nargin_temp == 3 && ischar(index1) && any(strcmp({'single','double'}, index1))
    precision = index1;
    nargin_temp = 2;
elseif nargin_temp == 4 && ischar(index2) && any(strcmp({'single','double'}, index2))
    precision = index2;
    nargin_temp = 3;
elseif nargin_temp < 5
    precision = 'double';
else
    % precision was explicitly passed as the 5th argument
    nargin_temp = 4;
end

% 2. Handle empty ParamNames cleanly with the correct precision
if isempty(ParamNames)
    VectorOfParamValues = zeros(1, 0, precision);
    return
end

nCalibParams=length(ParamNames);
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);

% 3. Pre-allocate the vector in the requested precision
VectorOfParamValues=zeros(1, nCalibParams, precision);

% 4. Route to the correct block using the effective argument count
if nargin_temp==2
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                VectorOfParamValues(iCalibParam)=cast(gather(Parameters.(FullParamNames{iField})), precision);
                found=1;
                break
            end
        end
        if found==0
            % Have added this check so that user can see if they are missing a parameter
            warning(['FAILED TO FIND PARAMETER ',ParamNames{iCalibParam}])
        end
    end

elseif nargin_temp==3
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                temp=cast(gather(Parameters.(FullParamNames{iField})), precision);
                if isscalar(temp)
                    % Some parameters will depend on the index, some will not.
                    VectorOfParamValues(iCalibParam)=temp;
                else
                    VectorOfParamValues(iCalibParam)=temp(index1);
                end
                found=1;
                break
            end
        end
        if found==0
            % Have added this check so that user can see if they are missing a parameter
            warning(['FAILED TO FIND PARAMETER ',ParamNames{iCalibParam}])
        end
    end

elseif nargin_temp==4
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                temp=cast(gather(Parameters.(FullParamNames{iField})), precision);
                if isscalar(temp)
                    % parameter is scalar, so just store it
                    VectorOfParamValues(iCalibParam)=temp;
                elseif numel(temp)>length(temp)
                    % Some parameters will depend on both index1 and index2
                    VectorOfParamValues(iCalibParam)=temp(index1,index2);
                elseif size(temp,1)==length(temp)
                    % Some parameters will depend only on index1.
                    VectorOfParamValues(iCalibParam)=temp(index1);
                elseif size(temp,2)==length(temp)
                    % Some parameters will depend only on index2.
                    VectorOfParamValues(iCalibParam)=temp(1,index2);
                end
                found=1;
                break
            end
        end
        if found==0
            % Have added this check so that user can see if they are missing a parameter
            warning(['FAILED TO FIND PARAMETER ',ParamNames{iCalibParam}])
        end
    end


end