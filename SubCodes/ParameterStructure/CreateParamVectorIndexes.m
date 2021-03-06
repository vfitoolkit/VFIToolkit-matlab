function varargout=CreateParamVectorIndexes(ParamNames,ParamNamesToGetIndexesOf)

% CreateParamVectorIndexes creates VectorOfIndexes which gives the indexes
% for the parameters named in ParamNamesToGetIndexesOf in the vector of
% parameters associated with the parameters named in ParamNames

nCalibParams=length(ParamNames);
nIndexParams=length(ParamNamesToGetIndexesOf);

VectorOfIndexes=nan; %nan(nIndexParams,1);
GotOne=0;
for iIndexParam=1:nIndexParams
    for iCalibParam = 1:nCalibParams
        if strcmp(ParamNamesToGetIndexesOf{iIndexParam},ParamNames{iCalibParam})
%             VectorOfIndexes(iIndexParam)=iCalibParam;
            GotOne=GotOne+1;
            VectorOfIndexes(GotOne)=iCalibParam;
            VectorOfUsed(GotOne)=iIndexParam;
        end
    end
end

varargout{1}=VectorOfIndexes;
if nargout==2 % If the second output is requested
    varargout{2}=VectorOfUsed;
end

end