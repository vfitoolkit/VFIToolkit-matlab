function CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters,ParamNames,N_j,dimJ)

% CreateCellOverAgeFromParams looks in structure called 'Parameters' and 
% then creates a matrix with each column contains the values of it's fields that
% correspond to those field names in ParamNames (and in the order
% given by CalibParamNames) and each row is a different age (j).
%
% Important difference from CreateCellFromParams is that it keeps the whole
% of the age-dependent parameters, rather than getting a specfic age j.

nCalibParams=length(ParamNames);
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);

% dimJ is which dimension age j should correspond to

CellOverAgeOfParamValues=cell(1,nCalibParams);
for iCalibParam = 1:nCalibParams
    found=0;
    for iField=1:nFields
        if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
            CellOverAgeOfParamValues(iCalibParam)={shiftdim(reshape(gpuArray(Parameters.(FullParamNames{iField})),[length(Parameters.(FullParamNames{iField})),1]).*ones(N_j,1,'gpuArray'),1-dimJ)}; % Note, if parameter depends on age this is just the column vector, if parameter does not depend on age then this turns it into a constant valued column vector
            found=1;
        end
    end
    if found==0 % Have added this check so that user can see if they are missing a parameter
        error(['Failed to find parameter ',ParamNames{iCalibParam}])
    end
end


end