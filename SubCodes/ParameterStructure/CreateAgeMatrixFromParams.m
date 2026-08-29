function AgeMatrixOfParamValues=CreateAgeMatrixFromParams(Parameters,ParamNames,N_j)
%
% AgeMatrixOfParamValues=CreateAgeMatrixFromParams(Parameters,ParamNames,N_j)
%
% CreateAgeMatrixFromParams looks in structure called 'Parameters' and
% then creates a matrix with each column contains the values of it's fields that
% correspond to those field names in ParamNames (and in the order
% given by CalibParamNames) and each row is a different age (j).

nCalibParams=length(ParamNames);
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);
ParamDict=dictionary(string(FullParamNames),(1:nFields)');

AgeMatrixOfParamValues=zeros(N_j,nCalibParams);
for iCalibParam = 1:nCalibParams
    try
        iField=ParamDict(ParamNames{iCalibParam});
        if isscalar(Parameters.(FullParamNames{iField}))==1
            AgeMatrixOfParamValues(:,iCalibParam)=Parameters.(FullParamNames{iField})*ones(N_j,1,'gpuArray');
        else
            AgeMatrixOfParamValues(:,iCalibParam)=reshape(gpuArray(Parameters.(FullParamNames{iField})),[N_j,1]);
        end
    catch ME
        error(['Failed to find parameter ',ParamNames{iCalibParam}])
    end
end


end
