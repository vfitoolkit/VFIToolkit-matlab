function AgeMatrixOfParamValues=CreateAgeMatrixFromParams(Parameters,ParamNames,N_j); %,index1,index2)
%
% AgeMatrixOfParamValues=CreateAgeMatrixFromParams(Parameters,ParamNames,N_j)
% AgeMatrixOfParamValues=CreateAgeMatrixFromParams(Parameters,ParamNames,N_j,index1)
% AgeMatrixOfParamValues=CreateAgeMatrixFromParams(Parameters,ParamNames,N_j,index1,index2)
%
% CreateAgeMatrixFromParams looks in structure called 'Parameters' and 
% then creates a matrix with each column contains the values of it's fields that
% correspond to those field names in ParamNames (and in the order
% given by CalibParamNames) and each row is a different age (j).
%
% Some parameters are stored in the Parameters structure as vectors or
% matrices (eg., because the parameter values depends on age). In these 
% cases 'index1' (and 'index2') can be used to specify which is the relevant element.
%

nCalibParams=length(ParamNames);
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);

AgeMatrixOfParamValues=zeros(N_j,nCalibParams);
% if nargin==2
    for iCalibParam = 1:nCalibParams
        found=0;
        for iField=1:nFields
            if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
                AgeMatrixOfParamValues(:,iCalibParam)=reshape(gather(Parameters.(FullParamNames{iField})),[length(Parameters.(FullParamNames{iField})),1]).*ones(N_j,1); % Note, if parameter depends on age this is just the column vector, if parameter does not depend on age then this turns it into a constant valued column vector
                found=1;
            end
        end
        if found==0 % Have added this check so that user can see if they are missing a parameter
            fprintf(['FAILED TO FIND PARAMETER ',ParamNames{iCalibParam}])
        end
    end
% elseif nargin==3
%     for iCalibParam = 1:nCalibParams
%         found=0;
%         for iField=1:nFields
%             if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
%                 temp=gather(Parameters.(FullParamNames{iField}));
%                 if length(temp)>1 % Some parameters will depend on the index, some will not.
%                     AgeMatrixOfParamValues(iCalibParam)=temp(index1); 
%                 else
%                     AgeMatrixOfParamValues(iCalibParam)=temp;
%                 end
%                 found=1;
%             end
%         end
%         if found==0 % Have added this check so that user can see if they are missing a parameter
%             fprintf(['FAILED TO FIND PARAMETER ',ParamNames{iCalibParam}])
%         end
%     end
% elseif nargin==4
%     for iCalibParam = 1:nCalibParams
%         found=0;
%         for iField=1:nFields
%             if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
%                 temp=gather(Parameters.(FullParamNames{iField}));
%                 if numel(temp)>length(temp) % Some parameters will depend on both index1 and index2
%                     AgeMatrixOfParamValues(iCalibParam)=temp(index1,index2);
%                 elseif size(temp,1)==length(temp) % Some parameters will depend only on index1.
%                     AgeMatrixOfParamValues(iCalibParam)=temp(index1); 
%                 elseif size(temp,2)==length(temp) % Some parameters will depend only on index2.
%                     AgeMatrixOfParamValues(iCalibParam)=temp(1,index2); 
%                 else
%                     AgeMatrixOfParamValues(iCalibParam)=temp;
%                 end
%                 found=1;
%             end
%         end
%         if found==0 % Have added this check so that user can see if they are missing a parameter
%             fprintf(['FAILED TO FIND PARAMETER ',ParamNames{iCalibParam}])
%         end
%     end
% end

end