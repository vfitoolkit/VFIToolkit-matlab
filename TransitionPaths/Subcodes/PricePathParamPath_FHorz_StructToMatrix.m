function [PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_FHorz_StructToMatrix(PricePathStruct,ParamPathStruct,N_j,T)

% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 

% Note: Internally PricePath is matrix of size T-by-'number of prices', similarly for ParamPath
% PricePath is matrix of size T-by-'number of prices'.
% Actually, some of those prices may be 1-by-N_j, so is more subtle than this.
PricePathNames=fieldnames(PricePathStruct);
PricePathSizeVec=zeros(1,length(PricePathNames)); % Allows for a given price param to depend on age (or permanent type)
for pp=1:length(PricePathNames)
    temp=PricePathStruct.(PricePathNames{pp});
    tempsize=size(temp);
    PricePathSizeVec(pp)=tempsize(tempsize~=T); % Get the dimension which is not T
    if ~any(PricePathSizeVec(pp)==[1,N_j])
        error(['PricePath for ', PricePathNames{pp}, ' appears to be the wrong size (should be 1-by-T or N_j-by-T)'])
    end
end
PricePathSizeVec=cumsum(PricePathSizeVec);
if length(PricePathNames)>1
    PricePathSizeVec=[[1,PricePathSizeVec(1:end-1)+1];PricePathSizeVec];
else
    PricePathSizeVec=[1;PricePathSizeVec];
end
PricePath=zeros(T,PricePathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for pp=1:length(PricePathNames)
    if size(PricePathStruct.(PricePathNames{pp}),1)==T
        PricePath(:,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=PricePathStruct.(PricePathNames{pp});
    else % Need to transpose
        PricePath(:,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=PricePathStruct.(PricePathNames{pp})';
    end
end
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
% Actually, some of those prices may be 1-by-N_j, so is more subtle than this.
ParamPathNames=fieldnames(ParamPathStruct);
ParamPathSizeVec=zeros(1,length(ParamPathNames)); % Allows for a given price param to depend on age (or permanent type)
for pp=1:length(ParamPathNames)
    temp=ParamPathStruct.(ParamPathNames{pp});
    tempsize=size(temp);
    ParamPathSizeVec(pp)=tempsize(tempsize~=T); % Get the dimension which is not T
    if ~any(ParamPathSizeVec(pp)==[1,N_j])
        error(['ParamPath for ', ParamPathNames{pp}, ' appears to be the wrong size (should be 1-by-T or N_j-by-T)'])
    end
end
ParamPathSizeVec=cumsum(ParamPathSizeVec);
if length(ParamPathNames)>1
    ParamPathSizeVec=[[1,ParamPathSizeVec(1:end-1)+1];ParamPathSizeVec];
else
    ParamPathSizeVec=[1;ParamPathSizeVec];
end
ParamPath=zeros(T,ParamPathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for pp=1:length(ParamPathNames)
    if size(ParamPathStruct.(ParamPathNames{pp}),1)==T
        ParamPath(:,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp))=ParamPathStruct.(ParamPathNames{pp});
    else % Need to transpose
        ParamPath(:,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp))=ParamPathStruct.(ParamPathNames{pp})';
    end
end



end