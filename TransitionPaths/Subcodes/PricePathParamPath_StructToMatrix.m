function [PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePathStruct,ParamPathStruct,T)

% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 

PricePathNames=fieldnames(PricePathStruct);
PricePathSizeVec=zeros(1,length(PricePathNames)); % Allows for a given price param to depend on age (or permanent type)
for pp=1:length(PricePathNames)
    temp=PricePathStruct.(PricePathNames{pp});
    tempsize=size(temp);
    PricePathSizeVec(pp)=tempsize(tempsize~=T); % Get the dimension which is not T
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
    %     PricePath(:,ii)=PricePathStruct.(PricePathNames{ii});
end

ParamPathNames=fieldnames(ParamPathStruct);
if ~isempty(ParamPathNames)
    ParamPathSizeVec=zeros(1,length(ParamPathNames)); % Allows for a given price param to depend on age (or permanent type)
    for pp=1:length(ParamPathNames)
        temp=ParamPathStruct.(ParamPathNames{pp});
        tempsize=size(temp);
        ParamPathSizeVec(pp)=tempsize(tempsize~=T); % Get the dimension which is not T
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
        %     ParamPath(:,pp)=ParamPathStruct.(ParamPathNames{pp});
    end
else
    ParamPathSizeVec=[];
    ParamPath=[];
end


end