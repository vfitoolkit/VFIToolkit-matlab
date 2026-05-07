function varargout=PricePathParamPath_FHorz_StructToMatrix(PricePathStruct,ParamPathStruct,N_j,T,N_i)
% varargout={PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec}
% but for models with permanent types, include N_i as input and get
% varargout={PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec,PricePathSizeVec_ii,ParamPathSizeVec_ii}
%
% N_i is an optional input, only used for models with permanent type
% PricePathSizeVec_ii,ParamPathSizeVec_ii are only output for models with permanent type
% 
% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 

if ~exist('N_i','var')
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
    
    varargout={PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec};

else
    %% With permanent types (need to allow for parameters to depend on permanent type)
    % Note: Internally PricePathOld is matrix of size T-by-'number of prices'.
    % ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
    % Actually, some of those prices are 1-by-N_j or N_i or both, so is more subtle than this.
    PricePathNames=fieldnames(PricePathStruct);
    PricePathSizeVec=zeros(1,length(PricePathNames)); % Allows for a given price param to depend on age (or permanent type)
    if isstruct(N_j)
        Names_i=fieldnames(N_j);
        for pp=1:length(PricePathNames)
            if isstruct(PricePathStruct.(PricePathNames{pp}))
                tempptypenames=fieldnames(PricePathStruct.(PricePathNames{pp}));
                temp=PricePathStruct.(PricePathNames{pp}).(tempptypenames{1});
                tempsize=size(temp);
                PricePathSizeVec(pp)=length(tempptypenames)*tempsize(tempsize~=T); % Get the dimension which is not T
                for ii=1:N_i
                    N_j_temp=N_j.(Names_i{ii});
                    if ~any(PricePathSizeVec(pp)==[1,N_i,N_j_temp,N_i*N_j_temp])
                        error(['PricePath for ', PricePathNames{pp}, ' appears to be the wrong size (should be 1-by-T or N_j-by-T or N_i-by-T)'])
                    end
                end
            else
                temp=PricePathStruct.(PricePathNames{pp});
                tempsize=size(temp);
                PricePathSizeVec(pp)=tempsize(tempsize~=T); % Get the dimension which is not T
                for ii=1:N_i
                    N_j_temp=N_j.(Names_i{ii});
                    if ~any(PricePathSizeVec(pp)==[1,N_i,N_j_temp])
                        error(['PricePath for ', PricePathNames{pp}, ' appears to be the wrong size (should be 1-by-T or N_j-by-T or N_i-by-T)'])
                    end
                end
            end
        end  
    else
        for pp=1:length(PricePathNames)
            if isstruct(PricePathStruct.(PricePathNames{pp}))
                tempptypenames=fieldnames(PricePathStruct.(PricePathNames{pp}));
                temp=PricePathStruct.(PricePathNames{pp}).(tempptypenames{1});
                tempsize=size(temp);
                PricePathSizeVec(pp)=length(tempptypenames)*tempsize(tempsize~=T); % Get the dimension which is not T
                for ii=1:N_i
                    if ~any(PricePathSizeVec(pp)==[1,N_i,N_j,N_i*N_j])
                        error(['PricePath for ', PricePathNames{pp}, ' appears to be the wrong size (should be 1-by-T or N_j-by-T or N_i-by-T)'])
                    end
                end
            else
                temp=PricePathStruct.(PricePathNames{pp});
                tempsize=size(temp);
                PricePathSizeVec(pp)=tempsize(tempsize~=T); % Get the dimension which is not T
                for ii=1:N_i
                    if ~any(PricePathSizeVec(pp)==[1,N_i,N_j])
                        error(['PricePath for ', PricePathNames{pp}, ' appears to be the wrong size (should be 1-by-T or N_j-by-T or N_i-by-T)'])
                    end
                end
            end
        end  
    end


    % Also need what size these are conditional on ptype (as some of PricePath/ParamPath may differ by ptype)
    PricePathSizeVec_ii=PricePathSizeVec;
    PricePathSizeVec_ii(PricePathSizeVec_ii==N_i)=1; % Just use one of a price that depends on ptype
    PricePathSizeVec_ii=cumsum(PricePathSizeVec_ii);
    if length(PricePathNames)>1
        PricePathSizeVec_ii=[[1,PricePathSizeVec_ii(1:end-1)+1];PricePathSizeVec_ii];
    else
        PricePathSizeVec_ii=[1;PricePathSizeVec_ii];
    end
    
    PricePathSizeVec=cumsum(PricePathSizeVec);
    if length(PricePathNames)>1
        PricePathSizeVec=[[1,PricePathSizeVec(1:end-1)+1];PricePathSizeVec];
    else
        PricePathSizeVec=[1;PricePathSizeVec];
    end
    PricePath=zeros(T,PricePathSizeVec(2,end)); % Do this seperately afterwards so that can preallocate the memory
    for pp=1:length(PricePathNames)
        if isstruct(PricePathStruct.(PricePathNames{pp})) % depends on ptype as structure
            tempptypenames=fieldnames(PricePathStruct.(PricePathNames{pp}));
            for ii=1:length(tempptypenames)
                if size(PricePathStruct.(PricePathNames{pp}).(tempptypenames{ii}),1)==T % Note: size(PricePathStruct.(PricePathNames{pp}),2) will be 1 or N_j
                    PricePath(:,PricePathSizeVec(1,pp)+(ii-1)*size(PricePathStruct.(PricePathNames{pp}).(tempptypenames{ii}),2):PricePathSizeVec(1,pp)-1+ii*size(PricePathStruct.(PricePathNames{pp}).(tempptypenames{ii}),2))=PricePathStruct.(PricePathNames{pp}).(tempptypenames{ii});
                else % Need to transpose
                    PricePath(:,PricePathSizeVec(1,pp)+(ii-1)*size(PricePathStruct.(PricePathNames{pp}).(tempptypenames{ii}),1):PricePathSizeVec(1,pp)-1+ii*size(PricePathStruct.(PricePathNames{pp}).(tempptypenames{ii}),1))=PricePathStruct.(PricePathNames{pp}).(tempptypenames{ii})';
                end
            end
        else
            if size(PricePathStruct.(PricePathNames{pp}),1)==T
                PricePath(:,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=PricePathStruct.(PricePathNames{pp});
            else % Need to transpose
                PricePath(:,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=PricePathStruct.(PricePathNames{pp})';
            end
        end
    end

    ParamPathNames=fieldnames(ParamPathStruct);
    ParamPathSizeVec=zeros(1,length(ParamPathNames)); % Allows for a given price param to depend on age (or permanent type)
    for pp=1:length(ParamPathNames)
        if isstruct(ParamPathStruct.(ParamPathNames{pp}))
            tempptypenames=fieldnames(PricePathStruct.(PricePathNames{pp}));
            temp=ParamPathStruct.(ParamPathNames{pp}).*(tempptypenames{1});
            tempsize=size(temp);
            ParamPathSizeVec(pp)=length(tempptypenames)*tempsize(tempsize~=T); % Get the dimension which is not T
            for ii=1:N_i
                N_j_temp=N_j.(Names_i{ii});
                if ~any(ParamPathSizeVec(pp)==[1,N_i,N_j_temp,N_i*N_j_temp])
                    error(['ParamPath for ', ParamPathNames{pp}, ' appears to be the wrong size (should be 1-by-T or N_j-by-T or N_i-by-T)'])
                end
            end
        else
            temp=ParamPathStruct.(ParamPathNames{pp});
            tempsize=size(temp);
            ParamPathSizeVec(pp)=tempsize(tempsize~=T); % Get the dimension which is not T
            for ii=1:N_i
                N_j_temp=N_j.(Names_i{ii});
                if ~any(ParamPathSizeVec(pp)==[1,N_i,N_j_temp])
                    error(['ParamPath for ', ParamPathNames{pp}, ' appears to be the wrong size (should be 1-by-T or N_j-by-T or N_i-by-T)'])
                end
            end
        end
    end

    % Also need what size these are conditional on ptype (as some of PricePath/ParamPath may differ by ptype)
    ParamPathSizeVec_ii=ParamPathSizeVec;
    ParamPathSizeVec_ii(ParamPathSizeVec_ii==N_i)=1; % Just use one of a price that depends on ptype
    ParamPathSizeVec_ii=cumsum(ParamPathSizeVec_ii);
    if length(ParamPathNames)>1
        ParamPathSizeVec_ii=[[1,ParamPathSizeVec_ii(1:end-1)+1];ParamPathSizeVec_ii];
    else
        ParamPathSizeVec_ii=[1;ParamPathSizeVec_ii];
    end

    ParamPathSizeVec=cumsum(ParamPathSizeVec);
    if length(ParamPathNames)>1
        ParamPathSizeVec=[[1,ParamPathSizeVec(1:end-1)+1];ParamPathSizeVec];
    else
        ParamPathSizeVec=[1;ParamPathSizeVec];
    end
    ParamPath=zeros(T,ParamPathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
    for pp=1:length(ParamPathNames)
        if isstruct(ParamPathStruct.(ParamPathNames{pp}))
            tempptypenames=fieldnames(ParamPathStruct.(ParamPathNames{pp}));
            for ii=1:length(tempptypenames)
                if size(ParamPathStruct.(ParamPathNames{pp}).(tempptypenames{ii}),1)==T % Note: size(PricePathStruct.(PricePathNames{pp}),2) will be 1 or N_j
                    ParamPath(:,ParamPathSizeVec(1,pp)+(ii-1)*size(ParamPathStruct.(ParamPathNames{pp}).(tempptypenames{ii}),2):ParamPathSizeVec(1,pp)-1+ii*size(ParamPathStruct.(ParamPathNames{pp}).(tempptypenames{ii}),2))=ParamPathStruct.(ParamPathNames{pp}).(tempptypenames{ii});
                else % Need to transpose
                    ParamPath(:,ParamPathSizeVec(1,pp)+(ii-1)*size(ParamPathStruct.(ParamPathNames{pp}).(tempptypenames{ii}),1):ParamPathSizeVec(1,pp)-1+ii*size(ParamPathStruct.(ParamPathNames{pp}).(tempptypenames{ii}),1))=ParamPathStruct.(ParamPathNames{pp}).(tempptypenames{ii})';
                end
            end
        else
            if size(ParamPathStruct.(ParamPathNames{pp}),1)==T
                ParamPath(:,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp))=ParamPathStruct.(ParamPathNames{pp});
            else % Need to transpose
                ParamPath(:,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp))=ParamPathStruct.(ParamPathNames{pp})';
            end
        end
    end

    varargout={PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec,PricePathSizeVec_ii,ParamPathSizeVec_ii};
end


end