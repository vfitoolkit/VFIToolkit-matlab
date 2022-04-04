function MeanMedianStdDev=EvalFnOnAgentDist_MeanMedianStdDev_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, simoptions)
% Allows for different permanent (fixed) types of agent.
% See ValueFnIter_PType for general idea.
%
%
% simoptions.verbose=1 will give feedback
% simoptions.verboseparams=1 will give further feedback on the param values of each permanent type
%
% Rest of this description describes how those inputs not already used for
% ValueFnIter_PType or StationaryDist_PType should be set up.
%
% jequaloneDist can either be same for all permanent types, or must be passed as a structure.
% AgeWeightParamNames is either same for all permanent types, or must be passed as a structure.
%
% The stationary distribution be a structure and will contain both the
% weights/distribution across the permenant types, as well as a pdf for the
% stationary distribution of each specific permanent type.
%
% How exactly to handle these differences between permanent (fixed) types
% is to some extent left to the user. You can, for example, input
% parameters that differ by permanent type as a vector with different rows f
% for each type, or as a structure with different fields for each type.
%
% Any input that does not depend on the permanent type is just passed in
% exactly the same form as normal.

% Names_i can either be a cell containing the 'names' of the different
% permanent types, or if there are no structures used (just parameters that
% depend on permanent type and inputted as vectors or matrices as appropriate)
% then Names_i can just be the number of permanent types (but does not have to be, can still be names).
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
    Names_i={'ptype001'};
    for ii=2:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end


if isstruct(FnsToEvaluate)
    numFnsToEvaluate=length(fieldnames(FnsToEvaluate));
else
    numFnsToEvaluate=length(FnsToEvaluate);
end
MeanMedianStdDev=zeros(numFnsToEvaluate,3,'gpuArray'); % The 3 are: mean, median, and standard deviation, respecitively
FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i,'gpuArray');

PTypeMeans=zeros(numFnsToEvaluate,N_i);
PTypeMedians=zeros(numFnsToEvaluate,N_i);
PTypeStdDev=zeros(numFnsToEvaluate,N_i);

% Set default of grouping all the PTypes together when reporting statistics
if ~exist('simoptions','var')
    simoptions.groupptypesforstats=1;
else
    if ~isfield(simoptions,'groupptypesforstats')
       simoptions.groupptypesforstats=1;
    end
end

%%
for ii=1:N_i
    % First set up simoptions
    if exist('simoptions','var')
        simoptions_temp=PType_Options(simoptions,Names_i,ii);
        if ~isfield(simoptions_temp,'verbose')
            simoptions_temp.verbose=0;
        end
        if ~isfield(simoptions_temp,'verboseparams')
            simoptions_temp.verboseparams=0;
        end
    else
        simoptions_temp.verbose=0;
        simoptions_temp.verboseparams=0;
    end
    
    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end
        
    PolicyIndexes_temp=Policy.(Names_i{ii});
    StationaryDist_temp=StationaryDist.(Names_i{ii});
    if isa(StationaryDist_temp, 'gpuArray')
        Parallel_temp=2;
    else
        Parallel_temp=1;
    end
    
    % Go through everything which might be dependent on permanent type (PType)
    % Notice that the way this is coded the grids (etc.) could be either
    % fixed, or a function (that depends on age, and possibly on permanent
    % type), or they could be a structure. Only in the case where they are
    % a structure is there a need to take just a specific part and send
    % only that to the 'non-PType' version of the command.
    
    % Start with those that determine whether the current permanent type is finite or
    % infinite horizon, and whether it is Case 1 or Case 2
    % Figure out which case is relevant to the current PType. This is done
    % using N_j which for the current type will evaluate to 'Inf' if it is
    % infinite horizon and a finite number for any other finite horizon.
    % First, check if it is a structure, and otherwise just get the
    % relevant value.
    
    % Horizon is determined via N_j
    if isstruct(N_j)
        N_j_temp=N_j.(Names_i{ii});
    elseif isscalar(N_j)
        N_j_temp=N_j;
    else % is a vector
        N_j_temp=N_j(ii);
    end
    
    n_d_temp=n_d;
    if isa(n_d,'struct')
        n_d_temp=n_d.(Names_i{ii});
    else
        temp=size(n_d);
        if temp(1)>1 % n_d depends on fixed type
            n_d_temp=n_d(ii,:);
        elseif temp(2)==N_i % If there is one row, but number of elements in n_d happens to coincide with number of permanent types, then just let user know
            fprintf('Possible Warning: Number of columns of n_d is the same as the number of permanent types. \n This may just be coincidence as number of d variables is equal to number of permanent types. \n If they are intended to be permanent types then n_d should have them as different rows (not columns). \n')
            dbstack
        end
    end
    n_a_temp=n_a;
    if isa(n_a,'struct')
        n_a_temp=n_a.(Names_i{ii});
    else
        temp=size(n_a);
        if temp(1)>1 % n_a depends on fixed type
            n_a_temp=n_a(ii,:);
        elseif temp(2)==N_i % If there is one row, but number of elements in n_a happens to coincide with number of permanent types, then just let user know
            fprintf('Possible Warning: Number of columns of n_a is the same as the number of permanent types. \n This may just be coincidence as number of a variables is equal to number of permanent types. \n If they are intended to be permanent types then n_a should have them as different rows (not columns). \n')
            dbstack
        end
    end
    n_z_temp=n_z;
    if isa(n_z,'struct')
        n_z_temp=n_z.(Names_i{ii});
    else
        temp=size(n_z);
        if temp(1)>1 % n_z depends on fixed type
            n_z_temp=n_z(ii,:);
        elseif temp(2)==N_i % If there is one row, but number of elements in n_d happens to coincide with number of permanent types, then just let user know
            fprintf('Possible Warning: Number of columns of n_z is the same as the number of permanent types. \n This may just be coincidence as number of z variables is equal to number of permanent types. \n If they are intended to be permanent types then n_z should have them as different rows (not columns). \n')
            dbstack
        end
    end
    

    if isa(d_grid,'struct')
        d_grid_temp=d_grid.(Names_i{ii});
    else
        d_grid_temp=d_grid;
    end
    if isa(a_grid,'struct')
        a_grid_temp=a_grid.(Names_i{ii});
    else
        a_grid_temp=a_grid;
    end
    if isa(z_grid,'struct')
        z_grid_temp=z_grid.(Names_i{ii});
    else
        z_grid_temp=z_grid;
    end
    
    % Parameters are allowed to be given as structure, or as vector/matrix
    % (in terms of their dependence on permanent type). So go through each of
    % these in term.
    % ie. Parameters.alpha=[0;1]; or Parameters.alpha.ptype1=0; Parameters.alpha.ptype2=1;
    Parameters_temp=Parameters;
    FullParamNames=fieldnames(Parameters); % all the different parameters
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check the current parameter for permanent type in structure form
            % Check if this parameter is used for the current permanent type (it may or may not be, some parameters are only used be a subset of permanent types)
            if isfield(Parameters.(FullParamNames{kField}),Names_i{ii})
                Parameters_temp.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(Names_i{ii});
            end
        elseif sum(size(Parameters.(FullParamNames{kField}))==N_i)>=1 % Check for permanent type in vector/matrix form.
            temp=Parameters.(FullParamNames{kField});
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType, it should be the row dimension, if it is not then give a warning.
            if ptypedim==1
                Parameters_temp.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                Parameters_temp.(FullParamNames{kField})=temp(:,ii);
            end
        end
    end
    % THIS TREATMENT OF PARAMETERS COULD BE IMPROVED TO BETTER DETECT INPUT SHAPE ERRORS.
    
    if simoptions_temp.verboseparams==1
        fprintf('Parameter values for the current permanent type \n')
        Parameters_temp
    end
    
    % Figure out which functions are actually relevant to the present PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluate and FnsToEvaluateFnParamNames are necessarily the same.
    % Allows for FnsToEvaluate as structure.
    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=1;
    end
    l_a_temp=length(n_a_temp);
    l_z_temp=length(n_z_temp);  
    [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;
%     % Figure out which functions are actually relevant to the present
%     % PType. Only the relevant ones need to be evaluated.
%     % The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are
%     % necessarily the same.
%     FnsToEvaluate_temp={};
%     FnsToEvaluateParamNames_temp=struct(); %(1).Names={}; % This is just an initialization value and will be overwritten
%     WhichFnsForCurrentPType=zeros(numFnsToEvaluate,1);
%     jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
%     for kk=1:numFnsToEvaluate
%         if isa(FnsToEvaluate{kk},'struct')
%             if isfield(FnsToEvaluate{kk}, Names_i{ii})
%                 FnsToEvaluate_temp{jj}=FnsToEvaluate{kk}.(Names_i{ii});
%                 if isa(FnsToEvaluateParamNames(kk).Names,'struct')
%                     FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(kk).Names.(Names_i{ii});
%                 else
%                     FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(kk).Names;
%                 end
%                 WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
%                 % else
%                 %  % do nothing as this FnToEvaluate is not relevant for the current PType
%                 % % Implicitly, WhichFnsForCurrentPType(kk)=0
%                 FnsAndPTypeIndicator(kk,ii)=1;
%             end
%         else
%             % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
%             FnsToEvaluate_temp{jj}=FnsToEvaluate{kk};
%             FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(kk).Names;
%             WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
%             FnsAndPTypeIndicator(kk,ii)=1;
%         end
%     end
    
    simoptions_temp.keepoutputasmatrix=1;
    MeanMedianStdDev_ii=EvalFnOnAgentDist_MeanMedianStdDev_FHorz_Case1(StationaryDist_temp,PolicyIndexes_temp, FnsToEvaluate_temp,Parameters_temp,FnsToEvaluateParamNames_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,z_grid_temp,Parallel_temp,simoptions_temp);
        
    % Store the means and the std devs of each type so as these can later be used to calculate the standard deviation
    for kk=1:numFnsToEvaluate
        jj=WhichFnsForCurrentPType(kk);
        if jj>0
            PTypeMeans(kk,ii)=MeanMedianStdDev_ii(jj,1);
            PTypeMedians(kk,ii)=MeanMedianStdDev_ii(jj,2);
            PTypeStdDev(kk,ii)=MeanMedianStdDev_ii(jj,3);
        end
    end
    
    if simoptions.groupptypesforstats==1
        ValuesOnGrid_ii=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp, simoptions_temp);
        
        if isfield(simoptions_temp,'n_e')
            n_z_temp=[n_z_temp,simoptions.n_e];
        end        
        N_a_temp=prod(n_a_temp);
        N_z_temp=prod(n_z_temp);
        ValuesOnGrid_Kron=zeros(N_a_temp*N_z_temp*N_j_temp,1);
        for kk=1:numFnsToEvaluate
            jj=WhichFnsForCurrentPType(kk);
            if jj>0
                ValuesOnGrid_Kron=reshape(ValuesOnGrid_ii(jj,:,:,:),[N_a_temp*N_z_temp*N_j_temp,1]);
            end
            ValuesOnGrid.(Names_i{ii}).(['k',num2str(kk)])=ValuesOnGrid_Kron;
        end
        
        % I can write over StationaryDist.(Names_i{ii}) as I don't need it
        % again, but I do need the reshaped and reweighed version in the next for loop.
        StationaryDist.(Names_i{ii})=reshape(StationaryDist.(Names_i{ii}).*StationaryDist.ptweights(ii),[N_a_temp*N_z_temp*N_j_temp,1]);
    end
    
%     % Caculating the median is more complicated as it cannot be expressed
%     % as a combination of the medians for each type.
%     LorenzCurve_ii=EvalFnOnAgentDist_LorenzCurve_FHorz_Case1(StationaryDist_temp,PolicyIndexes_temp, FnsToEvaluate_temp,Parameters_temp,FnsToEvaluateParamNames_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,z_grid_temp,Parallel_temp,npoints,simoptions_temp);
%     % Returns a Lorenz Curve npoints-by-1.
%     %
%     % Note that to unnormalize the Lorenz Curve you can just multiply it be the AggVars for the same variable. This will give you the inverse cdf.
%     
%     LorenzCurve(:,:,ii)=LorenzCurve_ii'; % kk by npoints
    
end


if simoptions.groupptypesforstats==1
    % Calculate the mean and standard deviation
    % (Formula: https://en.wikipedia.org/wiki/Pooled_variance#Aggregation_of_standard_deviation_data )
    % Calculate the median from the ValuesOnGrid
    for kk=1:numFnsToEvaluate
        SigmaNxi=sum(FnsAndPTypeIndicator(kk,:).*(StationaryDist.ptweights)'); % The sum of the masses of the relevant types
        
        % Mean
        MeanMedianStdDev(kk,1)=sum(FnsAndPTypeIndicator(kk,:).*(StationaryDist.ptweights').*PTypeMeans(kk,:))/SigmaNxi;
        
        % Standard Deviation
        if N_i==1
            MeanMedianStdDev(kk,3)=PTypeStdDev(kk,:);
        else
            temp2=zeros(N_i,1);
            for ii=2:N_i
                if FnsAndPTypeIndicator(kk,ii)==1
                    %             temp=StationaryDist.ptweights(ii)*FnsAndPTypeIndicator(kk,1:(ii-1)).*StationaryDist.ptweights(1:(ii-1)).*((PTypeMeans(kk,1:(ii-1))-PTypeMeans(kk,ii)).^2);
                    %             tempA=StationaryDist.ptweights(ii);
                    %             tempB=FnsAndPTypeIndicator(kk,1:(ii-1));
                    %             tempC=StationaryDist.ptweights(1:(ii-1));
                    %             tempD=((PTypeMeans(kk,1:(ii-1))-PTypeMeans(kk,ii)).^2);
                    %             [kk,ii]
                    %             tempA
                    %             tempB
                    %             tempC
                    %             tempD
                    %             size(temp)
                    temp2(ii)=StationaryDist.ptweights(ii)*sum(FnsAndPTypeIndicator(kk,1:(ii-1)).*(StationaryDist.ptweights(1:(ii-1))').*((PTypeMeans(kk,1:(ii-1))-PTypeMeans(kk,ii)).^2));
                end
            end
            MeanMedianStdDev(kk,3)=sqrt(sum(FnsAndPTypeIndicator(kk,:).*(StationaryDist.ptweights').*PTypeStdDev(kk,:))/SigmaNxi + sum(temp2)/(SigmaNxi^2));
        end
        
        % Median
        DistVec=[];
        ValuesVec=[];
        for ii=1:N_i
            if FnsAndPTypeIndicator(kk,ii)==1
                % Note, gather these  two, as otherwise they become a gpu memory bottleneck
                DistVec=[DistVec; gather(StationaryDist.(Names_i{ii})/SigmaNxi)]; % Note: StationaryDist.(Names_i{ii}) was overwritten in the main for-loop, it is actually =reshape(StationaryDist.(Names_i{ii}).*StationaryDist.ptweights(ii),[N_a_temp*N_z_temp*N_j_temp,1])
                ValuesVec=[ValuesVec;gather(ValuesOnGrid.(Names_i{ii}).(['k',num2str(kk)]))];
            end
        end
        [SortedValues,sortindex]=sort(ValuesVec);
        SortedDist=DistVec(sortindex);
        median_index=find(cumsum(SortedDist)>=0.5,1,'first');
        
        MeanMedianStdDev(kk,2)=SortedValues(median_index);
        
    end
else
    MeanMedianStdDev=zeros(numFnsToEvaluate,3,N_i);
    for kk=1:numFnsToEvaluate
        for ii=1:N_i
            MeanMedianStdDev(kk,1,ii)=PTypeMeans(kk,ii);
            MeanMedianStdDev(kk,2,ii)=PTypeMedians(kk,ii);
            MeanMedianStdDev(kk,3,ii)=PTypeStdDev(kk,ii);
        end
    end
end


% If using FnsToEvaluate as structure need to get in appropriate form for output
if isstruct(FnsToEvaluate)
    AggVarNames=fieldnames(FnsToEvaluate);
    % Change the output into a structure
    MeanMedianStdDev2=MeanMedianStdDev;
    clear MeanMedianStdDev
    MeanMedianStdDev=struct();
    %     AggVarNames=fieldnames(FnsToEvaluate);
    if simoptions.groupptypesforstats==1
        for ff=1:length(AggVarNames)
            MeanMedianStdDev.(AggVarNames{ff}).Mean=MeanMedianStdDev2(ff,1);
            MeanMedianStdDev.(AggVarNames{ff}).Median=MeanMedianStdDev2(ff,2);
            MeanMedianStdDev.(AggVarNames{ff}).StdDev=MeanMedianStdDev2(ff,3);
        end
    else % simoptions.groupptypesforstats==0
        for ff=1:length(AggVarNames)
            for ii=1:N_i
                MeanMedianStdDev.(AggVarNames{ff}).(Names_i{ii}).Mean=MeanMedianStdDev2(ff,1,ii);
                MeanMedianStdDev.(AggVarNames{ff}).(Names_i{ii}).Median=MeanMedianStdDev2(ff,2,ii);
                MeanMedianStdDev.(AggVarNames{ff}).(Names_i{ii}).StdDev=MeanMedianStdDev2(ff,3,ii);
            end
        end
    end
end


end
