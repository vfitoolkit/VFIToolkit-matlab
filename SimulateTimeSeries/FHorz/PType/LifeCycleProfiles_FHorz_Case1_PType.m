function AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, FnsToEvaluateParamNames, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, simoptions)
% Allows for different permanent (fixed) types of agent.
% See ValueFnIter_PType for general idea.
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
% depend on permanent type and inputted as vectors or matrices as appropriate; note that this cannot be done for 
% vfoptions, simoptions, etc as it then becomes impossible to tell that the vector/matrix is because of PType and not something else)
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
AgeConditionalStats=struct();
FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i,'gpuArray');

% Set default of grouping all the PTypes together when reporting statistics
if exist('simoptions','var')
    if ~isfield(simoptions,'groupptypesforstats')
       simoptions.groupptypesforstats=1;
    end
else
    simoptions.groupptypesforstats=1;
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
            sprintf('Possible Warning: Number of columns of n_d is the same as the number of permanent types. \n This may just be coincidence as number of d variables is equal to number of permanent types. \n If they are intended to be permanent types then n_d should have them as different rows (not columns). \n')
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
            sprintf('Possible Warning: Number of columns of n_a is the same as the number of permanent types. \n This may just be coincidence as number of a variables is equal to number of permanent types. \n If they are intended to be permanent types then n_a should have them as different rows (not columns). \n')
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
            sprintf('Possible Warning: Number of columns of n_z is the same as the number of permanent types. \n This may just be coincidence as number of z variables is equal to number of permanent types. \n If they are intended to be permanent types then n_z should have them as different rows (not columns). \n')
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
                sprintf('Possible Warning: some parameters appear to have been imputted with dependence on permanent type indexed by column rather than row \n')
                sprintf(['Specifically, parameter: ', FullParamNames{kField}, ' \n'])
                sprintf('(it is possible this is just a coincidence of number of columns) \n')
                dbstack
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
    [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate, FnsToEvaluateParamNames,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;
%     % Figure out which functions are actually relevant to the present
%     % PType. Only the relevant ones need to be evaluated.
%     % The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are
%     % necessarily the same.
%     FnsToEvaluate_temp={};
%     FnsToEvaluateParamNames_temp=struct(); %(1).Names={}; % This is just an initialization value and will be overwritten
%     WhichFnsForCurrentPType=zeros(numFnsToEvaluate,1);
%     if isstruct(FnsToEvaluate)
%         % Just conver from struct into the FnsToEvaluate_temp and FnsToEvaluateParamNames_temp format now.
%         FnNames=fieldnames(FnsToEvaluate);
%         FnsToEvaluateParamNames_temp=[]; % Ignore, is filled in by subcodes
%         jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
%         for kk=1:numFnsToEvaluate
%             % Note: I keep it as a structure to avoid having to find the input names here (which needs to allow for things like using n_e)
%             if isa(FnsToEvaluate.(FnNames{kk}),'struct')
%                 if isfield(FnsToEvaluate.(FnNames{kk}), Names_i{ii})
%                     FnsToEvaluate_temp.(FnNames{kk})=FnsToEvaluate.(FnNames{kk}).(Names_i{ii});
% %                     FnsToEvaluate_temp{jj}=FnsToEvaluate.(FnNames{kk}).(Names_i{ii});
% %                     FnsToEvaluateParamNames_temp(jj).Names=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{kk}).(Names_i{ii}));
%                     WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
%                     % else
%                     %  % do nothing as this FnToEvaluate is not relevant for the current PType
%                     %  % Implicitly, WhichFnsForCurrentPType(kk)=0
%                     FnsAndPTypeIndicator(kk,ii)=1;
%                 end
%             else
%                 % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
%                     FnsToEvaluate_temp.(FnNames{kk})=FnsToEvaluate.(FnNames{kk});
% %                 FnsToEvaluate_temp{jj}=FnsToEvaluate.(FnNames{kk});
% %                 FnsToEvaluateParamNames_temp(jj).Names=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{kk}));
%                 WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
%                 FnsAndPTypeIndicator(kk,ii)=1;
%             end
%         end
%     else
%         jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
%         for kk=1:numFnsToEvaluate
%             if isa(FnsToEvaluate{kk},'struct')
%                 if isfield(FnsToEvaluate{kk}, Names_i{ii})
%                     FnsToEvaluate_temp{jj}=FnsToEvaluate{kk}.(Names_i{ii});
%                     if isa(FnsToEvaluateParamNames(kk).Names,'struct')
%                         FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(kk).Names.(Names_i{ii});
%                     else
%                         FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(kk).Names;
%                     end
%                     WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
%                     % else
%                     %  % do nothing as this FnToEvaluate is not relevant for the current PType
%                     % % Implicitly, WhichFnsForCurrentPType(kk)=0
%                     FnsAndPTypeIndicator(kk,ii)=1;
%                 end
%             else
%                 % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
%                 FnsToEvaluate_temp{jj}=FnsToEvaluate{kk};
%                 FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(kk).Names;
%                 WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
%                 FnsAndPTypeIndicator(kk,ii)=1;
%             end
%         end
%     end
    
    if simoptions.groupptypesforstats==0
        simoptions_temp.keepoutputasmatrix=0;
        AgeConditionalStats_ii=LifeCycleProfiles_FHorz_Case1(StationaryDist_temp,PolicyIndexes_temp,FnsToEvaluate_temp,FnsToEvaluateParamNames_temp,Parameters_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,z_grid_temp,simoptions_temp);
        %     PTypeWeight_ii=StationaryDist.ptweights(ii);
        
        AggVarNames_temp=fieldnames(FnsToEvaluate);
        for kk=1:numFnsToEvaluate
            jj=WhichFnsForCurrentPType(kk);
            if jj>0
%                 AgeConditionalStats.(Names_i{ii})(kk)=AgeConditionalStats_ii.(AggVarNames_temp{jj});
                AgeConditionalStats.(AggVarNames_temp{jj}).(Names_i{ii})=AgeConditionalStats_ii.(AggVarNames_temp{jj});
            end
        end
    else % simoptions.groupptypesforstats==1
        simoptions_temp.keepoutputasmatrix=1;
        ValuesOnGrid_ii=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp, simoptions_temp);
        N_a_temp=prod(n_a_temp);
        if isfield(simoptions_temp,'n_e')
            n_z_temp=[n_z_temp,simoptions_temp.n_e];
            N_z_temp=prod(n_z_temp);
            ValuesOnDist_Kron=zeros(N_a_temp*N_z_temp*N_j_temp,1);
            for kk=1:numFnsToEvaluate
                jj=WhichFnsForCurrentPType(kk);
                if jj>0
                    ValuesOnDist_Kron=reshape(ValuesOnGrid_ii(jj,:,:,:,:),[N_a_temp*N_z_temp,N_j_temp]);
                end
                ValuesOnDist.(Names_i{ii}).(['k',num2str(kk)])=ValuesOnDist_Kron;
            end
        else
            N_z_temp=prod(n_z_temp);
            ValuesOnDist_Kron=zeros(N_a_temp*N_z_temp*N_j_temp,1);
            for kk=1:numFnsToEvaluate
                jj=WhichFnsForCurrentPType(kk);
                if jj>0
                    ValuesOnDist_Kron=reshape(ValuesOnGrid_ii(jj,:,:,:),[N_a_temp*N_z_temp,N_j_temp]);
                end
                ValuesOnDist.(Names_i{ii}).(['k',num2str(kk)])=ValuesOnDist_Kron;
            end
        end
        
        % I can write over StationaryDist.(Names_i{ii}) as I don't need it
        % again, but I do need the reshaped and reweighed version in the next for loop.
        StationaryDist.(Names_i{ii})=reshape(StationaryDist.(Names_i{ii}).*StationaryDist.ptweights(ii),[N_a_temp*N_z_temp,N_j_temp]);
    end
end

%% NOTE GROUPING ONLY WORKS IF THE GRIDS ARE THE SAME SIZES FOR EACH AGENT (for whom a given FnsToEvaluate is being calculated)
% (mainly because otherwise would have to deal with simoptions.agegroupings being different for each agent and this requires more complex code)
% Will throw an error if this is not the case

% If grouping, we have ValuesOnDist and StationaryDist that contain
% everythign we will need. Now we just have to compute them.
if simoptions.groupptypesforstats==1
    % Note that I do not currently allow the following simoptions to differ by PType
    if isfield(simoptions,'nquantiles')==0
        simoptions.nquantiles=20; % by default gives ventiles
    end
    if isfield(simoptions,'agegroupings')==0
        simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    end
    if isfield(simoptions,'npoints')==0
        simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    end
    if isfield(simoptions,'tolerance')==0    
        simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
    
    ngroups=length(simoptions.agegroupings);
    % Do some preallocation of the output structure
    AgeConditionalStats(length(FnsToEvaluate)).Mean=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).Median=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).Variance=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(simoptions.npoints,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(simoptions.nquantiles+1,ngroups,'gpuArray'); % Includes the min and max values
    AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(simoptions.nquantiles,ngroups,'gpuArray');
    
    if isstruct(FnsToEvaluate)
        numFnsToEvaluate=length(fieldnames(FnsToEvaluate));
    else
        numFnsToEvaluate=length(FnsToEvaluate);
    end
    
    for kk=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
        
        N_i_kk=sum(FnsAndPTypeIndicator(kk,:)); % How many agents is this statistic calculated for
        StationaryDistVec_kk=zeros(N_a_temp*N_z_temp,N_j_temp,N_i_kk,'gpuArray');
        Values_kk=zeros(N_a_temp*N_z_temp,N_j_temp,N_i_kk,'gpuArray');
        
        for ii=1:N_i_kk
            StationaryDistVec_kk(:,:,ii)=StationaryDist.(Names_i{ii}); % Note, has already been multiplied by StationaryDist.ptweights(ii)
            Values_kk(:,:,ii)=ValuesOnDist.(Names_i{ii}).(['k',num2str(kk)]);
        end
        
        for jj=1:length(simoptions.agegroupings)
            j1=simoptions.agegroupings(jj);
            if jj<length(simoptions.agegroupings)
                jend=simoptions.agegroupings(jj+1)-1;
            else
                jend=N_j;
            end
            StationaryDistVec_jj=reshape(StationaryDistVec_kk(:,j1:jend,:),[N_a_temp*N_z_temp*(jend-j1+1)*N_i_kk,1]);
            StationaryDistVec_jj=StationaryDistVec_jj./sum(StationaryDistVec_jj); % Normalize to sum to one for this 'agegrouping'
            
            Values_jj=reshape(Values_kk(:,j1:jend,:),[N_a_temp*N_z_temp*(jend-j1+1)*N_i,1]);
            
            [SortedValues,SortedValues_index] = sort(Values_jj);
            
            SortedWeights = StationaryDistVec_jj(SortedValues_index);
            CumSumSortedWeights=cumsum(SortedWeights);
            
            WeightedValues=Values_jj.*StationaryDistVec_jj;
            SortedWeightedValues=WeightedValues(SortedValues_index);
            
            % Calculate the 'age conditional' mean
            AgeConditionalStats(kk).Mean(jj)=sum(WeightedValues);
            % Calculate the 'age conditional' median
            AgeConditionalStats(kk).Median(jj)=SortedWeightedValues(max(1,floor(0.5*length(SortedWeightedValues)))); % The max is just to deal with 'corner' case where there is only one element in SortedWeightedValues
            % Calculate the 'age conditional' variance
            AgeConditionalStats(kk).Variance(jj)=sum((Values_jj.^2).*StationaryDistVec_jj)-(AgeConditionalStats(kk).Mean(jj))^2; % Weighted square of values - mean^2
            
            
            if simoptions.npoints>0
                % Calculate the 'age conditional' lorenz curve
                AgeConditionalStats(kk).LorenzCurve(:,jj)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,simoptions.npoints,2);
                % Calculate the 'age conditional' gini
                AgeConditionalStats(kk).Gini(jj)=Gini_from_LorenzCurve(AgeConditionalStats(kk).LorenzCurve(:,jj));
            end
            
            % Calculate the 'age conditional' quantile means (ventiles by default)
            % Calculate the 'age conditional' quantile cutoffs (ventiles by default)
            QuantileIndexes=zeros(1,simoptions.nquantiles-1,'gpuArray');
            QuantileCutoffs=zeros(1,simoptions.nquantiles-1,'gpuArray');
            QuantileMeans=zeros(1,simoptions.nquantiles,'gpuArray');
            
            for ll=1:simoptions.nquantiles-1
                tempindex=find(CumSumSortedWeights>=ll/simoptions.nquantiles,1,'first');
                QuantileIndexes(ll)=tempindex;
                QuantileCutoffs(ll)=SortedValues(tempindex);
                if ll==1
                    QuantileMeans(ll)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
                elseif ll<(simoptions.nquantiles-1) % (1<ll) &&
                    QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                else %if ll==(options.nquantiles-1)
                    QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                    QuantileMeans(ll+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
                end
            end
            % Min value
            tempindex=find(CumSumSortedWeights>=simoptions.tolerance,1,'first');
            minvalue=SortedValues(tempindex);
            % Max value
            tempindex=find(CumSumSortedWeights>=(1-simoptions.tolerance),1,'first');
            maxvalue=SortedValues(tempindex);
            AgeConditionalStats(kk).QuantileCutoffs(:,jj)=[minvalue, QuantileCutoffs, maxvalue]';
            AgeConditionalStats(kk).QuantileMeans(:,jj)=QuantileMeans';
            
        end
    end

end


%%
if isstruct(FnsToEvaluate) && simoptions.groupptypesforstats==1
    % Change the output into a structure
    AgeConditionalStats2=AgeConditionalStats;
    clear AgeConditionalStats
    AgeConditionalStats=struct();
    AggVarNames=fieldnames(FnsToEvaluate);
%     if simoptions.groupptypesforstats==0
        % Do nothing
%         for ii=1:N_i
%             for ff=1:length(AggVarNames)
%                 AgeConditionalStats.(AggVarNames{ff}).(Names_i{ii}).Mean=AgeConditionalStats2(ff).(Names_i{ii}).Mean;
%                 AgeConditionalStats.(AggVarNames{ff}).(Names_i{ii}).Median=AgeConditionalStats2(ff).(Names_i{ii}).Median;
%                 AgeConditionalStats.(AggVarNames{ff}).(Names_i{ii}).Variance=AgeConditionalStats2(ff).(Names_i{ii}).Variance;
%                 AgeConditionalStats.(AggVarNames{ff}).(Names_i{ii}).LorenzCurve=AgeConditionalStats2(ff).(Names_i{ii}).LorenzCurve;
%                 AgeConditionalStats.(AggVarNames{ff}).(Names_i{ii}).Gini=AgeConditionalStats2(ff).(Names_i{ii}).Gini;
%                 AgeConditionalStats.(AggVarNames{ff}).(Names_i{ii}).QuantileCutoffs=AgeConditionalStats2(ff).(Names_i{ii}).QuantileCutoffs;
%                 AgeConditionalStats.(AggVarNames{ff}).(Names_i{ii}).QuantileMeans=AgeConditionalStats2(ff).(Names_i{ii}).QuantileMeans;
%             end
%         end
%     else % simoptions.groupptypesforstats==1
        for ff=1:length(AggVarNames)
            AgeConditionalStats.(AggVarNames{ff}).Mean=AgeConditionalStats2(ff).Mean;
            AgeConditionalStats.(AggVarNames{ff}).Median=AgeConditionalStats2(ff).Median;
            AgeConditionalStats.(AggVarNames{ff}).Variance=AgeConditionalStats2(ff).Variance;
            AgeConditionalStats.(AggVarNames{ff}).LorenzCurve=AgeConditionalStats2(ff).LorenzCurve;
            AgeConditionalStats.(AggVarNames{ff}).Gini=AgeConditionalStats2(ff).Gini;
            AgeConditionalStats.(AggVarNames{ff}).QuantileCutoffs=AgeConditionalStats2(ff).QuantileCutoffs;
            AgeConditionalStats.(AggVarNames{ff}).QuantileMeans=AgeConditionalStats2(ff).QuantileMeans;
        end
%     end
end



end
