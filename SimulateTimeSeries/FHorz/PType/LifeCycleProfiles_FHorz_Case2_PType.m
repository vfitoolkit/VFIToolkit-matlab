function AgeConditionalStats=LifeCycleProfiles_FHorz_Case2_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, simoptions)
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

computeForThesei=ones(N_i,1); % Used to omit the infinite horizon PTypes from computations (a message is printed to say they are being ignored)

% Set default of grouping all the PTypes together when reporting statistics
if ~exist('simoptions','var')
    simoptions.groupptypesforstats=1; % Group all the PTypes together when reporting statistics (individual ones are always reported, this option is to also report the grouped stats)
    simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    simoptions.verbose=0;
    simoptions.verboseparams=0;
    simoptions.nquantiles=20; % by default gives ventiles
    defaultagegroupings=1;
    if isstruct(N_j)
        for ii=1:N_i
            if isfinite(N_j.(Names_i{ii}))
                simoptions.agegroupings.(Names_i{ii})=1:1:N_j.(Names_i{ii});
            else % Infinite horizon
                computeForThesei(ii)=0;
            end
        end
    else
        simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    end
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.agejshifter=0; % Use when different PTypes have different initial ages (will be a structure when actually used)
else
    if ~isfield(simoptions,'groupptypesforstats')
        simoptions.groupptypesforstats=1;  % Group all the PTypes together when reporting statistics (individual ones are always reported, this option is to also report the grouped stats)
    end
    if ~isfield(simoptions,'ptypestorecpu')
        if simoptions.groupptypesforstats==1
            simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
        elseif simoptions.groupptypesforstats==0
            simoptions.ptypestorecpu=0;
        end
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=100;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=100;
    end
    if isfield(simoptions,'nquantiles')==0
        simoptions.nquantiles=20; % by default gives ventiles
    end
    if isfield(simoptions,'agegroupings')==0
        defaultagegroupings=1;
        if isstruct(N_j)
            for ii=1:N_i
                if isfinite(N_j.(Names_i{ii}))
                    simoptions.agegroupings.(Names_i{ii})=1:1:N_j.(Names_i{ii});
                else % Infinite horizon
                    computeForThesei(ii)=0;
                end
            end
        else
            simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
        end
    else
        defaultagegroupings=0;
    end
    if isfield(simoptions,'npoints')==0
        simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    elseif simoptions.npoints==0
        error('simoptions.npoints must be a positive (non-zero) integer')
    end
    if isfield(simoptions,'tolerance')==0    
        simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
    if isfield(simoptions,'agejshifter')==0        
        simoptions.agejshifter=0; % Use when different PTypes have different initial ages (will be a structure when actually used)
    end
end

if isstruct(simoptions.agegroupings)
    ngroups=zeros(N_i,1);
    for ii=1:N_i
        ngroups(ii)=length(simoptions.agegroupings.(Names_i{ii}));
    end
else
    ngroups=length(simoptions.agegroupings)*ones(N_i,1);
end
maxngroups=max(ngroups(isfinite(ngroups)));
if isstruct(simoptions.agejshifter) % if using agejshifter
    tempagejshifter=simoptions.agejshifter;
    simoptions=rmfield(simoptions,'agejshifter');
    simoptions.agejshifter=zeros(N_i,1);
    for ii=1:N_i
        simoptions.agejshifter(ii)=tempagejshifter.(Names_i{ii});
    end
    simoptions.agejshifter=simoptions.agejshifter-min(simoptions.agejshifter); % put them all relative to the minimum
elseif length(simoptions.agejshifter)==1 % not using agejshifter
    simoptions.agejshifter=zeros(N_i,1);
else % have inputed as a vector
    simoptions.agejshifter=simoptions.agejshifter-min(simoptions.agejshifter); % put them all relative to the minimum
end
% You cannot use agejshifter together with any age grouping other than just every period
if max(simoptions.agejshifter)>0 && defaultagegroupings==0
    error('You cannot use agejshifter together with any age grouping other than the default (each period seperately)')
end




%% Drop anything that is infinite horizon and print out a message to say so
if any(computeForThesei==0)
    N_i=sum(computeForThesei);
    Names_i2=Names_i;
    Names_i=cell(N_i,1);
    ii=0;
    for ii2=1:length(computeForThesei)
        if computeForThesei(ii2)==1
            ii=ii+1;
            Names_i{ii}=Names_i2{ii2};
        else % tell the user about it
            fprintf(['LifeCycleProfiles_FHorz_Case2_PType: Ignoring the ', num2str(ii2), '-th PType, ',Names_i2{ii2}, ' because it is infinite horizon \n']);
        end
    end   
    % Eliminate any no longer relevant functions from FnsToEvaluate (those which are only used for infinite horizon)
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    FnsToEvaluate2=FnsToEvaluate;
    clear FnsToEvaluate
    for ff=1:length(fieldnames(FnsToEvaluate2))
        if isstruct(FnsToEvaluate2.(FnsToEvalNames{ff}))
            for ii=1:N_i
                if isfield(FnsToEvaluate2.(FnsToEvalNames{ff}),Names_i{ii})
                    FnsToEvaluate.(FnsToEvalNames{ff}).(Names_i{ii})=FnsToEvaluate2.(FnsToEvalNames{ff}).(Names_i{ii});
                end
            end
        else % Relevant to all the PTypes
            FnsToEvaluate.(FnsToEvalNames{ff})=FnsToEvaluate2.(FnsToEvalNames{ff});
        end
    end
    % Done. Because from here on we just use N_i and Names_i which now only
    % contain finite horizons. Note that it is anyway only possible to use
    % a mixture of infinite and finite horizon if you are explictly using
    % Names_i. So don't need to worry about vectors over N_i being the
    % wrong size.
end

%%
if isstruct(FnsToEvaluate)
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnsToEvalNames);
else
    error('You can only use PType when FnsToEvaluate is a structure')
end

% Preallocate a few things
minvaluevec=nan(N_i,1);
maxvaluevec=nan(N_i,1);

%% NOTE GROUPING ONLY WORKS IF THE GRIDS ARE THE SAME SIZES FOR EACH AGENT (for whom a given FnsToEvaluate is being calculated)
% (mainly because otherwise would have to deal with simoptions.agegroupings being different for each agent and this requires more complex code)
% Will throw an error if this is not the case

% If grouping, we have ValuesOnDist and StationaryDist that contain
% everything we will need. Now we just have to compute them.
% Note that I do not currently allow the following simoptions to differ by PType

for kk=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
    clear FnsToEvaluate_kk
    FnsToEvaluate_kk.(FnsToEvalNames{kk})=FnsToEvaluate.(FnsToEvalNames{kk}); % Structure containing just this funcion
    FnsAndPTypeIndicator_kk=zeros(1,N_i,'gpuArray');
    
    % Following few lines relate to the digest
    delta=10000;
    Cmerge=struct(); % Keep a seperate Cmerge for each agegrouping
    digestweightsmerge=struct(); % Keep a seperate digestweightsmerge for each agegrouping
    for jj=1:maxngroups
        Cmerge(jj).Cmerge=zeros(5000*N_i,1); % This is intended to be an upper limit on number of points that might be use
        digestweightsmerge(jj).digestweightsmerge=zeros(5000*N_i,1); % This is intended to be an upper limit on number of points that might be use
    end
    merge_nsofar=zeros(1,maxngroups); % Keep count (by age grouping)  
    
    MeanVec=zeros(N_i,maxngroups);
    StdDevVec=zeros(N_i,maxngroups);



    for ii=1:N_i
        % First set up simoptions
        simoptions_temp=PType_Options(simoptions,Names_i,ii); % Note: already check for existence of simoptions and created it if it was not inputted

        if simoptions_temp.verbose==1
            fprintf('Permanent type: %i of %i \n',ii, N_i)
        end

        if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
            PolicyIndexes_temp=gpuArray(Policy.(Names_i{ii}));
            StationaryDist_temp=gpuArray(StationaryDist.(Names_i{ii}));
        else
            PolicyIndexes_temp=Policy.(Names_i{ii});
            StationaryDist_temp=StationaryDist.(Names_i{ii});
        end
        % Parallel is determined by StationaryDist, unless it is specified
        if isa(StationaryDist_temp, 'gpuArray')
            Parallel_temp=2;
        else
            Parallel_temp=1;
        end
        if isfield(simoptions_temp,'parallel')
            Parallel_temp=simoptions_temp.parallel;
            if Parallel_temp~=2
                PolicyIndexes_temp=gather(PolicyIndexes_temp);
                StationaryDist_temp=gather(StationaryDist_temp);
            end
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

        if isa(n_d,'struct')
            n_d_temp=n_d.(Names_i{ii});
        else
            n_d_temp=n_d;
        end
        if isa(n_a,'struct')
            n_a_temp=n_a.(Names_i{ii});
        else
            n_a_temp=n_a;
        end
        if isa(n_z,'struct')
            n_z_temp=n_z.(Names_i{ii});
        else
            n_z_temp=n_z;
        end
        if isa(N_j,'struct')
            N_j_temp=N_j.(Names_i{ii});
        else
            N_j_temp=N_j;
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
                [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType.
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
        % Note: next line uses FnsToEvaluate_kk
        [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate_kk,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0,2);
        FnsAndPTypeIndicator_kk(ii)=FnsAndPTypeIndicator_ii;
        
        % Because of how we loop over both kk (FnsToEvaluate) and ii (PType),
        % WhichFnsForCurrentPType will be a scalar 1 or 0. If zero then we
        % can just skip everything for this agent permanent type, so only
        % do things when it is one.
        if WhichFnsForCurrentPType==1
            
            %% We have set up the current PType, now do some calculations for it.
            simoptions_temp.keepoutputasmatrix=2; %2: is a matrix, but of a different form to 1
            ValuesOnGrid_ii=gather(EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case2(PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp, simoptions_temp));
            N_a_temp=prod(n_a_temp);
            if isfield(simoptions_temp,'n_e')
                n_z_temp=[n_z_temp,simoptions_temp.n_e];
            end
            N_z_temp=prod(n_z_temp);
            
            ValuesOnGrid_ii=reshape(ValuesOnGrid_ii,[N_a_temp*N_z_temp,N_j_temp]);
            
            StationaryDist_ii=reshape(StationaryDist.(Names_i{ii}),[N_a_temp*N_z_temp,N_j_temp]); % Note: does not impose *StationaryDist.ptweights(ii)
            
            AgeConditionalStats_ii.Mean=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Median=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Variance=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.StdDev=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.LorenzCurve=nan(simoptions_temp.npoints,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Gini=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.QuantileCutoffs=nan(simoptions_temp.nquantiles+1,ngroups(ii),'gpuArray'); % Includes the min and max values
            AgeConditionalStats_ii.QuantileMeans=nan(simoptions_temp.nquantiles,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Top1share=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Top5share=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Top10share=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Bottom50share=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Median=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Percentile50th=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Percentile90th=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Percentile95th=nan(1,ngroups(ii),'gpuArray');
            AgeConditionalStats_ii.Percentile99th=nan(1,ngroups(ii),'gpuArray');
            
            for jj=1:length(simoptions_temp.agegroupings)
                j1=simoptions_temp.agegroupings(jj);
                if jj<length(simoptions_temp.agegroupings)
                    jend=simoptions_temp.agegroupings(jj+1)-1;
                else
                    jend=N_j_temp;
                end
                % simoptions.agejshifter
                
                
                % Calculate the individual stats
                StationaryDistVec_jj=reshape(StationaryDist_ii(:,j1:jend),[N_a_temp*N_z_temp*(jend-j1+1),1]);
                Values_jj=reshape(ValuesOnGrid_ii(:,j1:jend),[N_a_temp*N_z_temp*(jend-j1+1),1]);
                
                % Eliminate all the zero-weights from these (this would
                % increase run times if we only do exact calculations, but
                % because we plan to createDigest() it helps reduce runtimes)
                temp=logical(StationaryDistVec_jj~=0);
                StationaryDistVec_jj=StationaryDistVec_jj(temp);
                Values_jj=Values_jj(temp);
                
                % Should be mass one, but just enforce to reduce numerical rounding errors
                StationaryDistVec_jj=StationaryDistVec_jj./sum(StationaryDistVec_jj); % Normalize to sum to one for this 'agegrouping'
                
                % Sort by values
                [SortedValues,SortedValues_index] = sort(Values_jj);
                SortedWeights = StationaryDistVec_jj(SortedValues_index);
                
                CumSumSortedWeights=cumsum(SortedWeights);
                WeightedValues=Values_jj.*StationaryDistVec_jj;
                SortedWeightedValues=WeightedValues(SortedValues_index);
                
                
                % Calculate the 'age conditional' mean
                AgeConditionalStats_ii.Mean(jj)=sum(WeightedValues);
                % Calculate the 'age conditional' median
                [~,medianindex]=min(abs(SortedWeights-0.5));
                AgeConditionalStats_ii.Median(jj)=SortedValues(medianindex);
                
                % Do min and max before looking at the variance, std. dev.,
                % lorenz curve, etc. As that way can skip when the min and max
                % are the same (so variable is constant valued)
                
                % Min value
                tempindex=find(CumSumSortedWeights>=simoptions_temp.tolerance,1,'first');
                minvalue=SortedValues(tempindex);
                % Max value
                tempindex=find(CumSumSortedWeights>=(1-simoptions_temp.tolerance),1,'first');
                maxvalue=SortedValues(tempindex);
                % Numerical rounding can sometimes leave that there is no maxvalue satifying this criterion, in which case we loosen the tolerance
                if isempty(maxvalue)
                    tempindex=find(CumSumSortedWeights>=(1-10*simoptions_temp.tolerance),1,'first'); % If failed to find, then just loosen tolerance by order of magnitude
                    maxvalue=SortedValues(tempindex);
                end
                
                % Calculate the 'age conditional' variance
                if (maxvalue-minvalue)>0
                    AgeConditionalStats_ii.Variance(jj)=sum((Values_jj.^2).*StationaryDistVec_jj)-(AgeConditionalStats_ii.Mean(jj))^2; % Weighted square of values - mean^2
                else % There were problems at floating point error accuracy levels when there is no variance, so just treat this case directly
                    AgeConditionalStats_ii.Variance(jj)=0;
                end
                if AgeConditionalStats_ii.Variance(jj)<0 % Some variance still appear to be machine tolerance level errors.
                    AgeConditionalStats_ii.StdDev(jj)=0; % You will be able to see the machine tolerance level error in the variance, and it is just overwritten to zero in the standard deviation
                else
                    AgeConditionalStats_ii.StdDev(jj)=sqrt(AgeConditionalStats_ii.Variance(jj));
                end
                
                % Calculate the 'age conditional' lorenz curve
                % Note: Commented out following line as would also need to
                % change TopXshare stats, decided not to do this.
                %             if minvalue<0
                %                 AgeConditionalStats_ii.LorenzCurve(:,jj)=nan;
                %                 AgeConditionalStats_ii.Gini(jj)=nan;
                %             else
                if (maxvalue-minvalue)>0
                    LorenzCurve=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,simoptions_temp.npoints,2);
                    AgeConditionalStats_ii.LorenzCurve(:,jj)=LorenzCurve;
                    % Calculate the 'age conditional' gini
                    AgeConditionalStats_ii.Gini(jj)=Gini_from_LorenzCurve(LorenzCurve);
                else
                    LorenzCurve=linspace(0,1,simoptions_temp.npoints);
                    AgeConditionalStats_ii.Gini(jj)=1;
                end
                
                % Top X share indexes
                Top1cutpoint=round(0.99*simoptions_temp.npoints);
                Top5cutpoint=round(0.95*simoptions_temp.npoints);
                Top10cutpoint=round(0.90*simoptions_temp.npoints);
                Top50cutpoint=round(0.50*simoptions_temp.npoints);
                AgeConditionalStats_ii.Top1share(jj)=sum(LorenzCurve(1+Top1cutpoint:end));
                AgeConditionalStats_ii.Top5share(jj)=sum(LorenzCurve(1+Top5cutpoint:end));
                AgeConditionalStats_ii.Top10share(jj)=sum(LorenzCurve(1+Top10cutpoint:end));
                AgeConditionalStats_ii.Bottom50share(jj)=sum(LorenzCurve(1:Top50cutpoint));
                % Now some cutoffs
                index_median=find(CumSumSortedWeights>=0.5,1,'first');
                AgeConditionalStats_ii.Median(jj)=SortedValues(index_median);
                AgeConditionalStats_ii.Percentile50th(jj)=SortedValues(index_median);
                index_p90=find(CumSumSortedWeights>=0.90,1,'first');
                AgeConditionalStats_ii.Percentile90th(jj)=SortedValues(index_p90);
                index_p95=find(CumSumSortedWeights>=0.95,1,'first');
                AgeConditionalStats_ii.Percentile95th(jj)=SortedValues(index_p95);
                index_p99=find(CumSumSortedWeights>=0.99,1,'first');
                AgeConditionalStats_ii.Percentile99th(jj)=SortedValues(index_p99);
                
                
                % Calculate the 'age conditional' quantile means (ventiles by default)
                % Calculate the 'age conditional' quantile cutoffs (ventiles by default)
                QuantileIndexes=zeros(1,simoptions_temp.nquantiles-1);
                QuantileCutoffs=zeros(1,simoptions_temp.nquantiles-1);
                QuantileMeans=zeros(1,simoptions_temp.nquantiles);
                
                for ll=1:simoptions_temp.nquantiles-1
                    tempindex=find(CumSumSortedWeights>=ll/simoptions_temp.nquantiles,1,'first');
                    QuantileIndexes(ll)=tempindex;
                    QuantileCutoffs(ll)=SortedValues(tempindex);
                    if ll==1
                        QuantileMeans(ll)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
                    elseif ll<(simoptions_temp.nquantiles-1) % (1<ll) &&
                        QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                    else %if ll==(options.nquantiles-1)
                        QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                        QuantileMeans(ll+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
                    end
                end
                
                AgeConditionalStats_ii.QuantileCutoffs(:,jj)=[minvalue, QuantileCutoffs, maxvalue]';
                AgeConditionalStats_ii.QuantileMeans(:,jj)=QuantileMeans';
                
                minvaluevec(ii)=minvalue; % Keep so that we can calculate the grouped min directly from this
                maxvaluevec(ii)=maxvalue; % Keep so that we can calculate the grouped max directly from this
                
                % Now that we have done the individual stats, store the mean,
                % stddev, and t-Digests so that we can compute the grouped stats.
                % (Mean and stddev can just be done after the loop)
                
                %% Create digest
                [C_jj,digestweights_jj,~]=createDigest(SortedValues, SortedWeights,delta,1); % 1=presorted, as we sorted these above
                
                %% Keep the digests so far as a stacked vector that can then merge later
                jjageshifted=jj+simoptions.agejshifter(ii);
                % Note that this will be automatically created such that it
                % only contains the agents for whom it is relevant.
                merge_nsofar2_jj=merge_nsofar(jjageshifted)+length(C_jj);
                % Note: merge across the ii, but keep the different jj distinct
                Cmerge(jjageshifted).Cmerge(merge_nsofar(jjageshifted)+1:merge_nsofar2_jj)=C_jj;
                digestweightsmerge(jjageshifted).digestweightsmerge(merge_nsofar(jjageshifted)+1:merge_nsofar2_jj)=digestweights_jj*StationaryDist.ptweights(ii);
                merge_nsofar(jjageshifted)=merge_nsofar2_jj;

%                 merge_nsofar2_jj=merge_nsofar(jj)+length(C_jj);
%                 % Note: merge across the ii, but keep the different jj distinct
%                 Cmerge(jj).Cmerge(merge_nsofar(jj)+1:merge_nsofar2_jj)=C_jj;
%                 digestweightsmerge(jj).digestweightsmerge(merge_nsofar(jj)+1:merge_nsofar2_jj)=digestweights_jj*StationaryDist.ptweights(ii);
%                 merge_nsofar(jj)=merge_nsofar2_jj;
%                 
%                 % DEBUGGING
%                 if kk==1 || kk==2
%                     if any(isnan(C_jj))
%                         fprintf('For age %i there are %i nan values in the digest means for agent %i for function %i \n',jj,sum(isnan(C_jj)),ii,kk)
%                         nanindex = find(isnan(C_jj)); % Find the non-zero values of isnan(C_jj), which are the nan values of C_jj
%                         for aaa=1:length(nanindex)
%                             fprintf('The nan is in index %i of %i \n',nanindex(aaa),length(C_jj))
%                             fprintf('The corresponding digestweight is %8.8f \n',digestweights_jj(aaa))
%                         end
%                         fprintf('Mass of agent dist-1 is %8.12f (should be zero) \n',sum(StationaryDistVec_jj)-1)
%                         fprintf('Number of zero-mass points in agent dist is %i (out of %i) \n',sum(StationaryDistVec_jj==0),numel(StationaryDistVec_jj))
%                         % The digestweight is not zero. So why is C_jj nan?
%                         % Maybe something about the values?
%                         fprintf('Number of finite values in Values_jj=%i, out of total of %i \n',sum(isfinite(Values_jj)),numel(Values_jj))
%                         fprintf('Number of nan in Values_jj=%i, out of total of %i \n',sum(isnan(Values_jj)),numel(Values_jj))
%                     end
%                     if any(isnan(digestweights_jj))
%                         fprintf('For age %i there are %i nan values in the digest weights for agent %i for function %i \n',jj,sum(isnan(digestweights_jj)),ii,kk)
%                         nanindex = find(isnan(digestweights_jj));
%                         for aaa=1:length(nanindex)
%                             fprintf('The nan is in index %i of %i \n',nanindex(aaa),length(digestweights_jj))
%                         end
%                     end
%                     if any((digestweights_jj==0))
%                         fprintf('For age %i there are %i zero values in the digest weights for agent %i for function %i \n',jj,sum((digestweights_jj==0)),ii,kk)
%                         nanindex = find((digestweights_jj==0));
%                         for aaa=1:length(nanindex)
%                             fprintf('The nan is in index %i of %i \n',nanindex(aaa),length(digestweights_jj))
%                         end
%                     end
%                 end
                
            end
            MeanVec(ii,1:ngroups(ii))=AgeConditionalStats_ii.Mean; % Note: if ngroups(ii)<max(ngroups) there will just be some NaN left at the end of the row
            StdDevVec(ii,1:ngroups(ii))=AgeConditionalStats_ii.StdDev;
            
            % Put the individual ones into the output
            AgeConditionalStats.(FnsToEvalNames{kk}).(Names_i{ii}).Mean=AgeConditionalStats_ii.Mean;
            AgeConditionalStats.(FnsToEvalNames{kk}).(Names_i{ii}).Median=AgeConditionalStats_ii.Median;
            AgeConditionalStats.(FnsToEvalNames{kk}).(Names_i{ii}).Variance=AgeConditionalStats_ii.Variance;
            AgeConditionalStats.(FnsToEvalNames{kk}).(Names_i{ii}).StdDev=AgeConditionalStats_ii.StdDev;
            AgeConditionalStats.(FnsToEvalNames{kk}).(Names_i{ii}).LorenzCurve=AgeConditionalStats_ii.LorenzCurve;
            AgeConditionalStats.(FnsToEvalNames{kk}).(Names_i{ii}).Gini=AgeConditionalStats_ii.Gini;
            AgeConditionalStats.(FnsToEvalNames{kk}).(Names_i{ii}).QuantileCutoffs=AgeConditionalStats_ii.QuantileCutoffs;
            AgeConditionalStats.(FnsToEvalNames{kk}).(Names_i{ii}).QuantileMeans=AgeConditionalStats_ii.QuantileMeans;
        end
    end

%     disp('nmergesofar')
%     merge_nsofar

%     %% Now create the grouped stats from the mean, stddev and t-digests    
%     N_i_kk=sum(FnsAndPTypeIndicator_kk,:); % How many agents is this statistic calculated for

    % In principle could calculate mean and standard deviation without
    % needing to loop over jj. But I want to allow for agejshifter and different N_j, and so
    % it is much easier to use the loops.

    % Preallocate empty matrices
    AgeConditionalStats.(FnsToEvalNames{kk}).Mean=zeros(1,maxngroups); % Needs to be zeros as I add to it
    AgeConditionalStats.(FnsToEvalNames{kk}).StdDev=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Variance=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).LorenzCurve=nan(simoptions_temp.npoints,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Gini=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Top1share=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Top5share=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Top10share=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Bottom50share=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Median=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Gini=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Percentile50th=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Percentile90th=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Percentile95th=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).Percentile99th=nan(1,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).QuantileMeans=nan(simoptions_temp.nquantiles,maxngroups);
    AgeConditionalStats.(FnsToEvalNames{kk}).QuantileCutoffs=nan(simoptions_temp.nquantiles+1,maxngroups);
    
    
    for jj=1:max(ngroups)%length(simoptions.agegroupings)
        % Note: don't need j1 and jend this time round
        
        jjageshifted2=jj-simoptions.agejshifter(ii); % Note, here it is minus

        % Mass of agents of age jj for who this function is relevant
        SigmaNxi_jj=0;
        for ii=1:N_i
            if FnsAndPTypeIndicator_kk(ii)==1 && jj<ngroups(ii) && jj>simoptions.agejshifter(ii) % if current function and age are relevant for type ii
                SigmaNxi_jj=SigmaNxi_jj+FnsAndPTypeIndicator_kk(ii)*StationaryDist.ptweights(ii);
            end
        end
        
        % Grouped Mean
        for ii=1:N_i
            if FnsAndPTypeIndicator_kk(ii)==1 && jj<ngroups(ii) && jj>simoptions.agejshifter(ii) % if current function and age are relevant for type ii
                AgeConditionalStats.(FnsToEvalNames{kk}).Mean(jj)=AgeConditionalStats.(FnsToEvalNames{kk}).Mean(jj)+MeanVec(ii,jjageshifted2)*StationaryDist.ptweights(ii);
            end
        end
        
        if SigmaNxi_jj>0 % to avoid divide by zero error
            AgeConditionalStats.(FnsToEvalNames{kk}).Mean(jj)=AgeConditionalStats.(FnsToEvalNames{kk}).Mean(jj)/SigmaNxi_jj;
        end

        % Grouped Standard Deviation
        if N_i==1
            AgeConditionalStats.(FnsToEvalNames{kk}).StdDev(jj)=StdDevVec(1,jj); % If there is only one agent then obviously we can ignore checking if the current function and age are relevant
        else
            temp2=zeros(N_i,1);
            for ii=2:N_i
                if FnsAndPTypeIndicator_kk(ii)==1 && jj<ngroups(ii) && jj>simoptions.agejshifter(ii) % if current function and age are relevant for type ii
                    temp2(ii)=StationaryDist.ptweights(ii)*sum(FnsAndPTypeIndicator_kk(1:(ii-1))'.*(StationaryDist.ptweights(1:(ii-1))).*((MeanVec(1:(ii-1),jjageshifted2)-MeanVec(ii,jjageshifted2)).^2));
                end
            end
            AgeConditionalStats.(FnsToEvalNames{kk}).StdDev(jj)=sqrt(sum(FnsAndPTypeIndicator_kk.*(StationaryDist.ptweights').*StdDevVec(:,jjageshifted2)')/SigmaNxi_jj + sum(temp2)/(SigmaNxi_jj^2));
        end
        AgeConditionalStats.(FnsToEvalNames{kk}).Variance(jj)=(AgeConditionalStats.(FnsToEvalNames{kk}).StdDev(jj)).^2;
        
        Cmerge_jj=Cmerge(jj).Cmerge;
        digestweightsmerge_jj=digestweightsmerge(jj).digestweightsmerge;
        merge_nsofar_jj=merge_nsofar(jj);
        
        Cmerge_jj=Cmerge_jj(1:merge_nsofar_jj);
        digestweightsmerge_jj=digestweightsmerge_jj(1:merge_nsofar_jj);

        %% Now for the grouped stats which are calculated from digests
        % Note that the t-digests were already created incorporating any agejshifter
        if AgeConditionalStats.(FnsToEvalNames{kk}).StdDev(jj)>0
            % No point calculating all of these inequality stats if the (age-conditional) standard deviation is zero
        
            % Merge the digests
            [C_kk,digestweights_kk,qlimitvec_kk]=mergeDigest(Cmerge_jj, digestweightsmerge_jj, delta);
            % DEBUGGING
            if kk==1 || kk==2
                if jj==1
                    if any(isnan(C_jj))
                        fprintf('For age 1 there are %i nan values in the digest means for merged-agents \n',jj,sum(isnan(C_kk)))
                    end
                    if any(isnan(digestweights_kk))
                        fprintf('For age 1 there are %i nan values in the digest weights for merged-agents \n',jj,sum(isnan(digestweights_kk)))
                    end
                    if any(isnan(qlimitvec_kk))
                        fprintf('For age 1 there are %i nan values in the digest weights for merged-agents \n',jj,sum(isnan(qlimitvec_kk)))
                    end
                end
            end
            
            % Top X share indexes
            Top1cutpoint=round(0.99*simoptions_temp.npoints);
            Top5cutpoint=round(0.95*simoptions_temp.npoints);
            Top10cutpoint=round(0.90*simoptions_temp.npoints);
            Top50cutpoint=round(0.50*simoptions_temp.npoints);
            
            if C_kk(1)<0
                warning('Lorenz curve for the %i-th FnsToEvaluate is complicated as it takes some negative values \n',kk)
            end
            % Calculate the quantiles
            LorenzCurve=LorenzCurve_subfunction_PreSorted(C_kk.*digestweights_kk,qlimitvec_kk,simoptions_temp.npoints,1);
            AgeConditionalStats.(FnsToEvalNames{kk}).LorenzCurve(:,jj)=LorenzCurve;
            AgeConditionalStats.(FnsToEvalNames{kk}).Gini(jj)=Gini_from_LorenzCurve(LorenzCurve);
            AgeConditionalStats.(FnsToEvalNames{kk}).Top1share(jj)=sum(LorenzCurve(1+Top1cutpoint:end));
            AgeConditionalStats.(FnsToEvalNames{kk}).Top5share(jj)=sum(LorenzCurve(1+Top5cutpoint:end));
            AgeConditionalStats.(FnsToEvalNames{kk}).Top10share(jj)=sum(LorenzCurve(1+Top10cutpoint:end));
            AgeConditionalStats.(FnsToEvalNames{kk}).Bottom50share(jj)=sum(LorenzCurve(1:Top50cutpoint));
            
            cumsumdigestweights_kk=cumsum(digestweights_kk);
            % Now some cutoffs (note: qlimitvec is effectively already the cumulative sum)
            index_median=find(cumsumdigestweights_kk>=0.5,1,'first');
            AgeConditionalStats.(FnsToEvalNames{kk}).Median(jj)=C_kk(index_median);
            AgeConditionalStats.(FnsToEvalNames{kk}).Percentile50th(jj)=C_kk(index_median);
            index_p90=find(cumsumdigestweights_kk>=0.90,1,'first');
            AgeConditionalStats.(FnsToEvalNames{kk}).Percentile90th(jj)=C_kk(index_p90);
            index_p95=find(cumsumdigestweights_kk>=0.95,1,'first');
            AgeConditionalStats.(FnsToEvalNames{kk}).Percentile95th(jj)=C_kk(index_p95);
            index_p99=find(cumsumdigestweights_kk>=0.99,1,'first');
            AgeConditionalStats.(FnsToEvalNames{kk}).Percentile99th(jj)=C_kk(index_p99);
            
            % Calculate the quantiles directly from the digest
            quantiles=(1:1:simoptions_temp.nquantiles-1)/simoptions_temp.nquantiles;
            quantilecutoffs=interp1(qlimitvec_kk,C_kk,quantiles);
            quantilemeans=zeros(length(quantilecutoffs)+1,1);
            Ctimesdisgestweights=C_kk.*digestweights_kk;
            quantilemeans(1)=sum(Ctimesdisgestweights(qlimitvec_kk<quantiles(1)))/sum(digestweights_kk(qlimitvec_kk<quantiles(1)));
            for qq=2:length(quantilecutoffs)
                quantilemeans(qq)=sum(Ctimesdisgestweights(logical((qlimitvec_kk>quantiles(qq-1)).*(qlimitvec_kk<quantiles(qq)))))/sum(digestweights_kk(logical((qlimitvec_kk>quantiles(qq-1)).*(qlimitvec_kk<quantiles(qq)))));
            end
            quantilemeans(end)=sum(Ctimesdisgestweights(qlimitvec_kk>quantiles(end)))/sum(digestweights_kk(qlimitvec_kk>quantiles(end)));
            
            % The minvalue and maxvalue can just be calculated direct from the invididual agent ones
            % Note: the nan in minvaluevec and maxvaluevec are the preallocated size (which we then only partly fill)
            AgeConditionalStats.(FnsToEvalNames{kk}).QuantileCutoffs(:,jj)=[min(minvaluevec,[],'omitnan'), quantilecutoffs, max(maxvaluevec,[],'omitnan')]';
            AgeConditionalStats.(FnsToEvalNames{kk}).QuantileMeans(:,jj)=quantilemeans';
        else
            % Since these are all undefined just return them a message explaining why
            AgeConditionalStats.(FnsToEvalNames{kk}).Note='The inequality stats are odd because the (age-conditional) standard deviation is zero';
            AgeConditionalStats.(FnsToEvalNames{kk}).LorenzCurve(:,jj)=cumsum(ones(simoptions_temp.npoints,1)/simoptions_temp.npoints);
            AgeConditionalStats.(FnsToEvalNames{kk}).Gini(jj)=0;
            AgeConditionalStats.(FnsToEvalNames{kk}).Top1share(jj)=0.01;
            AgeConditionalStats.(FnsToEvalNames{kk}).Top5share(jj)=0.05;
            AgeConditionalStats.(FnsToEvalNames{kk}).Top10share(jj)=0.1;
            AgeConditionalStats.(FnsToEvalNames{kk}).Bottom50share(jj)=0.5;
            AgeConditionalStats.(FnsToEvalNames{kk}).Median(jj)=AgeConditionalStats.(FnsToEvalNames{kk}).Mean(jj);
            AgeConditionalStats.(FnsToEvalNames{kk}).Percentile50th(jj)=AgeConditionalStats.(FnsToEvalNames{kk}).Mean(jj);
            AgeConditionalStats.(FnsToEvalNames{kk}).Percentile90th(jj)=AgeConditionalStats.(FnsToEvalNames{kk}).Mean(jj);
            AgeConditionalStats.(FnsToEvalNames{kk}).Percentile95th(jj)=AgeConditionalStats.(FnsToEvalNames{kk}).Mean(jj);
            AgeConditionalStats.(FnsToEvalNames{kk}).Percentile99th(jj)=AgeConditionalStats.(FnsToEvalNames{kk}).Mean(jj);
            AgeConditionalStats.(FnsToEvalNames{kk}).QuantileCutoffs(:,jj)=AgeConditionalStats.(FnsToEvalNames{kk}).Mean(jj)*ones(simoptions_temp.nquantiles+1,1);
            AgeConditionalStats.(FnsToEvalNames{kk}).QuantileMeans(:,jj)=AgeConditionalStats.(FnsToEvalNames{kk}).Mean(jj)*ones(simoptions_temp.nquantiles,1);
        end
    end
end




end
