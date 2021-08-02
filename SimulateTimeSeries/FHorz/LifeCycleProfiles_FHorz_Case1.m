function AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)
% Similar to SimLifeCycleProfiles but works from StationaryDist rather than
% simulating panel data. Where applicable it is faster and more accurate.
% options.agegroupings can be used to do conditional on 'age bins' rather than age
% e.g., options.agegroupings=1:10:N_j will divide into 10 year age bins and calculate stats for each of them
% options.npoints can be used to determine how many points are used for the lorenz curve
% options.nquantiles can be used to change from reporting (age conditional) ventiles, to quartiles/deciles/percentiles/etc.
%
% Note that the quantile are what are typically reported as life-cycle profiles (or more precisely, the quantile cutoffs).
%
% Output takes following form
% ngroups=length(options.agegroupings);
% AgeConditionalStats(length(FnsToEvaluate)).Mean=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).Median=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).Variance=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(ngroups,options.npoints);
% AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(ngroups,options.nquantiles+1); % Includes the min and max values
% AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(ngroups,options.nquantiles);

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Check which simoptions have been declared, set all others to defaults 
if exist('options','var')==1
    %Check options for missing fields, if there are some fill them with the defaults
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
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
    
else
    %If options is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    simoptions.verbose=0;
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

%% Create a different 'Values' for each of the variable to be evaluated

StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);

ngroups=length(simoptions.agegroupings);
if simoptions.parallel==2
    PolicyValues=PolicyInd2Val_FHorz_Case1(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid, simoptions.parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]

    % Do some preallocation of the output structure
    AgeConditionalStats(length(FnsToEvaluate)).Mean=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).Median=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).Variance=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(simoptions.npoints,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(simoptions.nquantiles+1,ngroups,'gpuArray'); % Includes the min and max values
    AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(simoptions.nquantiles,ngroups,'gpuArray');
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]);  % I reshape here, and THEN JUST RESHAPE AGAIN WHEN USING. THIS IS STUPID AND SLOW.
    for kk=1:length(simoptions.agegroupings)
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDistVec(:,j1:jend),[N_a*N_z*(jend-j1+1),1]);
        StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'
        
        for ii=1:length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
            Values=nan(N_a*N_z,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                if fieldexists_ExogShockFn==1
                    if fieldexists_ExogShockFnParamNames==1
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for kk=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(kk,1)={ExogShockFnParamsVec(kk)};
                        end
                        [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    else
                        [z_grid,~]=simoptions.ExogShockFn(jj);
                    end
                end
                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames)% check for 'FnsToEvaluateParamNames={}'
                    FnsToEvaluateParamsVec=[];
                else
                    FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj);
                end
                Values(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{ii}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel),[N_a*N_z,1]);
            end
            
            Values=reshape(Values,[N_a*N_z*(jend-j1+1),1]);
            
            [SortedValues,SortedValues_index] = sort(Values);
            
            SortedWeights = StationaryDistVec_kk(SortedValues_index);
            CumSumSortedWeights=cumsum(SortedWeights);
            
            WeightedValues=Values.*StationaryDistVec_kk;
            SortedWeightedValues=WeightedValues(SortedValues_index);
            
            % Calculate the 'age conditional' mean
            AgeConditionalStats(ii).Mean(kk)=sum(WeightedValues);
            % Calculate the 'age conditional' median
            AgeConditionalStats(ii).Median(kk)=SortedWeightedValues(max(1,floor(0.5*length(SortedWeightedValues)))); % The max is just to deal with 'corner' case where there is only one element in SortedWeightedValues
            % Calculate the 'age conditional' variance
            AgeConditionalStats(ii).Variance(kk)=sum((Values.^2).*StationaryDistVec_kk)-(AgeConditionalStats(ii).Mean(kk))^2; % Weighted square of values - mean^2
            
            
            if simoptions.npoints>0
                % Calculate the 'age conditional' lorenz curve
                AgeConditionalStats(ii).LorenzCurve(:,kk)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,simoptions.npoints,2);
                % Calculate the 'age conditional' gini
                AgeConditionalStats(ii).Gini(kk)=Gini_from_LorenzCurve(AgeConditionalStats(ii).LorenzCurve(:,kk));
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
            AgeConditionalStats(ii).QuantileCutoffs(:,kk)=[minvalue, QuantileCutoffs, maxvalue]';
            AgeConditionalStats(ii).QuantileMeans(:,kk)=QuantileMeans';
            
        end
    end

else % options.parallel~=2
    % Do some preallocation of the output structure
    AgeConditionalStats(length(FnsToEvaluate)).Mean=nan(1,ngroups);
    AgeConditionalStats(length(FnsToEvaluate)).Median=nan(1,ngroups);
    AgeConditionalStats(length(FnsToEvaluate)).Variance=nan(1,ngroups);
    AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(simoptions.npoints,ngroups);
    AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups);
    AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(simoptions.nquantiles+1,ngroups); % Includes the min and max values
    AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(simoptions.nquantiles,ngroups);
    
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
    
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);

%     PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]);  % I reshape here, and THEN JUST RESHAPE AGAIN WHEN USING. THIS IS STUPID AND SLOW.
    for kk=1:length(simoptions.agegroupings)
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDistVec(:,j1:jend),[N_a*N_z*(jend-j1+1),1]);
        StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'
        
        clear gridvalsFull
        for jj=j1:jend
            PolicyIndexes=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);
            [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,:,jj-j1+1),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
            gridvalsFull(jj-j1+1).d_gridvals=d_gridvals;
            gridvalsFull(jj-j1+1).aprime_gridvals=aprime_gridvals;
            
            if fieldexists_ExogShockFn==1
                if fieldexists_ExogShockFnParamNames==1
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for kk=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(kk,1)={ExogShockFnParamsVec(kk)};
                    end
                    [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                else
                    [z_grid,~]=simoptions.ExogShockFn(jj);
                end
                z_gridvals=CreateGridvals(n_z,z_grid,2);
            end
            gridvalsFull(jj-j1+1).z_gridvals=z_gridvals;
        end
        
        for ii=1:length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
            Values=nan(N_a*N_z,jend-j1+1); % Preallocate
            if l_d>0
                for jj=j1:jend
                    d_gridvals=gridvalsFull(jj-j1+1).d_gridvals;
                    aprime_gridvals=gridvalsFull(jj-j1+1).aprime_gridvals;
                    z_gridvals=gridvalsFull(jj-j1+1).z_gridvals;
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names) % check for 'FnsToEvaluateParamNames={}'
                        for ll=1:N_a*N_z
                            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                            l1=rem(ll-1,N_a)+1;
                            l2=ceil(ll/N_a);
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(d_gridvals{l1+(l2-1)*N_a,:},aprime_gridvals{l1+(l2-1)*N_a,:},a_gridvals{l1,:},z_gridvals{l2,:});
                        end
                    else
                        FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                        for ll=1:N_a*N_z
                            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                            l1=rem(ll-1,N_a)+1;
                            l2=ceil(ll/N_a);
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(d_gridvals{l1+(l2-1)*N_a,:},aprime_gridvals{l1+(l2-1)*N_a,:},a_gridvals{l1,:},z_gridvals{l2,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end
            else % l_d==0
                for jj=j1:jend
                    aprime_gridvals=gridvalsFull(jj-j1+1).aprime_gridvals;
                    z_gridvals=gridvalsFull(jj-j1+1).z_gridvals;
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names) % check for 'FnsToEvaluateParamNames={}'
                        for ll=1:N_a*N_z
                            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                            l1=rem(ll-1,N_a)+1;
                            l2=ceil(ll/N_a);
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(aprime_gridvals{l1+(l2-1)*N_a,:},a_gridvals{l1,:},z_gridvals{l2,:});
                        end
                    else
                        FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                        for ll=1:N_a*N_z
                            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                            l1=rem(ll-1,N_a)+1;
                            l2=ceil(ll/N_a);
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(aprime_gridvals{l1+(l2-1)*N_a,:},a_gridvals{l1,:},z_gridvals{l2,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end                   
            end
            
            % From here on is essentially identical to the 'with gpu' case.
            Values=reshape(Values,[N_a*N_z*(jend-j1+1),1]);
            
            [SortedValues,SortedValues_index] = sort(Values);
            
            SortedWeights = StationaryDistVec_kk(SortedValues_index);
            CumSumSortedWeights=cumsum(SortedWeights);
            
            WeightedValues=Values.*StationaryDistVec_kk;
            SortedWeightedValues=WeightedValues(SortedValues_index);
            
            % Calculate the 'age conditional' mean
            AgeConditionalStats(ii).Mean(kk)=sum(WeightedValues);
            % Calculate the 'age conditional' median
            AgeConditionalStats(ii).Median(kk)=SortedWeightedValues(max(1,floor(0.5*length(SortedWeightedValues)))); % The max is just to deal with 'corner' case where there is only one element in SortedWeightedValues
            % Calculate the 'age conditional' variance
            AgeConditionalStats(ii).Variance(kk)=sum((Values.^2).*StationaryDistVec_kk)-(AgeConditionalStats(ii).Mean(kk))^2; % Weighted square of values - mean^2
            
            
            if simoptions.npoints>0
                % Calculate the 'age conditional' lorenz curve
                AgeConditionalStats(ii).LorenzCurve(:,kk)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,simoptions.npoints,1);
                % Calculate the 'age conditional' gini
                AgeConditionalStats(ii).Gini(kk)=Gini_from_LorenzCurve(AgeConditionalStats(ii).LorenzCurve(:,kk));
            end
            
            % Calculate the 'age conditional' quantile means (ventiles by default)
            % Calculate the 'age conditional' quantile cutoffs (ventiles by default)
            QuantileIndexes=zeros(1,simoptions.nquantiles-1);
            QuantileCutoffs=zeros(1,simoptions.nquantiles-1);
            QuantileMeans=zeros(1,simoptions.nquantiles);
            
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
            AgeConditionalStats(ii).QuantileCutoffs(:,kk)=[minvalue, QuantileCutoffs, maxvalue]';
            AgeConditionalStats(ii).QuantileMeans(:,kk)=QuantileMeans';
            
        end
    end
end


