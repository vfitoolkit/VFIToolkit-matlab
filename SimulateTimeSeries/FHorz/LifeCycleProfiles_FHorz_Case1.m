function AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,options)
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
    if isfield(options,'parallel')==0
        options.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if isfield(options,'verbose')==0
        options.verbose=0;
    end
    if isfield(options,'nquantiles')==0
        options.nquantiles=20; % by default gives ventiles
    end
    if isfield(options,'agegroupings')==0
        options.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    end
    if isfield(options,'npoints')==0
        options.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    end
    if isfield(options,'tolerance')==0    
        options.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
    
else
    %If options is not given, just use all the defaults
    options.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    options.verbose=0;
    options.nquantiles=20; % by default gives ventiles
    options.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    options.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    options.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

%% Create a different 'Values' for each of the variable to be evaluated

StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);

ngroups=length(options.agegroupings);
if options.parallel==2
    PolicyValues=PolicyInd2Val_FHorz_Case1(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid, options.parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]

    % Do some preallocation of the output structure
    AgeConditionalStats(length(FnsToEvaluate)).Mean=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).Median=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).Variance=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(options.npoints,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups,'gpuArray');
    AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(options.nquantiles+1,ngroups,'gpuArray'); % Includes the min and max values
    AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(options.nquantiles,ngroups,'gpuArray');
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]);  % I reshape here, and THEN JUST RESHAPE AGAIN WHEN USING. THIS IS STUPID AND SLOW.
    for kk=1:length(options.agegroupings)
        j1=options.agegroupings(kk);
        if kk<length(options.agegroupings)
            jend=options.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDistVec(:,j1:jend),[N_a*N_z*(jend-j1+1),1]);
        StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'
        
        for ii=1:length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
            Values=nan(N_a*N_z,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames)% check for 'FnsToEvaluateParamNames={}'
                    FnsToEvaluateParamsVec=[];
                else
                    FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj);
                end
                Values(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{ii}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,options.parallel),[N_a*N_z,1]);
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
            AgeConditionalStats(ii).Median(kk)=SortedWeightedValues(floor(0.5*length(SortedWeightedValues)));
            % Calculate the 'age conditional' variance
            AgeConditionalStats(ii).Variance(kk)=sum((Values.^2).*StationaryDistVec_kk)-(AgeConditionalStats(ii).Mean(kk))^2; % Weighted square of values - mean^2
            
            
            if options.npoints>0
                % Calculate the 'age conditional' lorenz curve
                AgeConditionalStats(ii).LorenzCurve(:,kk)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,options.npoints);
                % Calculate the 'age conditional' gini
                AgeConditionalStats(ii).Gini(kk)=Gini_from_LorenzCurve(AgeConditionalStats(ii).LorenzCurve(:,kk));
            end
            
            % Calculate the 'age conditional' quantile means (ventiles by default)
            % Calculate the 'age conditional' quantile cutoffs (ventiles by default)
            QuantileIndexes=zeros(1,options.nquantiles-1,'gpuArray');
            QuantileCutoffs=zeros(1,options.nquantiles-1,'gpuArray');
            QuantileMeans=zeros(1,options.nquantiles,'gpuArray');
            
            for ll=1:options.nquantiles-1
                tempindex=find(CumSumSortedWeights>=ll/options.nquantiles,1,'first');
                QuantileIndexes(ll)=tempindex;
                QuantileCutoffs(ll)=SortedValues(tempindex);
                if ll==1
                    QuantileMeans(ll)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
                elseif ll<(options.nquantiles-1) % (1<ll) &&
                    QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                else %if ll==(options.nquantiles-1)
                    QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                    QuantileMeans(ll+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
                end
            end
            % Min value
            tempindex=find(CumSumSortedWeights>=options.tolerance,1,'first');
            minvalue=SortedValues(tempindex);
            % Max value
            tempindex=find(CumSumSortedWeights>=(1-options.tolerance),1,'first');
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
    AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(options.npoints,ngroups);
    AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups);
    AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(options.nquantiles+1,ngroups); % Includes the min and max values
    AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(options.nquantiles,ngroups);
    
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
    
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);

%     PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]);  % I reshape here, and THEN JUST RESHAPE AGAIN WHEN USING. THIS IS STUPID AND SLOW.
    for kk=1:length(options.agegroupings)
        j1=options.agegroupings(kk);
        if kk<length(options.agegroupings)
            jend=options.agegroupings(kk+1)-1;
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
        end
        
        for ii=1:length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
            Values=nan(N_a*N_z,jend-j1+1); % Preallocate
            if l_d>0
                for jj=j1:jend
                    d_gridvals=gridvalsFull(jj-j1+1).d_gridvals;
                    aprime_gridvals=gridvalsFull(jj-j1+1).aprime_gridvals;
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
                    d_gridvals=gridvalsFull(jj-j1+1).d_gridvals;
                    aprime_gridvals=gridvalsFull(jj-j1+1).aprime_gridvals;
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
            AgeConditionalStats(ii).Median(kk)=SortedWeightedValues(floor(0.5*length(SortedWeightedValues)));
            % Calculate the 'age conditional' variance
            AgeConditionalStats(ii).Variance(kk)=sum((Values.^2).*StationaryDistVec_kk)-(AgeConditionalStats(ii).Mean(kk))^2; % Weighted square of values - mean^2
            
            
            if options.npoints>0
                % Calculate the 'age conditional' lorenz curve
                AgeConditionalStats(ii).LorenzCurve(:,kk)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,options.npoints,1);
                % Calculate the 'age conditional' gini
                AgeConditionalStats(ii).Gini(kk)=Gini_from_LorenzCurve(AgeConditionalStats(ii).LorenzCurve(:,kk));
            end
            
            % Calculate the 'age conditional' quantile means (ventiles by default)
            % Calculate the 'age conditional' quantile cutoffs (ventiles by default)
            QuantileIndexes=zeros(1,options.nquantiles-1);
            QuantileCutoffs=zeros(1,options.nquantiles-1);
            QuantileMeans=zeros(1,options.nquantiles);
            
            for ll=1:options.nquantiles-1
                tempindex=find(CumSumSortedWeights>=ll/options.nquantiles,1,'first');
                QuantileIndexes(ll)=tempindex;
                QuantileCutoffs(ll)=SortedValues(tempindex);
                if ll==1
                    QuantileMeans(ll)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
                elseif ll<(options.nquantiles-1) % (1<ll) &&
                    QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                else %if ll==(options.nquantiles-1)
                    QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                    QuantileMeans(ll+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
                end
            end
            % Min value
            tempindex=find(CumSumSortedWeights>=options.tolerance,1,'first');
            minvalue=SortedValues(tempindex);
            % Max value
            tempindex=find(CumSumSortedWeights>=(1-options.tolerance),1,'first');
            maxvalue=SortedValues(tempindex);
            AgeConditionalStats(ii).QuantileCutoffs(:,kk)=[minvalue, QuantileCutoffs, maxvalue]';
            AgeConditionalStats(ii).QuantileMeans(:,kk)=QuantileMeans';
            
        end
    end
end


