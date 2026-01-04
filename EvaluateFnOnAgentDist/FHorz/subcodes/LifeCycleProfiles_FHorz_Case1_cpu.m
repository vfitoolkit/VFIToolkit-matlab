function AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_cpu(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)
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
% AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(options.npoints,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(options.nquantiles+1,ngroups); % Includes the min and max values
% AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(options.nquantiles,ngroups);


if simoptions.parallel==2 % just make sure things are on gpu as they should be
    StationaryDist=gpuArray(StationaryDist);
    Policy=gpuArray(Policy);
end

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if isempty(n_d)
    l_d=0;
    n_d=0;
elseif n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

if n_z(1)==0
    l_z=0;
else
    l_z=length(n_z);
end



%% Implement new way of handling FnsToEvaluate
% Figure out l_daprime from Policy
l_daprime=size(Policy,1);

% Note: l_z includes e and semiz (when appropriate)
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'keepoutputasmatrix')
    if simoptions.keepoutputasmatrix==1
        FnsToEvaluateStruct=0;
    end
end


%% Create a different 'Values' for each of the variable to be evaluated

StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);

ngroups=length(simoptions.agegroupings);

% Do some preallocation of the output structure
AgeConditionalStats=struct();
for ii=length(FnsToEvaluate):-1:1 % Backwards as preallocating
    % length(FnsToEvaluate)
    AgeConditionalStats(ii).Mean=nan(1,ngroups);
    AgeConditionalStats(ii).Median=nan(1,ngroups);
    AgeConditionalStats(ii).Variance=nan(1,ngroups);
    AgeConditionalStats(ii).LorenzCurve=nan(simoptions.npoints,ngroups);
    AgeConditionalStats(ii).Gini=nan(1,ngroups);
    AgeConditionalStats(ii).QuantileCutoffs=nan(simoptions.nquantiles+1,ngroups); % Includes the min and max values
    AgeConditionalStats(ii).QuantileMeans=nan(simoptions.nquantiles,ngroups);
end

d_grid=gather(d_grid);
a_grid=gather(a_grid);
%     z_grid=gather(z_grid);

a_gridvals=CreateGridvals(n_a,a_grid,2);

PolicyIndexes=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);
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
        [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,simoptions,1, 2);
        gridvalsFull(jj-j1+1).d_gridvals=d_gridvals;
        gridvalsFull(jj-j1+1).aprime_gridvals=aprime_gridvals;

        if N_z>0
            if jointzgrid==0
                z_gridvals=CreateGridvals(n_z,z_grid_J(:,jj),2);
            else
                z_gridvals=z_grid_J(:,:,jj);
            end
        end
        if N_e>0
            if jointegrid==0
                e_gridvals=CreateGridvals(simoptions.n_e,e_grid_J(:,jj),2);
            else
                e_gridvals=e_grid_J(:,:,jj);
            end
        end
        if N_z>0 && N_e>0
            z_gridvals=[kron(z_gridvals,ones(N_e,1)),kron(ones(N_z,1),e_gridvals)];
        elseif N_e>0 && N_z==0
            z_gridvals=e_gridvals;
            % Note that n_z>0 and n_e=0 we just leave z_gridvals as is
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
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
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
                        l1=rem(ll-1,N_a)+1;
                        l2=ceil(ll/N_a);
                        Values(ll,jj-j1+1)=FnsToEvaluate{ii}(aprime_gridvals{l1+(l2-1)*N_a,:},a_gridvals{l1,:},z_gridvals{l2,:});
                    end
                else
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                    for ll=1:N_a*N_z
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
        [~,medianindex]=min(abs(SortedWeights-0.5));
        AgeConditionalStats(ii).Median(kk)=SortedValues(medianindex); % The max is just to deal with 'corner' case where there is only one element in SortedWeightedValues
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

if FnsToEvaluateStruct==1
    % Change the output into a structure
    AgeConditionalStats2=AgeConditionalStats;
    clear AgeConditionalStats
    AgeConditionalStats=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        AgeConditionalStats.(AggVarNames{ff}).Mean=AgeConditionalStats2(ff).Mean;
        AgeConditionalStats.(AggVarNames{ff}).Median=AgeConditionalStats2(ff).Median;
        AgeConditionalStats.(AggVarNames{ff}).Variance=AgeConditionalStats2(ff).Variance;
        AgeConditionalStats.(AggVarNames{ff}).LorenzCurve=AgeConditionalStats2(ff).LorenzCurve;
        AgeConditionalStats.(AggVarNames{ff}).Gini=AgeConditionalStats2(ff).Gini;
        AgeConditionalStats.(AggVarNames{ff}).QuantileCutoffs=AgeConditionalStats2(ff).QuantileCutoffs;
        AgeConditionalStats.(AggVarNames{ff}).QuantileMeans=AgeConditionalStats2(ff).QuantileMeans;
    end
end


end


