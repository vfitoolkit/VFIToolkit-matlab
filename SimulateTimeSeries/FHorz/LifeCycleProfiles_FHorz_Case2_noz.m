function AgeConditionalStats=LifeCycleProfiles_FHorz_Case2_noz(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,N_j,d_grid,a_grid,simoptions, AgeDependentGridParamNames)
% Similar to SimLifeCycleProfiles but works from StationaryDist rather than
% simulating panel data. Where applicable it is faster and more accurate.
% options.agegroupings can be used to do conditional on 'age bins' rather than age
% e.g., options.agegroupings=1:10:N_j will divide into 10 year age bins and calculate stats for each of them
% options.npoints can be used to determine how many points are used for the lorenz curve
% options.nquantiles can be used to change from reporting (age conditional) ventiles, to quartiles/deciles/percentiles/etc.
%
% options.crosssectioncorrelation can be used to create additional output
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
%
% When options.crosssectioncorrelation=1, there is additional output:
% AgeConditionalStats(length(FnsToEvaluate),length(FnsToEvaluate)).CrossSectionalCorrelation=nan(1,ngroups);
%
% AgeDependentGridParamNames is an optional input, and only used if prod(options.agedependentgrids)~=0


l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+1:end}}; % the first inputs will always be (d,aprime,a,z)
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
    elseif simoptions.keepoutputasmatrix==2
        FnsToEvaluateStruct=2;
    end
end

%% Create a different 'Values' for each of the variable to be evaluated
PolicyValues=PolicyInd2Val_FHorz_Case2_noz(Policy,n_d,n_a,N_j,d_grid);
permuteindexes=[1+(1:1:(l_a)),1,1+l_a+1];
PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]

StationaryDistVec=reshape(StationaryDist,[N_a,N_j]);

% Do some preallocation of the output structure
ngroups=length(simoptions.agegroupings);
AgeConditionalStats(length(FnsToEvaluate)).Mean=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).Median=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).Variance=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(simoptions.npoints,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(simoptions.nquantiles+1,ngroups,'gpuArray'); % Includes the min and max values
AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(simoptions.nquantiles,ngroups,'gpuArray');

if simoptions.crosssectioncorrelation==1
    AgeConditionalStats(length(FnsToEvaluate),length(FnsToEvaluate)).CrossSectionalCorrelation=nan(1,ngroups,'gpuArray');
end

PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*l_d,N_j]); % I reshape here, and THEN JUST RESHAPE AGAIN WHEN USING. THIS IS STUPID AND SLOW.
for kk=1:length(simoptions.agegroupings)
    j1=simoptions.agegroupings(kk);
    if kk<length(simoptions.agegroupings)
        jend=simoptions.agegroupings(kk+1)-1;
    else
        jend=N_j;
    end
    StationaryDistVec_kk=reshape(StationaryDistVec(:,j1:jend),[N_a*(jend-j1+1),1]);
    StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'
        
    for ii=1:length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
        Values=nan(N_a,jend-j1+1,'gpuArray'); % Preallocate
        for jj=j1:jend
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
                FnsToEvaluateParamsVec=[];
            else
                FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj);
            end            
            Values(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case2_noz(FnsToEvaluate{ii}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,l_d]),n_d,n_a,a_grid,simoptions.parallel),[N_a,1]);
        end
        
        Values=reshape(Values,[N_a*(jend-j1+1),1]);

        [SortedValues,SortedValues_index] = sort(Values);
        
        SortedWeights = StationaryDistVec_kk(SortedValues_index);
        CumSumSortedWeights=cumsum(SortedWeights);
        
        WeightedValues=Values.*StationaryDistVec_kk;
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        % Calculate the 'age conditional' mean
        AgeConditionalStats(ii).Mean(kk)=sum(WeightedValues);
        % Calculate the 'age conditional' median
        [~,medianindex]=min(abs(SortedWeights-0.5));
        AgeConditionalStats(ii).Median(kk)=SortedValues(medianindex);
        % Calculate the 'age conditional' variance
        
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

        
        if simoptions.crosssectioncorrelation==1 % Not coded the fastest way, but will use less memory than keeping round all of the older 'Values'
            for aa=(ii+1):length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
                Values2=nan(N_a,jend-j1+1,'gpuArray'); % Preallocate
                for jj=j1:jend
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
                        FnsToEvaluateParamsVec=[];
                    else
                        FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(aa).Names,jj);
                    end
                    Values2(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case2_noz(FnsToEvaluate{aa}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,l_d+l_a]),n_d,n_a,a_grid,simoptions.parallel),[N_a,1]);
                end
                
                Values2=reshape(Values2,[N_a*(jend-j1+1),1]);
                
                Mean1=sum(Values.*StationaryDistVec_kk);
                Mean2=sum(Values2.*StationaryDistVec_kk);
                StdDev1=sqrt(sum(StationaryDistVec_kk.*((Values-Mean1.*ones(N_a*(jend-j1+1),1)).^2)));
                StdDev2=sqrt(sum(StationaryDistVec_kk.*((Values2-Mean2.*ones(N_a*(jend-j1+1),1)).^2)));
                
                Numerator=sum((Values-Mean1*ones(N_a*(jend-j1+1),1,'gpuArray')).*(Values2-Mean2*ones(N_a*(jend-j1+1),1,'gpuArray')).*StationaryDistVec_kk);
                
                AgeConditionalStats(ii,aa).CrossSectionalCorrelation(kk)=Numerator/(StdDev1*StdDev2);
            end
        end
        
    end
end


if simoptions.crosssectioncorrelation==1
    % Just to make it easier to call the output later (turns upper triangular matrix into a full matrix)
    for kk=1:length(simoptions.agegroupings)
        for ff1=1:length(FnsToEvaluate)
            for ff2=ff1:length(FnsToEvaluate)
                if ff2==ff1
                    AgeConditionalStats(ff1,ff2).CrossSectionalCorrelation(kk)=1;
                else
                    AgeConditionalStats(ff2,ff1).CrossSectionalCorrelation(kk)=AgeConditionalStats(ff1,ff2).CrossSectionalCorrelation(kk); % Note, set (ff2,ff1) to (ff1,ff2)
                end
            end
        end
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
    if simoptions.crosssectioncorrelation==1
        for ff1=1:length(FnsToEvaluate)
            for ff2=ff1:length(FnsToEvaluate)
                AgeConditionalStats(ff2,ff1).CrossSectionalCorrelation=AgeConditionalStats2(ff2,ff1).CrossSectionalCorrelation;
            end
        end
    end
end



end

