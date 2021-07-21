function AgeConditionalStats=LifeCycleProfiles_FHorz_Case2(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,options, AgeDependentGridParamNames)
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

%% Check which option have been declared, set all others to defaults 
if exist('options','var')==1
    %Check options for missing fields, if there are some fill them with the defaults
    if isfield(options,'parallel')==0
        options.parallel=2;
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
    if isfield(options,'agedependentgrids')==0
        options.agedependentgrids=0;
    end
    if isfield(options,'crosssectioncorrelation')==0
        options.crosssectioncorrelation=0;
    end    
else
    %If options is not given, just use all the defaults
    options.parallel=2;
    options.verbose=0;
    options.nquantiles=20; % by default gives ventiles
    options.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    options.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    options.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    options.agedependentgrids=0;
    options.crosssectioncorrelation=0;
end

if prod(options.agedependentgrids)~=0
    % Note d_grid is actually d_gridfn
    % Note a_grid is actually a_gridfn
    % Note z_grid is actually z_gridfn
    % Note pi_z is actually AgeDependentGridParamNames
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case2_AgeDepGrids(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,options, AgeDependentGridParamNames);
    return
end

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);

if options.parallel~=2 % Just make sure
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
end

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

%% Create a different 'Values' for each of the variable to be evaluated
PolicyValues=PolicyInd2Val_FHorz_Case2(Policy,n_d,n_a,n_z,N_j,d_grid,options.parallel);
permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]

StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);

% Do some preallocation of the output structure
ngroups=length(options.agegroupings);
AgeConditionalStats(length(FnsToEvaluate)).Mean=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).Median=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).Variance=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(options.npoints,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(options.nquantiles+1,ngroups,'gpuArray'); % Includes the min and max values
AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(options.nquantiles,ngroups,'gpuArray');

if options.crosssectioncorrelation==1
    AgeConditionalStats(length(FnsToEvaluate),length(FnsToEvaluate)).CrossSectionalCorrelation=nan(1,ngroups,'gpuArray');
end

PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*l_d,N_j]); % I reshape here, and THEN JUST RESHAPE AGAIN WHEN USING. THIS IS STUPID AND SLOW.
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
            if fieldexists_ExogShockFn==1
                if fieldexists_ExogShockFnParamNames==1
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                    [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsVec);
                else
                    [z_grid,~]=simoptions.ExogShockFn(jj);
                end
            end
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
                FnsToEvaluateParamsVec=[];
            else
                FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj);
            end            
            Values(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case2(FnsToEvaluate{ii}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d]),n_d,n_a,n_z,a_grid,z_grid,options.parallel),[N_a*N_z,1]);
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

        
        if options.crosssectioncorrelation==1 % Not coded the fastest way, but will use less memory than keeping round all of the older 'Values'
            for aa=(ii+1):length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
                Values2=nan(N_a*N_z,jend-j1+1,'gpuArray'); % Preallocate
                for jj=j1:jend
                    if fieldexists_ExogShockFn==1
                        if fieldexists_ExogShockFnParamNames==1
                            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                            [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsVec);
                        else
                            [z_grid,~]=simoptions.ExogShockFn(jj);
                        end
                    end
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
                        FnsToEvaluateParamsVec=[];
                    else
                        FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(aa).Names,jj);
                    end
                    Values2(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case2(FnsToEvaluate{aa}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,options.parallel),[N_a*N_z,1]);
                end
                
                Values2=reshape(Values2,[N_a*N_z*(jend-j1+1),1]);
                
                SSvalues_Mean1=sum(Values.*StationaryDistVec_kk);
                SSvalues_Mean2=sum(Values2.*StationaryDistVec_kk);
                SSvalues_StdDev1=sqrt(sum(StationaryDistVec_kk.*((Values-SSvalues_Mean1.*ones(N_a*N_z*(jend-j1+1),1)).^2)));
                SSvalues_StdDev2=sqrt(sum(StationaryDistVec_kk.*((Values2-SSvalues_Mean2.*ones(N_a*N_z*(jend-j1+1),1)).^2)));
                
                Numerator=sum((Values-SSvalues_Mean1*ones(N_a*N_z*(jend-j1+1),1,'gpuArray')).*(Values2-SSvalues_Mean2*ones(N_a*N_z*(jend-j1+1),1,'gpuArray')).*StationaryDistVec_kk);
                
                AgeConditionalStats(ii,aa).CrossSectionalCorrelation(kk)=Numerator/(SSvalues_StdDev1*SSvalues_StdDev2);
            end
        end
        
    end
end


if options.crosssectioncorrelation==1
    % Just to make it easier to call the output later (turns upper triangular matrix into a full matrix)
    for kk=1:length(options.agegroupings)
        for ii=1:length(FnsToEvaluate)
            for aa=ii:length(FnsToEvaluate)
                if aa==ii
                    AgeConditionalStats(ii,aa).CrossSectionalCorrelation(kk)=1;
                else
                    AgeConditionalStats(aa,ii).CrossSectionalCorrelation(kk)=AgeConditionalStats(ii,aa).CrossSectionalCorrelation(kk); % Note, set (aa,ii) to (ii,aa)
                end
            end
        end
    end
end

end

