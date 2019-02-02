function AgeConditionalStats=LifeCycleProfiles_FHorz_Case2_AgeDepGrids(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_gridfn,a_gridfn,z_gridfn,options, AgeDependentGridParamNames)
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
% AgeConditionalStats(length(FnsToEvaluate)).StdDeviation=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(ngroups,options.npoints);
% AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(ngroups,options.npoints);
% AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(ngroups,options.npoints);
% AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(ngroups,options.nquantiles+1); % Includes the min and max values
% AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(ngroups,options.nquantiles);
    
daz_gridstructure=AgeDependentGrids_Create_daz_gridstructure(n_d,n_a,n_z,N_j,d_gridfn, a_gridfn, z_gridfn, AgeDependentGridParamNames, Parameters, options);
% Creates daz_gridstructure which contains both the grids themselves and a
% bunch of info about the grids in an easy to access way.
% e.g., the d_grid for age j=10: daz_gridstructure.d_grid.j010
% e.g., the value of N_a for age j=5: daz_gridstructure.N_a.j005
% e.g., the zprime_grid for age j=20: daz_gridstructure.zprime_grid.j020
% e.g., the jstr for age j=15: daz_gridstructure.jstr(15)='j015'

%% Create a different 'Values' for each of the variable to be evaluated
StationaryDistVec=struct();
for jj=1:N_j
    jstr=daz_gridstructure.jstr{jj};
    n_d_j=daz_gridstructure.n_d.(jstr(:));
    n_a_j=daz_gridstructure.n_a.(jstr(:));
    n_z_j=daz_gridstructure.n_z.(jstr(:));
    N_a_j=daz_gridstructure.N_a.(jstr(:));
    N_z_j=daz_gridstructure.N_z.(jstr(:));
    d_grid_j=daz_gridstructure.d_grid.(jstr(:));
%     a_grid_j=daz_gridstructure.a_grid.(jstr(:));
    if options.parallel~=2
        PolicyKronTemp=KronPolicyIndexes_Case2(Policy.(jstr(:)), n_d_j', n_a_j', n_z_j');%,simoptions); % Note, use Case2 without the FHorz as I have to do this seperately for each age j in any case.
        PolicyValuestemp=PolicyInd2Val_Case2(gather(PolicyKronTemp),n_d_j,n_a_j,n_z_j,gather(d_grid_j),options.parallel);
    else
        PolicyValuestemp=PolicyInd2Val_Case2(Policy.(jstr(:)),n_d_j,n_a_j,n_z_j,d_grid_j,options.parallel);
    end
    l_d=length(n_d_j);
    l_a=length(n_a_j);
    l_z=length(n_z_j);
    if options.parallel~=2 % The permute indexes follow from PolicyInd2Val_Case2()
        permuteindexes=[2,3,1];
    else
        permuteindexes=[1+(1:1:(l_a+l_z)),1];
    end
    PolicyValuesPermute.(jstr(:))=reshape(permute(PolicyValuestemp,permuteindexes),[n_a_j,n_z_j,l_d]); % The reshape may be redundant (if on options.parallel=2)
    StationaryDistVec.(jstr(:))=reshape(StationaryDist.(jstr(:)),[N_a_j*N_z_j,1]);
end

options.parallel=2; % For the present this command just assumes/imposes this. Need to set it as otherwise the ValuesOnSSGrid_Case2() command complains.

% Do some preallocation of the output structure
ngroups=length(options.agegroupings);
AgeConditionalStats(length(FnsToEvaluate)).Mean=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).Median=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).Variance=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).StdDeviation=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(ngroups,options.npoints,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(options.npoints,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(options.npoints,ngroups,'gpuArray');
AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(options.nquantiles+1,ngroups,'gpuArray'); % Includes the min and max values
AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(options.nquantiles,ngroups,'gpuArray');

% PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]); % [n_a,n_z,l_d+l_a]
for kk=1:length(options.agegroupings)
    j1=options.agegroupings(kk);
    if kk<length(options.agegroupings)
        jend=options.agegroupings(kk+1)-1;
    else
        jend=N_j;
    end
    
    N_a_k=1;
    N_z_k=1;
    for jj=j1:jend
        jstr=daz_gridstructure.jstr{jj};
        N_a_k=N_a_k*daz_gridstructure.N_a.(jstr(:));
        N_z_k=N_z_k*daz_gridstructure.N_z.(jstr(:));
    end
    StationaryDistVec_kk=zeros(N_a_k*N_z_k,1,'gpuArray');
    N_az_c=0;
    for jj=j1:jend
        jstr=daz_gridstructure.jstr{jj};
        N_a_j=daz_gridstructure.N_a.(jstr(:));
        N_z_j=daz_gridstructure.N_z.(jstr(:));    
        StationaryDistVec_kk((N_az_c+1):(N_az_c+N_a_j*N_z_j))=StationaryDistVec.(jstr(:));
        N_az_c=N_az_c+N_a_j*N_z_j;
    end
    StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDist.AgeWeights(j1:jend)); % Normalize to sum to one for this 'agegrouping'
    
    for ii=1:length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
        Values=nan(N_a_k*N_z_k,1,'gpuArray'); % Preallocate
        N_az_c=0;
        for jj=j1:jend
            jstr=daz_gridstructure.jstr{jj};
            N_a_j=daz_gridstructure.N_a.(jstr(:));
            N_z_j=daz_gridstructure.N_z.(jstr(:));
            n_d_j=daz_gridstructure.n_d.(jstr(:));
            n_a_j=daz_gridstructure.n_a.(jstr(:));
            n_z_j=daz_gridstructure.n_z.(jstr(:));
            a_grid_j=daz_gridstructure.a_grid.(jstr(:));
            z_grid_j=daz_gridstructure.z_grid.(jstr(:));
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
                FnsToEvaluateParamsVec=[];
            else
                FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj);
            end            
            Values((N_az_c+1):(N_az_c+N_a_j*N_z_j))=reshape(ValuesOnSSGrid_Case2(FnsToEvaluate{ii}, FnsToEvaluateParamsVec,PolicyValuesPermute.(jstr(:)),n_d_j,n_a_j,n_z_j,a_grid_j,z_grid_j,options.parallel),[N_a_j*N_z_j,1]);
            N_az_c=N_az_c+N_a_j*N_z_j;
        end
        
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
        AgeConditionalStats(ii).Variance(kk)=sum(StationaryDistVec_kk.*((Values-(AgeConditionalStats(ii).Mean(kk).*ones(N_a_k*N_z_k,1))).^2));
        % Calculate the 'age conditional' standard deviation
        AgeConditionalStats(ii).StdDeviation(kk)=sqrt(AgeConditionalStats(ii).Variance(kk));
        
        % Calculate the 'age conditional' lorenz curve
        AgeConditionalStats(ii).LorenzCurve(:,kk)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,options.npoints);                
        % Calculate the 'age conditional' gini
        AgeConditionalStats(ii).Gini(kk)=Gini_from_LorenzCurve(AgeConditionalStats(ii).LorenzCurve(:,kk));
        
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


