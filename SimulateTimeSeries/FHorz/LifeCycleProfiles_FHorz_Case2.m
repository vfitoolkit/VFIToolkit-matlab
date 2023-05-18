function AgeConditionalStats=LifeCycleProfiles_FHorz_Case2(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions, AgeDependentGridParamNames)
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
if exist('simoptions','var')==1
    %Check options for missing fields, if there are some fill them with the defaults
    simoptions.parallel=2; % Case2 is only on gpu
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
    if isfield(simoptions,'agedependentgrids')==0
        simoptions.agedependentgrids=0;
    end
    if isfield(simoptions,'crosssectioncorrelation')==0
        simoptions.crosssectioncorrelation=0;
    end    
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
    if isfield(simoptions,'EiidShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.EiidShockFnParamNames=getAnonymousFnInputNames(simoptions.EiidShockFn);
    end
else
    %If options is not given, just use all the defaults
    simoptions.parallel=2; % Case2 is only on gpu
    simoptions.verbose=0;
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.agedependentgrids=0;
    simoptions.crosssectioncorrelation=0;
end

if prod(simoptions.agedependentgrids)~=0
    % Note d_grid is actually d_gridfn
    % Note a_grid is actually a_gridfn
    % Note z_grid is actually z_gridfn
    % Note pi_z is actually AgeDependentGridParamNames
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case2_AgeDepGrids(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions, AgeDependentGridParamNames);
    return
end

if simoptions.parallel==2
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
end

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);

if simoptions.parallel~=2 % Just make sure
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
end

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% z_grid (and e_grid where appropriate)
eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

if fieldexists_pi_z_J==1
    z_grid_J=simoptions.z_grid_J;
elseif fieldexists_ExogShockFn==1
    if size(z_grid,2)==1 % kronecker-grid
        z_grid_J=zeros(sum(n_z),N_j);
        for jj=1:N_j
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            else
                [z_grid,~]=simoptions.ExogShockFn(jj);
            end
            z_grid_J(:,jj)=gather(z_grid);
        end
    elseif size(z_grid,2)==l_z % joint-grids
        z_grid_J=zeros(N_z,l_z,N_j);
        for jj=1:N_j
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            else
                [z_grid,~]=simoptions.ExogShockFn(jj);
            end
            z_grid_J(:,:,jj)=gather(z_grid);
        end
    end
else
    if size(z_grid,2)==1 % kronecker-grid
        z_grid_J=repmat(z_grid,1,N_j);
    elseif size(z_grid,2)==l_z % joint-grids
        z_grid_J=zeros(N_z,l_z,N_j);
        for jj=1:N_j
            z_grid_J(:,:,jj)=z_grid;
        end
    end
end
if ndims(z_grid_J)==2
    jointzgrid=0;
elseif ndims(z_grid_J)==3
    jointzgrid=1;
end

if isfield(simoptions,'SemiExoStateFn') % If using semi-exogenous shocks
    % For purposes of function evaluation we can just treat the semi-exogenous states as exogenous states
    n_z=[n_z,simoptions.n_semiz];
    l_z=length(n_z);
    N_z=prod(n_z);
    if jointzgrid==0
        z_grid_Jold=z_grid_J;
        z_grid_J=zeros(sum(n_z),N_j);
        for jj=1:N_j
            z_grid_J(:,jj)=[z_grid_Jold(:,jj);simoptions.semiz_grid];
        end
    else % jointzgrid==1
        error('Have not implemented joint z_grid being used at same time as semi-exogenous states')
    end
end

if isfield(simoptions,'n_e')
    % Because of how FnsToEvaluate works I can just get the e variables and
    % then 'combine' them with z
    eval('fieldexists_EiidShockFn=1;simoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
    eval('fieldexists_EiidShockFnParamNames=1;simoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
    eval('fieldexists_pi_e_J=1;simoptions.pi_e_J;','fieldexists_pi_e_J=0;')
    
    n_e=simoptions.n_e;
    N_e=prod(n_e);
    l_e=length(n_e);
    
    if fieldexists_pi_e_J==1
        e_grid_J=simoptions.e_grid_J;
    elseif fieldexists_EiidShockFn==1
        if size(simoptions.e_grid,2)==1 % kronecker-grid
            e_grid_J=zeros(sum(simoptions.n_e),N_j);
            for jj=1:N_j
                if fieldexists_EiidShockFnParamNames==1
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [e_grid,~]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
                else
                    [e_grid,~]=simoptions.EiidShockFn(jj);
                end
                e_grid_J(:,jj)=gather(e_grid);
            end
        elseif size(simoptions.e_grid,2)==l_e % joint-grids
            e_grid_J=zeros(N_e,l_e,N_j);
            for jj=1:N_j
                if fieldexists_EiidShockFnParamNames==1
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [e_grid,~]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
                else
                    [e_grid,~]=simoptions.EiidShockFn(jj);
                end
                e_grid_J(:,:,jj)=gather(e_grid);
            end
        end
    else
        if size(simoptions.e_grid,2)==1 % kronecker-grid
            e_grid_J=repmat(simoptions.e_grid,1,N_j);
        elseif size(simoptions.e_grid,2)==l_z % joint-grids
            e_grid_J=zeros(N_e,l_e,N_j);
            for jj=1:N_j
                e_grid_J(:,:,jj)=simoptions.e_grid;
            end
        end
    end
    
    if ndims(e_grid_J)==2
        jointegrid=0;
    elseif ndims(e_grid_J)==3
        jointegrid=1;
    end

    
    % Now combine into z
    if n_z(1)==0
        l_ze=l_e;
        n_ze=simoptions.n_e;
    else
        l_ze=l_z+l_e;
        n_ze=[n_z,n_e];
    end
    N_ze=prod(n_ze);
else
    N_e=0;
    n_ze=n_z;
    N_ze=N_z;
    l_ze=l_z;
end


%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
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
PolicyValues=PolicyInd2Val_FHorz_Case2(Policy,n_d,n_a,n_z,N_j,d_grid);
permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]

StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);

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

PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*l_d,N_j]); % I reshape here, and THEN JUST RESHAPE AGAIN WHEN USING. THIS IS STUPID AND SLOW.
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
            z_grid=z_grid_J(:,jj);
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
                FnsToEvaluateParamsVec=[];
            else
                FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj);
            end            
            Values(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case2(FnsToEvaluate{ii}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d]),n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel),[N_a*N_z,1]);
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
                Values2=nan(N_a*N_z,jend-j1+1,'gpuArray'); % Preallocate
                for jj=j1:jend
                    z_grid=z_grid_J(:,jj);
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
                        FnsToEvaluateParamsVec=[];
                    else
                        FnsToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(aa).Names,jj);
                    end
                    Values2(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case2(FnsToEvaluate{aa}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel),[N_a*N_z,1]);
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

