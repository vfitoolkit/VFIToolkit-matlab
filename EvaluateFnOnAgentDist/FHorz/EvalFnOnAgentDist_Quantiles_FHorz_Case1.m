function  varargout=EvalFnOnAgentDist_Quantiles_FHorz_Case1(StationaryDist,PolicyIndexes, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,Parallel,simoptions)
% Returns the cut-off values and the within percentile means from dividing the StationaryDist into simoptions.nquantiles quantiles (so 4 gives quartiles, 5 gives quintiles, 100 gives percentiles).
%
% Uses digests.

if exist('Parallel','var')==0
    if isa(StationaryDist, 'gpuArray')
        Parallel=2;
    else
        Parallel=1;
    end
else
    if isempty(Parallel)
        if isa(StationaryDist, 'gpuArray')
            Parallel=2;
        else
            Parallel=1;
        end
    end
end

if ~exist('simoptions','var')
    simoptions.nquantiles=100;
else
    if ~isfield(simoptions,'nquantiles')
        simoptions.nquantiles=100;
    end
end

if isfield('simoptions','n_semiz') % If using semi-exogenous shocks
    n_z=[n_z,simoptions.n_semiz]; % For purposes of function evaluation we can just treat the semi-exogenous states as exogenous states
end

Tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.

% Note that to unnormalize the Lorenz Curve you can just multiply it be the
% AggVars for the same variable. This will give you the inverse cdf.

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

%% This implementation is slightly inefficient when shocks are not age dependent, but speed loss is fairly trivial
if exist('simoptions','var')
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
    if isfield(simoptions,'EiidShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.EiidShockFnParamNames=getAnonymousFnInputNames(simoptions.EiidShockFn);
    end
end
eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

if fieldexists_pi_z_J==1
    z_grid_J=simoptions.z_grid_J;
elseif fieldexists_ExogShockFn==1
    z_grid_J=zeros(sum(n_z),N_j);
    for jj=1:N_j
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=num2cell(ExogShockFnParamsVec);
            [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
        else
            [z_grid,~]=simoptions.ExogShockFn(jj);
        end
        z_grid_J(:,jj)=z_grid;
    end
else
    z_grid_J=repmat(z_grid,1,N_j);
end
if Parallel==2
    z_grid_J=gpuArray(z_grid_J);
end

if isfield(simoptions,'n_e')
    % Because of how FnsToEvaluate works I can just get the e variables and then 'combine' them with z
    eval('fieldexists_EiidShockFn=1;simoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
    eval('fieldexists_EiidShockFnParamNames=1;simoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
    eval('fieldexists_pi_e_J=1;simoptions.pi_e_J;','fieldexists_pi_e_J=0;')
    
    N_e=prod(simoptions.n_e);
    l_e=length(simoptions.n_e);
    
    if fieldexists_pi_e_J==1
        e_grid_J=simoptions.e_grid_J;
    elseif fieldexists_EiidShockFn==1
        e_grid_J=zeros(sum(simoptions.n_e),N_j);
        for jj=1:N_j
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=num2cell(EiidShockFnParamsVec);
                [e_grid,~]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [e_grid,~]=simoptions.EiidShockFn(jj);
            end
            e_grid_J(:,jj)=gather(e_grid);
        end
    else
        e_grid_J=repmat(simoptions.e_grid,1,N_j);
    end
    
    % Now combine into z
    if n_z(1)==0
        l_z=l_e;
        n_z=simoptions.n_e;
        z_grid_J=e_grid_J;
    else
        l_z=l_z+l_e;
        n_z=[n_z,simoptions.n_e];
        z_grid_J=[z_grid_J; e_grid_J];
    end
    N_z=prod(n_z);
        
end

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

%%
if Parallel==2

    QuantileCutOffs=zeros(length(FnsToEvaluate),simoptions.nquantiles+1,'gpuArray'); %Includes min and max
    QuantileMeans=zeros(length(FnsToEvaluate),simoptions.nquantiles,'gpuArray');
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);
    
    PolicyValues=PolicyInd2Val_FHorz_Case1(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]);
    for ii=1:length(FnsToEvaluate)
        Values=nan(N_a*N_z,N_j,'gpuArray');
        for jj=1:N_j
            z_grid=z_grid_J(:,jj);

            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj);
            end
            Values(:,jj)=reshape(EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{ii}, FnToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,Parallel),[N_a*N_z,1]);
        end
        Values=reshape(Values,[N_a*N_z*N_j,1]);

        [SortedValues,SortedValues_index] = sort(Values);
        SortedWeights = StationaryDistVec(SortedValues_index);
        
        CumSumSortedWeights=cumsum(SortedWeights);
        
        WeightedValues=Values.*StationaryDistVec;
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        QuantileIndexes_ii=zeros(1,simoptions.nquantiles-1,'gpuArray');
        QuantileCutoffs_ii=zeros(1,simoptions.nquantiles-1,'gpuArray');
        QuantileMeans_ii=zeros(1,simoptions.nquantiles,'gpuArray');
        for qq=1:simoptions.nquantiles-1
            [tempindex,~]=find(CumSumSortedWeights>=qq/simoptions.nquantiles,1,'first');
            QuantileIndexes_ii(qq)=tempindex;
            QuantileCutoffs_ii(qq)=SortedValues(tempindex);
            if qq==1
                QuantileMeans_ii(qq)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
            elseif (1<qq) && (qq<(simoptions.nquantiles-1))
                QuantileMeans_ii(qq)=sum(SortedWeightedValues(QuantileIndexes_ii(qq-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_ii(qq-1)));
            elseif qq==(simoptions.nquantiles-1)
                QuantileMeans_ii(qq)=sum(SortedWeightedValues(QuantileIndexes_ii(qq-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_ii(qq-1)));
                QuantileMeans_ii(qq+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
            end
        end
        
        % Min value
        tempindex=find(CumSumSortedWeights>=Tolerance,1,'first');
        minvalue=SortedValues(tempindex);
        % Max value
        tempindex=find(CumSumSortedWeights>=(1-Tolerance),1,'first');
        maxvalue=SortedValues(tempindex);
        
        QuantileCutOffs(ii,:)=[minvalue, QuantileCutoffs_ii, maxvalue];
        QuantileMeans(ii,:)=QuantileMeans_ii;
    end
    
else
    QuantileCutOffs=zeros(length(FnsToEvaluate),simoptions.nquantiles+1); %Includes min and max
    QuantileMeans=zeros(length(FnsToEvaluate),simoptions.nquantiles);
    if l_d>0
        d_val=zeros(1,l_d);
    end
    aprime_val=zeros(1,l_a);
    a_gridvals=CreateGridvals(n_a,a_grid,1);
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);
    
    sizePolicyIndexes=size(PolicyIndexes);
    if sizePolicyIndexes(2:end)~=[N_a,N_z,N_j] % If not in vectorized form
        PolicyIndexes=reshape(PolicyIndexes,[sizePolicyIndexes(1),N_a,N_z,N_j]);
    end
    dPolicy_gridvals=zeros(N_a*N_z,N_j);
    for jj=1:N_j
        dPolicy_gridvals(:,jj)=CreateGridvals_Policy(PolicyIndexes(:,:,jj),n_d,[],n_a,n_z,d_grid,[],2,1);
    end
    
    for ii=1:length(FnsToEvaluate)
        Values=zeros(N_a,N_z,N_j);
        if l_d==0
            for jj=1:N_j
                if fieldexists_ExogShockFn==1
                    z_grid=z_grid_J(:,jj);
                    z_gridvals=CreateGridvals(n_z,z_grid,2);
                end
                
                [~, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
                if ~isempty(FnsToEvaluateParamNames(qq).Names)
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(qq).Names,jj));
                end
                for a_c=1:N_a
                    for z_c=1:N_z
                        % Includes check for cases in which no parameters are actually required
                        if isempty(FnsToEvaluateParamNames(qq).Names)
                             Values(a_c,z_c,jj)=FnsToEvaluate{qq}(aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:});
                        else
                             Values(a_c,z_c,jj)=FnsToEvaluate{qq}(aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end
            end
        else
            for jj=1:N_j
                if fieldexists_ExogShockFn==1
                    z_grid=z_grid_J(:,jj);
                    z_gridvals=CreateGridvals(n_z,z_grid,2);
                end

                [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
                if ~isempty(FnsToEvaluateParamNames(qq).Names)
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(qq).Names,jj));
                end
                for a_c=1:N_a
                    for z_c=1:N_z
                        % Includes check for cases in which no parameters are actually required
                        if isempty(FnsToEvaluateParamNames(qq).Names)
                             Values(a_c,z_c,jj)=FnsToEvaluate{qq}(d_gridvals{a_c+(z_c-1)*N_a,:},aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:});
                        else
                             Values(a_c,z_c,jj)=FnsToEvaluate{qq}(d_gridvals{a_c+(z_c-1)*N_a,:},aprime_gridvals{a_c+(z_c-1)*N_a,:},a_gridvals{a_c,:},z_gridvals{z_c,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end
            end
        end
        Values=reshape(Values,[N_a*N_z*N_j,1]);
        
        [SortedValues,SortedValues_index] = sort(Values);
        SortedWeights = StationaryDistVec(SortedValues_index);
        
        CumSumSortedWeights=cumsum(SortedWeights);
        
        WeightedValues=Values.*StationaryDistVec;
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        QuantileIndexes_ii=zeros(1,simoptions.nquantiles-1);
        QuantileCutoffs_ii=zeros(1,simoptions.nquantiles-1);
        QuantileMeans_ii=zeros(1,simoptions.nquantiles);
        for qq=1:simoptions.nquantiles-1
            [tempindex,~]=find(CumSumSortedWeights>=qq/simoptions.nquantiles,1,'first');
            QuantileIndexes_ii(qq)=tempindex;
            QuantileCutoffs_ii(qq)=SortedValues(tempindex);
            if qq==1
                QuantileMeans_ii(qq)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
            elseif (1<qq) && (qq<(simoptions.nquantiles-1))
                QuantileMeans_ii(qq)=sum(SortedWeightedValues(QuantileIndexes_ii(qq-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_ii(qq-1)));
            elseif qq==(simoptions.nquantiles-1)
                QuantileMeans_ii(qq)=sum(SortedWeightedValues(QuantileIndexes_ii(qq-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_ii(qq-1)));
                QuantileMeans_ii(qq+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
            end
        end
        
        % Min value
        tempindex=find(CumSumSortedWeights>=Tolerance,1,'first');
        minvalue=SortedValues(tempindex);
        % Max value
        tempindex=find(CumSumSortedWeights>=(1-Tolerance),1,'first');
        maxvalue=SortedValues(tempindex);
        
        QuantileCutOffs(ii,:)=[minvalue, QuantileCutoffs_ii, maxvalue];
        QuantileMeans(ii,:)=QuantileMeans_ii;
    end
    
end

varargout={QuantileCutOffs, QuantileMeans};

%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    QuantileCutOffs2=QuantileCutOffs;
    QuantileMeans2=QuantileMeans;
    clear AggVars
    Quantiles=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        Quantiles.(AggVarNames{ff}).CutOffs=QuantileCutOffs2(ff,:);
        Quantiles.(AggVarNames{ff}).Means=QuantileMeans2(ff,:);
    end
    varargout={Quantiles};
end




end