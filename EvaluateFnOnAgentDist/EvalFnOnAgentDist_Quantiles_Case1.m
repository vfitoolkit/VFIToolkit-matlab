function [QuantileCutOffs, QuantileMeans]=EvalFnOnAgentDist_Quantiles_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, NumQuantiles, n_d, n_a, n_z, d_grid, a_grid, z_grid,Parallel)
%Returns the cut-off values and the within percentile means from dividing
%the StationaryDist into NumPercentiles percentiles.
%
% Parallel is an optional input

if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
elseif isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0);
end

Tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.

%Note that to unnormalize the Lorenz Curve you can just multiply it be the
%SSvalues_AggVars for the same variable. This will give you the inverse
%cdf.

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

if Parallel==2    
    QuantileCutOffs=zeros(length(FnsToEvaluate),NumQuantiles+1,'gpuArray'); %Includes min and max
    QuantileMeans=zeros(length(FnsToEvaluate),NumQuantiles,'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names);
        end
        
        Values=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        
        [SortedValues,SortedValues_index] = sort(Values);
        SortedWeights = StationaryDistVec(SortedValues_index);
        
        CumSumSortedWeights=cumsum(SortedWeights);
        
        WeightedValues=Values.*StationaryDistVec;
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        QuantileIndexes=zeros(1,NumQuantiles-1,'gpuArray');
        QuantileCutoffs=zeros(1,NumQuantiles-1,'gpuArray');
        QuantileMeans=zeros(1,NumQuantiles,'gpuArray');
        for ii=1:NumQuantiles-1
            [~,tempindex]=find(CumSumSortedWeights>=ii/NumQuantiles,1,'first');
            QuantileIndexes(ii)=tempindex;
            QuantileCutoffs(ii)=SortedValues(tempindex);
            if ii==1
                QuantileMeans(ii)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
            elseif (1<ii) && (ii<(NumQuantiles-1))
                QuantileMeans(ii)=sum(SortedWeightedValues(QuantileIndexes(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ii-1)));
            elseif ii==(NumQuantiles-1)
                QuantileMeans(ii)=sum(SortedWeightedValues(QuantileIndexes(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ii-1)));
                QuantileMeans(ii+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
            end
        end
        
        % Min value
        [~,tempindex]=find(CumSumSortedWeights>=Tolerance,1,'first');
        minvalue=SortedValues(tempindex);
        % Max value
        [~,tempindex]=find(CumSumSortedWeights>=(1-Tolerance),1,'first');
        maxvalue=SortedValues(tempindex);
        
        QuantileCutOffs(i,:)=[minvalue, QuantileCutoffs, maxvalue];
        QuantileMeans(i,:)=QuantileMeans;
    end
    
else
    QuantileCutOffs=zeros(length(FnsToEvaluate),NumQuantiles+1); %Includes min and max
    QuantileMeans=zeros(length(FnsToEvaluate),NumQuantiles);
%     d_val=zeros(l_d,1);
%     aprime_val=zeros(l_a,1);
%     a_val=zeros(l_a,1);
%     s_val=zeros(l_z,1);
%     
%     PolicyIndexesKron=reshape(PolicyIndexes,[l_d+l_a,N_a,N_z]);
    
%     [d_gridvals, aprime_gridvals, a_gridvals, z_gridvals]=CreateGridvals(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,z_grid,1,2);
    [d_gridvals, ~]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,2, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    for i=1:length(FnsToEvaluate)
%         Values=zeros(N_a,N_z);
%         % Includes check for cases in which no parameters are actually required
%         if isempty(FnsToEvaluateParamNames) % check for 'SSvalueParamNames={}'
%             if l_d==0
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_z
%                         s_ind=ind2sub_homemade([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 s_val(jj2)=z_grid(s_ind(jj2));
%                             else
%                                 s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         d_val=0;
%                         aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
%                         for kk=1:l_a
%                             if kk==1
%                                 aprime_val(kk)=a_grid(aprime_ind(kk));
%                             else
%                                 aprime_val(kk)=a_grid(aprime_ind(kk)+sum(n_a(1:kk-1)));
%                             end
%                         end
%                         Values(j1,j2)=FnsToEvaluate{i}(aprime_val,a_val,s_val);
%                     end
%                 end
%             else
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_z
%                         s_ind=ind2sub_homemade([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 s_val(jj2)=z_grid(s_ind(jj2));
%                             else
%                                 s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         d_ind=PolicyIndexesKron(1:l_d,j1,j2);
%                         for kk=1:l_d
%                             if kk==1
%                                 d_val(kk)=d_grid(d_ind(kk));
%                             else
%                                 d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
%                             end
%                         end
%                         aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
%                         for kk=1:l_a
%                             if kk==1
%                                 aprime_val(kk)=a_grid(aprime_ind(kk));
%                             else
%                                 aprime_val(kk)=a_grid(aprime_ind(kk)+sum(n_a(1:kk-1)));
%                             end
%                         end
%                         Values(j1,j2)=FnsToEvaluate{i}(d_val,aprime_val,a_val,s_val);
%                     end
%                 end
%             end
%         else
%             FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names);
%             if l_d==0
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_z
%                         s_ind=ind2sub_homemade([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 s_val(jj2)=z_grid(s_ind(jj2));
%                             else
%                                 s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         d_val=0;
%                         aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
%                         for kk=1:l_a
%                             if kk==1
%                                 aprime_val(kk)=a_grid(aprime_ind(kk));
%                             else
%                                 aprime_val(kk)=a_grid(aprime_ind(kk)+sum(n_a(1:kk-1)));
%                             end
%                         end
%                         Values(j1,j2)=FnsToEvaluate{i}(aprime_val,a_val,s_val,FnToEvaluateParamsVec);
%                     end
%                 end
%             else
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_z
%                         s_ind=ind2sub_homemade([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 s_val(jj2)=z_grid(s_ind(jj2));
%                             else
%                                 s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         d_ind=PolicyIndexesKron(1:l_d,j1,j2);
%                         for kk=1:l_d
%                             if kk==1
%                                 d_val(kk)=d_grid(d_ind(kk));
%                             else
%                                 d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
%                             end
%                         end
%                         aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
%                         for kk=1:l_a
%                             if kk==1
%                                 aprime_val(kk)=a_grid(aprime_ind(kk));
%                             else
%                                 aprime_val(kk)=a_grid(aprime_ind(kk)+sum(n_a(1:kk-1)));
%                             end
%                         end
%                         Values(j1,j2)=FnsToEvaluate{i}(d_val,aprime_val,a_val,s_val,FnToEvaluateParamsVec);
%                     end
%                 end
%             end
%         end
%         
%         Values=reshape(Values,[N_a*N_z,1]);
% Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names) % check for 'FnsToEvaluateParamNames={}'
            Values=zeros(N_a*N_z,1);
            if l_d==0
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(aprime_val,a_val,z_val);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
            else % l_d>0
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    %                     a_val=a_gridvals{j1,:};
                    %                     z_val=z_gridvals{j2,:};
                    %                     d_val=d_gridvals{j1+(j2-1)*N_a,:};
                    %                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
                    %                     Values(ii)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
            end
        else
            Values=zeros(N_a*N_z,1);
            if l_d==0
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(aprime_val,a_val,z_val,SSvalueParamsVec);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
            else % l_d>0
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     d_val=d_gridvals{j1+(j2-1)*N_a,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val,SSvalueParamsVec);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
            end
        end
        
        [SortedValues,SortedValues_index] = sort(Values);
        SortedWeights = StationaryDistVec(SortedValues_index);
        
        CumSumSortedWeights=cumsum(SortedWeights);
        
        WeightedValues=Values.*StationaryDistVec;
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        QuantileIndexes=zeros(1,NumQuantiles-1);
        QuantileCutoffs=zeros(1,NumQuantiles-1);
        QuantileMeans=zeros(1,NumQuantiles);
        for ii=1:NumQuantiles-1
            [~,tempindex]=find(CumSumSortedWeights>=ii/NumQuantiles,1,'first');
            QuantileIndexes(ii)=tempindex;
            QuantileCutoffs(ii)=SortedValues(tempindex);
            if ii==1
                QuantileMeans(ii)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
            elseif (1<ii) && (ii<(NumQuantiles-1))
                QuantileMeans(ii)=sum(SortedWeightedValues(QuantileIndexes(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ii-1)));
            elseif ii==(NumQuantiles-1)
                QuantileMeans(ii)=sum(SortedWeightedValues(QuantileIndexes(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ii-1)));
                QuantileMeans(ii+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
            end
        end
        
        % Min value
        [~,tempindex]=find(CumSumSortedWeights>=Tolerance,1,'first');
        minvalue=SortedValues(tempindex);
        % Max value
        [~,tempindex]=find(CumSumSortedWeights>=(1-Tolerance),1,'first');
        maxvalue=SortedValues(tempindex);
        
        QuantileCutOffs(i,:)=[minvalue, QuantileCutoffs, maxvalue];
        QuantileMeans(i,:)=QuantileMeans;
    end
    
end


end

