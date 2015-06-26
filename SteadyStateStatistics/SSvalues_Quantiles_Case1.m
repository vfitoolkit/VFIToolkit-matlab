function [SSvalues_QuantileCutOffs, SSvalues_QuantileMeans]=SSvalues_Quantiles_Case1(SteadyStateDist, PolicyIndexes, SSvaluesFn, SSvalueParams, NumQuantiles, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p_val,Parallel)
%Returns the cut-off values and the within percentile means from dividing
%the SteadyStateDist into NumPercentiles percentiles.

Tolerance=10^(-12); % Numerical tolerance used when calculated min and max values.

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

% Check if the SSvaluesFn depends on pi_s, if not then can do it all much
% faster. (I have been unable to figure out how to really take advantage of GPU
% when there is pi_s).
nargin_vec=zeros(numel(SSvaluesFn),1);
for ii=1:numel(SSvaluesFn)
    nargin_vec(ii)=nargin(SSvaluesFn{ii});
end
if max(nargin_vec)==(l_d+2*l_a+l_z+1+length(SSvalueParams)) && Parallel==2
    [SSvalues_QuantileCutOffs, SSvalues_QuantileMeans]=SSvalues_Quantiles_Case1_NoPi(SteadyStateDist, PolicyIndexes, SSvaluesFn, SSvalueParams, NumQuantiles, n_d, n_a, n_z, d_grid, a_grid, z_grid, p_val, Parallel);
    return 
end

if Parallel~=2
    SSvalues_QuantileCutOffs=zeros(length(SSvaluesFn),NumQuantiles+1); %Includes min and max
    SSvalues_QuantileMeans=zeros(length(SSvaluesFn),NumQuantiles);
    d_val=zeros(l_d,1);
    aprime_val=zeros(l_a,1);
    a_val=zeros(l_a,1);
    s_val=zeros(l_z,1);
    PolicyIndexesKron=reshape(PolicyIndexes,[l_d+l_a,N_a,N_z]);
    SteadyStateDistVec=reshape(SteadyStateDist,[N_a*N_z,1]);
    for i=1:length(SSvaluesFn)
        Values=zeros(N_a,N_z);
        for j1=1:N_a
            a_ind=ind2sub_homemade([n_a],j1);
            for jj1=1:l_a
                if jj1==1
                    a_val(jj1)=a_grid(a_ind(jj1));
                else
                    a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
                end
            end
            for j2=1:N_z
                s_ind=ind2sub_homemade([n_z],j2);
                for jj2=1:l_z
                    if jj2==1
                        s_val(jj2)=z_grid(s_ind(jj2));
                    else
                        s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                    end
                end
                if l_d==0
                    d_val=0;
                else
                    d_ind=PolicyIndexesKron(1:l_d,j1,j2);
                    for kk=1:l_d
                        if kk==1
                            d_val(kk)=d_grid(d_ind(kk));
                        else
                            d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
                        end
                    end
                end
                aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
                for kk=1:l_a
                    if kk==1
                        aprime_val(kk)=a_grid(aprime_ind(kk));
                    else
                        aprime_val(kk)=a_grid(aprime_ind(kk)+sum(n_a(1:kk-1)));
                    end
                end
                Values(j1,j2)=SSvaluesFn{i}(d_val,aprime_val,a_val,s_val,pi_z,p_val);
            end
        end
        
        Values=reshape(Values,[N_a*N_z,1]);
        
        [SortedValues,SortedValues_index] = sort(Values);
        SortedWeights = SteadyStateDistVec(SortedValues_index);
        
        CumSumSortedWeights=cumsum(SortedWeights);
        
        WeightedValues=Values.*SteadyStateDistVec;
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        QuantileIndexes=zeros(1,NumQuantiles-1);
        QuantileCufoffs=zeros(1,NumQuantiles-1);
        QuantileMeans=zeros(1,NumQuantiles);
        for ii=1:NumQuantiles-1
            [~,tempindex]=min(CumSumSortedWeights,ii/NumQuantiles);
            QuantileIndexes(ii)=tempindex;
            QuantileCufoffs(ii)=SortedValues(tempindex);
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
        temp=(CumSumSortedWeights>=Tolerance);
        [~,tempindex]=min(CumSumSortedWeights,ii/NumQuantiles);
        minvalue=SortedValues(tempindex);
        % Max value
        temp=(CumSumSortedWeights>=(1-Tolerance));
        [~,tempindex]=max(CumSumSortedWeights,ii/NumQuantiles);
        maxvalue=SortedValues(tempindex);
        
        SSvalues_QuantileCutOffs(i,:)=[minvalue, QuantileCutoffs, maxvalue]; 
        SSvalues_QuantileMeans(i,:)=QuantileMeans;
    end
    
else %Parallel==2
    SSvalues_QuantileCutOffs=zeros(length(SSvaluesFn),NumQuantiles+1,'gpuArray'); %Includes min and max
    SSvalues_QuantileMeans=zeros(length(SSvaluesFn),NumQuantiles,'gpuArray');
    d_val=zeros(l_d,1,'gpuArray');
    aprime_val=zeros(l_a,1,'gpuArray');
    a_val=zeros(l_a,1,'gpuArray');
    s_val=zeros(l_z,1,'gpuArray');
    PolicyIndexesKron=reshape(PolicyIndexes,[l_d+l_a,N_a,N_z]);
    SteadyStateDistVec=reshape(SteadyStateDist,[N_a*N_z,1]);
    for i=1:length(SSvaluesFn)
        Values=zeros(N_a,N_z,'gpuArray');
        for j1=1:N_a
            a_ind=ind2sub_homemade_gpu([n_a],j1);
            for jj1=1:l_a
                if jj1==1
                    a_val(jj1)=a_grid(a_ind(jj1));
                else
                    a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
                end
            end
            for j2=1:N_z
                s_ind=ind2sub_homemade_gpu([n_z],j2);
                for jj2=1:l_z
                    if jj2==1
                        s_val(jj2)=z_grid(s_ind(jj2));
                    else
                        s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                    end
                end
                if l_d==0
                    d_val=0;
                else
                    d_ind=PolicyIndexesKron(1:l_d,j1,j2);
                    for kk=1:l_d
                        if kk==1
                            d_val(kk)=d_grid(d_ind(kk));
                        else
                            d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
                        end
                    end
                end
                aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
                for kk=1:l_a
                    if kk==1
                        aprime_val(kk)=a_grid(aprime_ind(kk));
                    else
                        aprime_val(kk)=a_grid(aprime_ind(kk)+sum(n_a(1:kk-1)));
                    end
                end
                Values(j1,j2)=SSvaluesFn{i}(d_val,aprime_val,a_val,s_val,pi_z,p_val);
            end
        end
        
        Values=reshape(Values,[N_a*N_z,1]);
        
        [SortedValues,SortedValues_index] = sort(Values);
        SortedWeights = SteadyStateDistVec(SortedValues_index);
        
        CumSumSortedWeights=cumsum(SortedWeights);
        
        WeightedValues=Values.*SteadyStateDistVec;
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        QuantileIndexes=zeros(1,NumQuantiles-1,'gpuArray');
        QuantileCufoffs=zeros(1,NumQuantiles-1,'gpuArray');
        QuantileMeans=zeros(1,NumQuantiles,'gpuArray');
        for ii=1:NumQuantiles-1
            [~,tempindex]=min(CumSumSortedWeights,ii/NumQuantiles);
            QuantileIndexes(ii)=tempindex;
            QuantileCufoffs(ii)=SortedValues(tempindex);
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
        temp=(CumSumSortedWeights>=Tolerance);
        [~,tempindex]=min(CumSumSortedWeights,ii/NumQuantiles);
        minvalue=SortedValues(tempindex);
        % Max value
        temp=(CumSumSortedWeights>=(1-Tolerance));
        [~,tempindex]=max(CumSumSortedWeights,ii/NumQuantiles);
        maxvalue=SortedValues(tempindex);
        
        SSvalues_QuantileCutOffs(i,:)=[minvalue, QuantileCutoffs, maxvalue]; 
        SSvalues_QuantileMeans(i,:)=QuantileMeans;
    end
    
end

end

