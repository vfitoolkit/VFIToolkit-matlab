function SSvalues_LorenzCurve=SSvalues_LorenzCurve_Case2(SteadyStateDist, PolicyIndexes, SSvaluesFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p_val,Parallel)
%Returns a Lorenz Curve 100-by-1 that contains all of the quantiles from 1 to 100

%Note that to unnormalize the Lorenz Curve you can just multiply it be the
%SSvalues_AggVars for the same variable. This will give you the inverse
%cdf.

if nargin<12 %Default is to assume SteadyStateDist exists on the gpu
    Parallel=2;
end

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

if Parallel==2
    PolicyIndexesKron=gather(reshape(PolicyIndexes,[l_d,N_a,N_z]));
    SteadyStateDistVec=gather(reshape(SteadyStateDist,[N_a*N_z,1]));
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
    pi_z=gather(pi_z);
else
    PolicyIndexesKron=reshape(PolicyIndexes,[l_d,N_a,N_z]);
    SteadyStateDistVec=reshape(SteadyStateDist,[N_a*N_z,1]);
end

SSvalues_AggVars=zeros(length(SSvaluesFn),1);
SSvalues_LorenzCurve=zeros(length(SSvaluesFn),100);
d_val=zeros(l_d,1);
a_val=zeros(l_a,1);
s_val=zeros(l_z,1);


ValuesFull=zeros(N_a,N_z,length(SSvaluesFn));
% Should be able to improve this using parfor
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
        d_ind=PolicyIndexesKron(:,j1,j2);
        for kk=1:l_d
            if kk==1
                d_val(kk)=d_grid(d_ind(kk));
            else
                d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
            end
        end
        for i=1:length(SSvaluesFn)
            ValuesFull(j1,j2,i)=SSvaluesFn{i}(d_val,a_val,s_val,pi_z,p_val);
        end
    end
end

for i=1:length(SSvaluesFn)

    Values=reshape(ValuesFull(:,:,i),[N_a*N_z,1]);
    
    WeightedValues=Values.*SteadyStateDistVec;
    SSvalues_AggVars(i)=sum(WeightedValues);
    
    
    [~,SortedValues_index] = sort(Values);

    SortedSteadyStateDistVec=SteadyStateDistVec(SortedValues_index);
    SortedWeightedValues=WeightedValues(SortedValues_index);
    
    CumSumSortedSteadyStateDistVec=cumsum(SortedSteadyStateDistVec);
    
    %We now want to use interpolation, but this won't work unless all
    %values in are CumSumSortedSteadyStateDist distinct. So we now remove
    %any duplicates (ie. points of zero probability mass/density). We then
    %have to remove the corresponding points of SortedValues
    [~,UniqueIndex] = unique(CumSumSortedSteadyStateDistVec,'first');
    CumSumSortedSteadyStateDistVec_NoDuplicates=CumSumSortedSteadyStateDistVec(sort(UniqueIndex));
    SortedWeightedValues_NoDuplicates=SortedWeightedValues(sort(UniqueIndex));
    
    CumSumSortedWeightedValues_NoDuplicates=cumsum(SortedWeightedValues_NoDuplicates);

    InverseCDF_xgrid=0.01:0.01:1;
    
    InverseCDF_SSvalues=interp1(CumSumSortedSteadyStateDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
    % interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we
    % have already sorted and removed duplicates this will just be the last
    % point so we can just grab it directly.
    InverseCDF_SSvalues(100)=CumSumSortedWeightedValues_NoDuplicates(length(CumSumSortedWeightedValues_NoDuplicates));
    % interp1 may have similar problems at the bottom of the cdf
    j=1; %use j to figure how many points with this problem
    while InverseCDF_xgrid(j)<CumSumSortedSteadyStateDistVec_NoDuplicates(1)
        j=j+1;
    end
    for jj=1:j-1 %divide evenly through these states (they are all identical)
        InverseCDF_SSvalues(jj)=(jj/j)*InverseCDF_SSvalues(j);
    end

    SSvalues_LorenzCurve(i,:)=InverseCDF_SSvalues./SSvalues_AggVars(i);
end


% SSvalues_AggVars=zeros(length(SSvaluesFn),1);
% SSvalues_LorenzCurve=zeros(length(SSvaluesFn),100);
% d_val=zeros(l_d,1);
% a_val=zeros(l_a,1);
% s_val=zeros(l_s,1);
% PolicyIndexesKron=reshape(PolicyIndexes,[l_d,N_a,N_s]);
% SteadyStateDistVec=reshape(SteadyStateDist,[N_a*N_s,1]);
% for i=1:length(SSvaluesFn)
%     Values=zeros(N_a,N_s);
%     for j1=1:N_a
%         a_ind=ind2sub_homemade([n_a],j1);
%         for jj1=1:l_a
%             if jj1==1
%                 a_val(jj1)=a_grid(a_ind(jj1));
%             else
%                 a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%             end
%         end
%         for j2=1:N_s
%             s_ind=ind2sub_homemade([n_s],j2);
%             for jj2=1:l_s
%                 if jj2==1
%                     s_val(jj2)=s_grid(s_ind(jj2));
%                 else
%                     s_val(jj2)=s_grid(s_ind(jj2)+sum(n_s(1:jj2-1)));
%                 end
%             end
%             d_ind=PolicyIndexesKron(:,j1,j2);
%             for kk=1:l_d
%                 if kk==1
%                     d_val(kk)=d_grid(d_ind(kk));
%                 else
%                     d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
%                 end
%             end
%             Values(j1,j2)=SSvaluesFn{i}(d_val,a_val,s_val,pi_s,p_val);
%         end
%     end
%     
%     Values=reshape(Values,[N_a*N_s,1]);
%     
%     WeightedValues=Values.*SteadyStateDistVec;
%     SSvalues_AggVars(i)=sum(WeightedValues);
%     
%     
%     [trash1,SortedValues_index] = sort(Values);
% 
%     SortedSteadyStateDistVec=SteadyStateDistVec(SortedValues_index);
%     SortedWeightedValues=WeightedValues(SortedValues_index);
%     
%     CumSumSortedSteadyStateDistVec=cumsum(SortedSteadyStateDistVec);
%     
%     %We now want to use interpolation, but this won't work unless all
%     %values in are CumSumSortedSteadyStateDist distinct. So we now remove
%     %any duplicates (ie. points of zero probability mass/density). We then
%     %have to remove the corresponding points of SortedValues
%     [trash2,UniqueIndex] = unique(CumSumSortedSteadyStateDistVec,'first');
%     CumSumSortedSteadyStateDistVec_NoDuplicates=CumSumSortedSteadyStateDistVec(sort(UniqueIndex));
%     SortedWeightedValues_NoDuplicates=SortedWeightedValues(sort(UniqueIndex));
%     
%     CumSumSortedWeightedValues_NoDuplicates=cumsum(SortedWeightedValues_NoDuplicates);
% 
%     InverseCDF_xgrid=0.01:0.01:1;
%     
%     InverseCDF_SSvalues=interp1(CumSumSortedSteadyStateDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
%     % interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we
%     % have already sorted and removed duplicates this will just be the last
%     % point so we can just grab it directly.
%     InverseCDF_SSvalues(100)=CumSumSortedWeightedValues_NoDuplicates(length(CumSumSortedWeightedValues_NoDuplicates));
%     % interp1 may have similar problems at the bottom of the cdf
%     j=1; %use j to figure how many points with this problem
%     while InverseCDF_xgrid(j)<CumSumSortedSteadyStateDistVec_NoDuplicates(1)
%         j=j+1;
%     end
%     for jj=1:j-1 %divide evenly through these states (they are all identical)
%         InverseCDF_SSvalues(jj)=(jj/j)*InverseCDF_SSvalues(j);
%     end
% 
%     
%     SSvalues_LorenzCurve(i,:)=InverseCDF_SSvalues./SSvalues_AggVars(i);
% end

end

