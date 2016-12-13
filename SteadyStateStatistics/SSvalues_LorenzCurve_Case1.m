function [SSvalues_LorenzCurve]=SSvalues_LorenzCurve_Case1(StationaryDist, PolicyIndexes, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p_val,Parallel,npoints)
%Returns a Lorenz Curve 100-by-1 that contains all of the quantiles from 1
%to 100. Unless the optional npoints input is used in which case it will be
%npoints-by-1.

%Note that to unnormalize the Lorenz Curve you can just multiply it be the
%SSvalues_AggVars for the same variable. This will give you the inverse
%cdf.

SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames);

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

if nargin<15
    npoints=100;
end

% Check if the SSvaluesFn depends on pi_s, if not then can do it all much
% faster. (I have been unable to figure out how to really take advantage of GPU
% when there is pi_s).
nargin_vec=zeros(numel(SSvaluesFn),1);
for ii=1:numel(SSvaluesFn)
    nargin_vec(ii)=nargin(SSvaluesFn{ii});
end
if max(nargin_vec)==(l_d+2*l_a+l_z+1+length(SSvalueParamsVec)) && Parallel==2
    % Faster as allows use of ValuesOnSSGrid_Case1() rather than loop.
    SSvalues_LorenzCurve=SSvalues_LorenzCurve_Case1_NoPi(StationaryDist, PolicyIndexes, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, p_val, Parallel,npoints);
    return 
end

if Parallel~=2
    SSvalues_AggVars=zeros(length(SSvaluesFn),1);
    SSvalues_LorenzCurve=zeros(length(SSvaluesFn),npoints);
    d_val=zeros(l_d,1);
    aprime_val=zeros(l_a,1);
    a_val=zeros(l_a,1);
    s_val=zeros(l_z,1);
    PolicyIndexesKron=reshape(PolicyIndexes,[l_d+l_a,N_a,N_z]);
    SteadyStateDistVec=reshape(StationaryDist,[N_a*N_z,1]);
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
                
                aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
                for kk=1:l_a
                    if kk==1
                        aprime_val(kk)=a_grid(aprime_ind(kk));
                    else
                        aprime_val(kk)=a_grid(aprime_ind(kk)+sum(n_a(1:kk-1)));
                    end
                end
                if l_d==0
%                     d_val=0;
                    Values(j1,j2)=SSvaluesFn{i}(aprime_val,a_val,s_val,pi_z,p_val,SSvalueParamsVec);
                else
                    d_ind=PolicyIndexesKron(1:l_d,j1,j2);
                    for kk=1:l_d
                        if kk==1
                            d_val(kk)=d_grid(d_ind(kk));
                        else
                            d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
                        end
                    end
                    Values(j1,j2)=SSvaluesFn{i}(d_val,aprime_val,a_val,s_val,pi_z,p_val,SSvalueParamsVec);
                end
            end
        end
        
        Values=reshape(Values,[N_a*N_z,1]);
        
        WeightedValues=Values.*SteadyStateDistVec;
        SSvalues_AggVars(i)=sum(WeightedValues);
        
        
        [~,SortedValues_index] = sort(Values);
        
        SortedSteadyStateDistVec=SteadyStateDistVec(SortedValues_index);
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        CumSumSortedSteadyStateDistVec=cumsum(SortedSteadyStateDistVec);
        
%         %We now want to use interpolation, but this won't work unless all
%         %values in are CumSumSortedSteadyStateDist distinct. So we now remove
%         %any duplicates (ie. points of zero probability mass/density). We then
%         %have to remove the corresponding points of SortedValues
%         [trash2,UniqueIndex] = unique(CumSumSortedSteadyStateDistVec,'first');
        %We now want to use interpolation, but this won't work unless all
        %values in are CumSumSortedSteadyStateDist distinct. So we now remove
        %any duplicates (ie. points of zero probability mass/density). We then
        %have to remove the corresponding points of SortedValues. Since we
        %are just looking for 100 points to make up our cdf I round all
        %variables to 5 decimal points before checking for uniqueness (Do 
        %this because otherwise rounding in the ~12th decimal place was causing 
        % problems with vector not being sorted as strictly increasing.
        [~,UniqueIndex] = unique(floor(CumSumSortedSteadyStateDistVec*10^5),'first');
        CumSumSortedSteadyStateDistVec_NoDuplicates=CumSumSortedSteadyStateDistVec(sort(UniqueIndex));
        SortedWeightedValues_NoDuplicates=SortedWeightedValues(sort(UniqueIndex));
        
        CumSumSortedWeightedValues_NoDuplicates=cumsum(SortedWeightedValues_NoDuplicates);
        
%         % I now also get rid of all of those points after the cdf reaches
%         % 1-10^(-9). This is just because otherwise rounding in the ~12th
%         % decimal place was causing problems with vector not being
%         % 'sorted'.
%         firstIndex = find((CumSumSortedSteadyStateDistVec_NoDuplicates-1+10^(-9))>0,1,'first');
%         CumSumSortedSteadyStateDistVec_NoDuplicates=CumSumSortedSteadyStateDistVec_NoDuplicates(1:firstIndex);
%         CumSumSortedWeightedValues_NoDuplicates=CumSumSortedWeightedValues_NoDuplicates(1:firstIndex);
        
%         InverseCDF_xgrid=0.01:0.01:1;
        InverseCDF_xgrid=1/npoints:1/npoints:1;

        
        InverseCDF_SSvalues=interp1(CumSumSortedSteadyStateDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
        % interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we
        % have already sorted and removed duplicates this will just be the last
        % point so we can just grab it directly.
%         InverseCDF_SSvalues(100)=CumSumSortedWeightedValues_NoDuplicates(end);
        InverseCDF_SSvalues(npoints)=CumSumSortedWeightedValues_NoDuplicates(end);
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
    
else %Parallel==2
    SSvalues_AggVars=zeros(length(SSvaluesFn),1,'gpuArray');
%     SSvalues_LorenzCurve=zeros(length(SSvaluesFn),100,'gpuArray');
    SSvalues_LorenzCurve=zeros(length(SSvaluesFn),npoints,'gpuArray');
    d_val=zeros(l_d,1,'gpuArray');
    aprime_val=zeros(l_a,1,'gpuArray');
    a_val=zeros(l_a,1,'gpuArray');
    s_val=zeros(l_z,1,'gpuArray');
    PolicyIndexesKron=reshape(PolicyIndexes,[l_d+l_a,N_a,N_z]);
    SteadyStateDistVec=reshape(StationaryDist,[N_a*N_z,1]);
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
                aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
                for kk=1:l_a
                    if kk==1
                        aprime_val(kk)=a_grid(aprime_ind(kk));
                    else
                        aprime_val(kk)=a_grid(aprime_ind(kk)+sum(n_a(1:kk-1)));
                    end
                end
                if l_d==0
                    %d_val=0;
                    Values(j1,j2)=SSvaluesFn{i}(aprime_val,a_val,s_val,pi_z,p_val,SSvalueParamsVec);
                else
                    d_ind=PolicyIndexesKron(1:l_d,j1,j2);
                    for kk=1:l_d
                        if kk==1
                            d_val(kk)=d_grid(d_ind(kk));
                        else
                            d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
                        end
                    end
                    Values(j1,j2)=SSvaluesFn{i}(d_val,aprime_val,a_val,s_val,pi_z,p_val,SSvalueParamsVec);
                end
                
            end
        end
        
        Values=reshape(Values,[N_a*N_z,1]);
        
        WeightedValues=Values.*SteadyStateDistVec;
        SSvalues_AggVars(i)=sum(WeightedValues);
        
        
        [~,SortedValues_index] = sort(Values);
        
        SortedSteadyStateDistVec=SteadyStateDistVec(SortedValues_index);
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        CumSumSortedSteadyStateDistVec=cumsum(SortedSteadyStateDistVec);
        
        %We now want to use interpolation, but this won't work unless all
        %values in are CumSumSortedSteadyStateDist distinct. So we now remove
        %any duplicates (ie. points of zero probability mass/density). We then
        %have to remove the corresponding points of SortedValues. Since we
        %are just looking for 100 points to make up our cdf I round all
        %variables to 5 decimal points before checking for uniqueness (Do 
        %this because otherwise rounding in the ~12th decimal place was causing 
        % problems with vector not being sorted as strictly increasing.
        [~,UniqueIndex] = unique(floor(CumSumSortedSteadyStateDistVec*10^5),'first');
        CumSumSortedSteadyStateDistVec_NoDuplicates=CumSumSortedSteadyStateDistVec(sort(UniqueIndex));
        SortedWeightedValues_NoDuplicates=SortedWeightedValues(sort(UniqueIndex));
        
        CumSumSortedWeightedValues_NoDuplicates=cumsum(SortedWeightedValues_NoDuplicates);
        
%         % I now also get rid of all of those points after the cdf reaches
%         % 1-10^(-9). This is just because otherwise rounding in the ~12th
%         % decimal place was causing problems with vector not being
%         % 'sorted'.
%         firstIndex = find((CumSumSortedSteadyStateDistVec_NoDuplicates-1+10^(-9))>0,1,'first');
%         CumSumSortedSteadyStateDistVec_NoDuplicates=CumSumSortedSteadyStateDistVec_NoDuplicates(1:firstIndex);
%         CumSumSortedWeightedValues_NoDuplicates=CumSumSortedWeightedValues_NoDuplicates(1:firstIndex);
        
%         InverseCDF_xgrid=gpuArray(0.01:0.01:1);
        InverseCDF_xgrid=gpuArray(1/npoints:1/npoints:1);
                
        
        InverseCDF_SSvalues=interp1(CumSumSortedSteadyStateDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
        % interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we
        % have already sorted and removed duplicates this will just be the last
        % point so we can just grab it directly.
%         InverseCDF_SSvalues(100)=CumSumSortedWeightedValues_NoDuplicates(end);
        InverseCDF_SSvalues(npoints)=CumSumSortedWeightedValues_NoDuplicates(end);
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
    
end

end

