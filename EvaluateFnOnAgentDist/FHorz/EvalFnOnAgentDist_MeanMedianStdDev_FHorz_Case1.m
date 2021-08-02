function MeanMedianStdDev=EvalFnOnAgentDist_MeanMedianStdDev_FHorz_Case1(StationaryDist,PolicyIndexes, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,Parallel,simoptions)


if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

if Parallel==2
    MeanMedianStdDev=zeros(length(FnsToEvaluate),3,'gpuArray');
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);
    
    PolicyValues=PolicyInd2Val_FHorz_Case1(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]);
    for ii=1:length(FnsToEvaluate)
        Values=nan(N_a*N_z,N_j,'gpuArray');
        for jj=1:N_j
            if fieldexists_ExogShockFn==1
                if fieldexists_ExogShockFnParamNames==1
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for kk=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(kk,1)={ExogShockFnParamsVec(kk)};
                    end
                    [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                else
                    [z_grid,~]=simoptions.ExogShockFn(jj);
                end
            end

            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
            end
%             % Includes check for cases in which no parameters are actually required
%             if isempty(FnsToEvaluateParamNames(ii).Names)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
%                 FnToEvaluateParamsVec=[];
%             else
%                 FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj);
%             end
            Values(:,jj)=reshape(EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{ii}, FnToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,Parallel),[N_a*N_z,1]);
        end
        Values=reshape(Values,[N_a*N_z*N_j,1]);
%         StationaryDistVec=reshape(StationaryDistVec,[N_a*N_z*N_j,1]);
        % Mean
        MeanMedianStdDev(ii,1)=sum(Values.*StationaryDistVec);
        % Median
        [SortedValues,SortedValues_index] = sort(Values);
        SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
        median_index=find(cumsum(SortedStationaryDistVec)>=0.5,1,'first');
        MeanMedianStdDev(ii,2)=SortedValues(median_index);
        % SSvalues_MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedStationaryDistVec)>0.5));
        % Standard Deviation
        MeanMedianStdDev(ii,3)=sqrt(sum(StationaryDistVec.*((Values-MeanMedianStdDev(ii,1).*ones(N_a*N_z*N_j,1)).^2)));
    end
    
else
    MeanMedianStdDev=zeros(length(FnsToEvaluate),3);
    if l_d>0
        d_val=zeros(1,l_d);
    end
    aprime_val=zeros(1,l_a);
    a_val=zeros(1,l_a);
    z_val=zeros(1,l_z);
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);
    
    sizePolicyIndexes=size(PolicyIndexes);
    if sizePolicyIndexes(2:end)~=[N_a,N_z,N_j] % If not in vectorized form
        PolicyIndexes=reshape(PolicyIndexes,[sizePolicyIndexes(1),N_a,N_z,N_j]);
    end
    
    for ii=1:length(FnsToEvaluate)
        Values=zeros(N_a,N_z,N_j);
        if l_d==0
            for jj=1:N_j
                if fieldexists_ExogShockFn==1
                    if fieldexists_ExogShockFnParamNames==1
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for kk=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(kk,1)={ExogShockFnParamsVec(kk)};
                        end
                        [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    else
                        [z_grid,~]=simoptions.ExogShockFn(jj);
                    end
                end
                
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
                                z_val(jj2)=z_grid(s_ind(jj2));
                            else
                                z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                            end
                        end
                        [aprime_ind]=PolicyIndexes(:,j1,j2,jj);
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        % Includes check for cases in which no parameters are actually required
                        if isempty(FnsToEvaluateParamNames(ii).Names)
                            tempv=[aprime_val,a_val,z_val];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        else
                            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj);
                            tempv=[aprime_val,a_val,z_val,FnToEvaluateParamsVec];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        end
                        Values(j1,j2,jj)=FnsToEvaluate{ii}(tempcell{:});
                    end
                end
            end
        else
            for jj=1:N_j
                if fieldexists_ExogShockFn==1
                    if fieldexists_ExogShockFnParamNames==1
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for kk=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(kk,1)={ExogShockFnParamsVec(kk)};
                        end
                        [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    else
                        [z_grid,~]=simoptions.ExogShockFn(jj);
                    end
                end
                
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
                                z_val(jj2)=z_grid(s_ind(jj2));
                            else
                                z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                            end
                        end
                        d_ind=PolicyIndexes(1:l_d,j1,j2,jj);
                        aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2,jj);
                        for kk1=1:l_d
                            if kk1==1
                                d_val(kk1)=d_grid(d_ind(kk1));
                            else
                                d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                            end
                        end
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        % Includes check for cases in which no parameters are actually required
                        if isempty(FnsToEvaluateParamNames(ii).Names)
                            tempv=[d_val,aprime_val,a_val,z_val];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        else
                            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj);
                            tempv=[d_val,aprime_val,a_val,z_val,FnToEvaluateParamsVec];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        end
                        Values(j1,j2,jj)=FnsToEvaluate{ii}(tempcell{:});
                    end
                end
            end
        end        
        Values=reshape(Values,[N_a*N_z*N_j,1]);
%         StationaryDistVec=reshape(StationaryDistVec,[N_a*N_z*N_j,1]);
        % Mean
        MeanMedianStdDev(ii,1)=sum(Values.*StationaryDistVec);
        % Median
        [SortedValues,SortedValues_index] = sort(Values);
        SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
        median_index=find(cumsum(SortedStationaryDistVec)>=0.5,1,'first');
        MeanMedianStdDev(ii,2)=SortedValues(median_index);
        % SSvalues_MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedStationaryDistVec)>0.5));
        % Standard Deviation
        MeanMedianStdDev(ii,3)=sqrt(sum(StationaryDistVec.*((Values-MeanMedianStdDev(ii,1).*ones(N_a*N_z*N_j,1)).^2)));
    end
    
end


end