function MeanMedianStdDev=EvalFnOnAgentDist_MeanMedianStdDev_FHorz_Case2_AgeDepGrids(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_gridfn, a_gridfn, z_gridfn, options, AgeDependentGridParamNames) %pi_z,p_val
% Evaluates the aggregate value (weighted sum/integral) for each element of SSvaluesFn

% Note: n_d,n_a,n_z are used once here at beginning and then overwritten with their age-conditional equivalents for all further usage.
daz_gridstructure=AgeDependentGrids_Create_daz_gridstructure(n_d,n_a,n_z,N_j,d_gridfn, a_gridfn, z_gridfn, AgeDependentGridParamNames, Parameters, options);
% Creates daz_gridstructure which contains both the grids themselves and a
% bunch of info about the grids in an easy to access way.
% e.g., the d_grid for age j=10: daz_gridstructure.d_grid.j010
% e.g., the value of N_a for age j=5: daz_gridstructure.N_a.j005
% e.g., the zprime_grid for age j=20: daz_gridstructure.zprime_grid.j020

% % THERE IS ALMOST NO CHANCE THERE WILL BE ENOUGH GPU MEMORY TO DO THIS ON THE GPU
% if isa(StationaryDist.j001, 'gpuArray')%Parallel==2
%     for jj=1:N_j
%         % Make a three digit number out of jj
%         if jj<10
%             jstr=['j00',num2str(jj)];
%         elseif jj>=10 && jj<100
%             jstr=['j0',num2str(jj)];
%         else
%             jstr=['j',num2str(jj)];
%         end
%     StationaryDist.(jstr)=gather(StationaryDist.(jstr));
%     end
% end

% % % For memory reasons doing this directly will typically be impossible (the
% % % amount of memory required to store the full solution will just be too
% % % much. I have left the code that implements this in this command, but
% % % currently it will never actually be called. At some time in the future I
% % % may add it as an option.
% % % ACTUALLY, decided not to bother for now.
% % if isa(StationaryDist.j001, 'gpuArray')%Parallel==2
% %     % Idea is that I can calculate the necessary numbers for each age
% group and then combine them. Doing this for means is trivial (weighted
% mean), for standard deviation it is also easy enough (composite standard
% deviation,
% http://www.burtonsys.com/climate/composite_standard_deviations.html ),
% but as far as I can tell there is no way to do this for the median. One
% option for the median might be just to keep the middle 20% of
% observations and keep updating these as I do each age, but this will only
% work if the median of the whole is not to far from the median of the
% first group. Another option would be a 'two stage' version, namely first
% take the median of the group medians, then on the assumption this will be
% roughly the true median go through again and keep the 20% of observations
% closest to the median of the group medians.
% %     SSvalues_MeanMedianStdDev=zeros(length(SSvaluesFn),3,'gpuArray');    
% %     
% %     FullStationaryDistVec=zeros(prod(prod(n_a))*prod(prod(n_z)),1); % This will be across all the ages j (note that n_a and n_z at this point are still for all ages, not age specific)
% % %     FullValuesVec=zeros(prod(prod(n_a))*prod(prod(n_z)),length(SSvaluesFn)); % Actually is a matrix, with a column vector for each SSvaluesFn
% %     FullValuesVec=struct();
% % 
% %     counter_Naz=0;
% %     for jj=1:N_j
% %         % Make a three digit number out of jj
% %         if jj<10
% %             jstr=['j00',num2str(jj)];
% %         elseif jj>=10 && jj<100
% %             jstr=['j0',num2str(jj)];
% %         else
% %             jstr=['j',num2str(jj)];
% %         end
% %         n_d=daz_gridstructure.n_d.(jstr);
% %         n_a=daz_gridstructure.n_a.(jstr);
% %         n_z=daz_gridstructure.n_z.(jstr);
% %         N_a=daz_gridstructure.N_a.(jstr);
% %         N_z=daz_gridstructure.N_z.(jstr);
% %         d_grid=daz_gridstructure.d_grid.(jstr);
% %         a_grid=daz_gridstructure.a_grid.(jstr);
% %         z_grid=daz_gridstructure.z_grid.(jstr);
% %         
% %         l_a=length(n_a);
% %         l_z=length(n_z);
% %         
% %         StationaryDistVec=reshape(StationaryDist.(jstr),[N_a*N_z,1]);
% %         
% %         FullStationaryDistVec((counter_Naz+1):(counter_Naz+N_a*N_z))=StationaryDistVec;
% %         
% %         PolicyValues=PolicyInd2Val_Case2(PolicyIndexes.(jstr),n_d,n_a,n_z,d_grid,2);
% %         permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
% %         PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
% %         
% %         for i=1:length(SSvaluesFn)
% % %             Values=zeros(N_a*N_z,'gpuArray');
% %             % Includes check for cases in which no parameters are actually required
% %             if isempty(SSvalueParamNames) %|| strcmp(SSvalueParamNames(i).Names(1),'')) % check for 'SSvalueParamNames={} or SSvalueParamNames={''}'
% %                 SSvalueParamsVec=[];
% %             else
% %                 SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames(i).Names,jj);
% %             end
% %             Values=reshape(ValuesOnSSGrid_Case2(SSvaluesFn{i}, SSvalueParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,2),[N_a*N_z,1]);
% %             FullValuesVec.(num2str(i))(counter_Naz+1:counter_Naz+N_a*N_z)=Values;
% %             
% %         end
% %         counter_Naz=counter_Naz+N_a*N_z;
% %     end
% % 
% %     for i=1:length(SSvaluesFn)
% %         Values=FullValuesVec.(num2str(i));%(:,i);
% %         % FullValuesVec and FullStationaryVec can now be used as usual
% %         % Mean
% %         SSvalues_MeanMedianStdDev(i,1)=sum(Values.*FullStationaryDistVec);
% %         % Median
% %         [SortedValues,SortedValues_index] = sort(Values);
% %         SortedStationaryDistVec=FullStationaryDistVec(SortedValues_index);
% %         median_index=find(cumsum(SortedStationaryDistVec)>=0.5,1,'first');
% %         SSvalues_MeanMedianStdDev(i,2)=SortedValues(median_index);
% %         % SSvalues_MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedStationaryDistVec)>0.5));
% %         % Standard Deviation
% %         SSvalues_MeanMedianStdDev(i,3)=sqrt(sum(FullStationaryDistVec.*((Values-SSvalues_MeanMedianStdDev(i,1).*ones(length(FullStationaryDistVec),1)).^2)));
% %     end

% The following implements things directly on GPU and using full state
% space, it will be faster than the alternatives but it is so memory
% intensive that it is unlikely to be of much use in practice.
    if isa(StationaryDist.j001, 'gpuArray')%Parallel==2
    MeanMedianStdDev=zeros(length(FnsToEvaluate),3,'gpuArray');    
    
    FullStationaryDistVec=zeros(sum(prod(n_a'))*sum(prod(n_z')),1,'gpuArray'); % This will be across all the ages j (note that n_a and n_z at this point are still for all ages, not age specific)
    FullValuesVec=zeros(sum(prod(n_a'))*sum(prod(n_z')),length(FnsToEvaluate),'gpuArray'); % Actually is a matrix, with a column vector for each SSvaluesFn

    counter_Naz=0;
    for jj=1:N_j
        % Make a three digit number out of jj
        if jj<10
            jstr=['j00',num2str(jj)];
        elseif jj>=10 && jj<100
            jstr=['j0',num2str(jj)];
        else
            jstr=['j',num2str(jj)];
        end
        n_d=daz_gridstructure.n_d.(jstr);
        n_a=daz_gridstructure.n_a.(jstr);
        n_z=daz_gridstructure.n_z.(jstr);
        N_a=daz_gridstructure.N_a.(jstr);
        N_z=daz_gridstructure.N_z.(jstr);
        d_grid=daz_gridstructure.d_grid.(jstr);
        a_grid=daz_gridstructure.a_grid.(jstr);
        z_grid=daz_gridstructure.z_grid.(jstr);
        
        l_a=length(n_a);
        l_z=length(n_z);
        
        StationaryDistVec=reshape(StationaryDist.(jstr),[N_a*N_z,1]);
        
        FullStationaryDistVec((counter_Naz+1):(counter_Naz+N_a*N_z))=StationaryDistVec;
        
        PolicyValues=PolicyInd2Val_Case2(PolicyIndexes.(jstr),n_d,n_a,n_z,d_grid,2);
        permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
        PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
        
        for i=1:length(FnsToEvaluate)
%             Values=zeros(N_a*N_z,'gpuArray');
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames) %|| strcmp(SSvalueParamNames(i).Names(1),'')) % check for 'SSvalueParamNames={} or SSvalueParamNames={''}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names,jj);
            end
            Values=reshape(ValuesOnSSGrid_Case2(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,2),[N_a*N_z,1]);
            FullValuesVec(counter_Naz+1:counter_Naz+N_a*N_z,i)=Values;
            
        end
        counter_Naz=counter_Naz+N_a*N_z;
    end

    for i=1:length(FnsToEvaluate)
        Values=FullValuesVec(:,i);
        % FullValuesVec and FullStationaryVec can now be used as usual
        % Mean
        MeanMedianStdDev(i,1)=sum(Values.*FullStationaryDistVec);
        % Median
        [SortedValues,SortedValues_index] = sort(Values);
        SortedStationaryDistVec=FullStationaryDistVec(SortedValues_index);
        median_index=find(cumsum(SortedStationaryDistVec)>=0.5,1,'first');
        MeanMedianStdDev(i,2)=SortedValues(median_index);
        % SSvalues_MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedStationaryDistVec)>0.5));
        % Standard Deviation
        MeanMedianStdDev(i,3)=sqrt(sum(FullStationaryDistVec.*((Values-MeanMedianStdDev(i,1).*ones(length(FullStationaryDistVec),1)).^2)));
    end
else
    MeanMedianStdDev=zeros(length(FnsToEvaluate),3);
    % Seems likely that outer loop over age jj and inner loop over
    % SSvaluesFn i is the faster option. But I have not actually checked
    % this.
    FullStationaryDistVec=zeros(prod(prod(n_a))*prod(prod(n_z)),1); % This will be across all the ages j (note that n_a and n_z at this point are still for all ages, not age specific)
%     FullValuesVec=zeros(prod(prod(n_a))*prod(prod(n_z)),length(SSvaluesFn)); % Actually is a matrix, with a column vector for each SSvaluesFn
    FullValuesVec=struct();

    counter_Naz=0;
    for jj=1:N_j
        % Make a three digit number out of jj
        if jj<10
            jstr=['j00',num2str(jj)];
        elseif jj>=10 && jj<100
            jstr=['j0',num2str(jj)];
        else
            jstr=['j',num2str(jj)];
        end
        n_d=daz_gridstructure.n_d.(jstr);
        n_a=daz_gridstructure.n_a.(jstr);
        n_z=daz_gridstructure.n_z.(jstr);
        N_a=daz_gridstructure.N_a.(jstr);
        N_z=daz_gridstructure.N_z.(jstr);
        d_grid=gather(daz_gridstructure.d_grid.(jstr));
        a_grid=gather(daz_gridstructure.a_grid.(jstr));
        z_grid=gather(daz_gridstructure.z_grid.(jstr));
        
        l_d=length(n_d);
        l_a=length(n_a);
        l_z=length(n_z);
        d_val=zeros(l_d,1);
        a_val=zeros(l_a,1);
        z_val=zeros(l_z,1);

        StationaryDistVec=reshape(StationaryDist.(jstr),[N_a*N_z,1]);
        FullStationaryDistVec(counter_Naz+1:counter_Naz+N_a*N_z)=StationaryDistVec;

        for i=1:length(FnsToEvaluate)
            Values=zeros(N_a,N_z);
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
                    d_ind=PolicyIndexes(1:l_d,j1,j2);
                    for kk1=1:l_d
                        if kk1==1
                            d_val(kk1)=d_grid(d_ind(kk1));
                        else
                            d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                        end
                    end
                    
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(i).Names)
                        tempv=[d_val,a_val,z_val];
                        tempcell=cell(1,length(tempv));
                        for temp_c=1:length(tempv)
                            tempcell{temp_c}=tempv(temp_c);
                        end
                    else
                        FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names,jj);
                        tempv=[d_val,a_val,z_val,FnToEvaluateParamsVec];
                        tempcell=cell(1,length(tempv));
                        for temp_c=1:length(tempv)
                            tempcell{temp_c}=tempv(temp_c);
                        end
                    end
                    Values(j1,j2)=FnsToEvaluate{i}(tempcell{:});
                end
            end
            FullValuesVec.(num2str(i))(counter_Naz+1:counter_Naz+N_a*N_z,i)=reshape(Values,[N_a*N_z,1]);
        end
        counter_Naz=counter_Naz+N_a*N_z;
    end

    for i=1:length(FnsToEvaluate)
        Values=FullValuesVec.(num2str(i));%(:,i);
        % FullValuesVec and FullStationaryVec can now be used as usual
        % Mean
        MeanMedianStdDev(i,1)=sum(Values.*FullStationaryDistVec);
        % Median
        [SortedValues,SortedValues_index] = sort(Values);
        SortedStationaryDistVec=FullStationaryDistVec(SortedValues_index);
        median_index=find(cumsum(SortedStationaryDistVec)>=0.5,1,'first');
        MeanMedianStdDev(i,2)=SortedValues(median_index);
        % SSvalues_MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedStationaryDistVec)>0.5));
        % Standard Deviation
        MeanMedianStdDev(i,3)=sqrt(sum(FullStationaryDistVec.*((Values-MeanMedianStdDev(i,1).*ones(length(FullStationaryDistVec),1)).^2)));
    end
end

end

