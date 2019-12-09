function CrossSectionCorr=EvalFnOnAgentDist_CrossSectionCorr_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel)
% Evaluates the cross-sectional correlation for each column of SSvaluesFn
% eg. if you give a FnsToEvaluate which is 2x3 (functions) then you will get
% three correlation coefficients (one for the pair of functions that
% constitute each column).

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
    CrossSectionCorr=zeros(length(FnsToEvaluate),1,'gpuArray');
    
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
        
        Values1=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{1,i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values1=reshape(Values1,[N_a*N_z,1]);
        Values2=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{2,i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values2=reshape(Values2,[N_a*N_z,1]);
        
        Mean1=sum(Values1.*StationaryDistVec);
        Mean2=sum(Values2.*StationaryDistVec);
        StdDev1=sqrt(sum(StationaryDistVec.*((Values1-Mean1.*ones(N_a*N_z,1)).^2)));
        StdDev2=sqrt(sum(StationaryDistVec.*((Values2-Mean2.*ones(N_a*N_z,1)).^2)));
        
        Numerator=sum((Values1-Mean1*ones(N_a*N_z,1,'gpuArray')).*(Values2-Mean2*ones(N_a*N_z,1,'gpuArray')).*StationaryDistVec);
        CrossSectionCorr(i)=Numerator/(StdDev1*StdDev2);
    end
    
else
    CrossSectionCorr=zeros(length(FnsToEvaluate),1);
%     d_val=zeros(l_d,1);
%     aprime_val=zeros(l_a,1);
%     a_val=zeros(l_a,1);
%     s_val=zeros(l_z,1);
    
%     [d_gridvals, aprime_gridvals, a_gridvals, z_gridvals]=CreateGridvals(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,z_grid,1,2);
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    for i=1:length(FnsToEvaluate)
%         % Includes check for cases in which no parameters are actually required
%         if isempty(FnsToEvaluateParamNames) % check for 'SSvalueParamNames={}'
%             if l_d==0
%                 Values1=zeros(N_a,N_z);
%                 Values2=zeros(N_a,N_z);
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade_gpu([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_z
%                         s_ind=ind2sub_homemade_gpu([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 s_val(jj2)=z_grid(s_ind(jj2));
%                             else
%                                 s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         [aprime_ind]=PolicyIndexes(:,j1,j2);
%                         for kk2=1:l_a
%                             if kk2==1
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2));
%                             else
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
%                             end
%                         end
%                         Values1(j1,j2)=FnsToEvaluate{1,i}(aprime_val,a_val,s_val);
%                         Values2(j1,j2)=FnsToEvaluate{2,i}(aprime_val,a_val,s_val);
%                     end
%                 end
%             else
%                 Values1=zeros(N_a,N_z);
%                 Values2=zeros(N_a,N_z);
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade_gpu([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_z
%                         s_ind=ind2sub_homemade_gpu([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 s_val(jj2)=z_grid(s_ind(jj2));
%                             else
%                                 s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         d_ind=PolicyIndexes(1:l_d,j1,j2);
%                         aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2);
%                         for kk1=1:l_d
%                             if kk1==1
%                                 d_val(kk1)=d_grid(d_ind(kk1));
%                             else
%                                 d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
%                             end
%                         end
%                         for kk2=1:l_a
%                             if kk2==1
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2));
%                             else
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
%                             end
%                         end
%                         Values1(j1,j2)=FnsToEvaluate{1,i}(d_val,aprime_val,a_val,s_val);
%                         Values2(j1,j2)=FnsToEvaluate{2,i}(d_val,aprime_val,a_val,s_val);
%                     end
%                 end
%             end
%         else
%             FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames);
%             if l_d==0
%                 Values1=zeros(N_a,N_z);
%                 Values2=zeros(N_a,N_z);
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade_gpu([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_z
%                         s_ind=ind2sub_homemade_gpu([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 s_val(jj2)=z_grid(s_ind(jj2));
%                             else
%                                 s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         [aprime_ind]=PolicyIndexes(:,j1,j2);
%                         for kk2=1:l_a
%                             if kk2==1
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2));
%                             else
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
%                             end
%                         end
%                         Values1(j1,j2)=FnsToEvaluate{1,i}(aprime_val,a_val,s_val,FnToEvaluateParamsVec);
%                         Values2(j1,j2)=FnsToEvaluate{2,i}(aprime_val,a_val,s_val,FnToEvaluateParamsVec);
%                     end
%                 end
%             else
%                 Values1=zeros(N_a,N_z);
%                 Values2=zeros(N_a,N_z);
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade_gpu([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_z
%                         s_ind=ind2sub_homemade_gpu([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 s_val(jj2)=z_grid(s_ind(jj2));
%                             else
%                                 s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         d_ind=PolicyIndexes(1:l_d,j1,j2);
%                         aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2);
%                         for kk1=1:l_d
%                             if kk1==1
%                                 d_val(kk1)=d_grid(d_ind(kk1));
%                             else
%                                 d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
%                             end
%                         end
%                         for kk2=1:l_a
%                             if kk2==1
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2));
%                             else
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
%                             end
%                         end
%                         Values1(j1,j2)=FnsToEvaluate{1,i}(d_val,aprime_val,a_val,s_val,FnToEvaluateParamsVec);
%                         Values2(j1,j2)=FnsToEvaluate{2,i}(d_val,aprime_val,a_val,s_val,FnToEvaluateParamsVec);
%                     end
%                 end
%             end
%         end
%         
%     end
%     Values1=reshape(Values1,[N_a*N_z,1]);
%     Values2=reshape(Values2,[N_a*N_z,1]);
    
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names) % check for 'FnsToEvaluateParamNames={}'
            Values1=zeros(N_a*N_z,1);
            Values2=zeros(N_a*N_z,1);
            if l_d==0
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    %                     a_val=a_gridvals{j1,:};
                    %                     z_val=z_gridvals{j2,:};
                    %                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
                    %                     Values(ii)=SSvaluesFn{i}(aprime_val,a_val,z_val);
                    Values1(ii)=FnsToEvaluate{1,i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                    Values2(ii)=FnsToEvaluate{2,i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
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
                    Values1(ii)=FnsToEvaluate{1,i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                    Values2(ii)=FnsToEvaluate{2,i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
            end
        else
            Values1=zeros(N_a*N_z,1);
            Values2=zeros(N_a*N_z,1);
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
                    Values1(ii)=FnsToEvaluate{1,i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                    Values2(ii)=FnsToEvaluate{2,i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
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
                    Values1(ii)=FnsToEvaluate{1,i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                    Values2(ii)=FnsToEvaluate{2,i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
            end
        end
        
        Mean1=sum(Values1.*StationaryDistVec);
        Mean2=sum(Values2.*StationaryDistVec);
        StdDev1=sqrt(sum(StationaryDistVec.*((Values1-Mean1.*ones(N_a*N_z,1)).^2)));
        StdDev2=sqrt(sum(StationaryDistVec.*((Values2-Mean2.*ones(N_a*N_z,1)).^2)));
        
        Numerator=sum((Values1-Mean1*ones(N_a*N_z,1,'gpuArray')).*(Values2-Mean2*ones(N_a*N_z,1,'gpuArray')).*StationaryDistVec);
        CrossSectionCorr(i)=Numerator/(StdDev1*StdDev2);
    end
end

end