function SSvalues_AggVars=SSvalues_AggVars_Case1(StationaryDist, PolicyIndexes, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel)
% Evaluates the aggregate value (weighted sum/integral) for each element of SSvaluesFn

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

% % % Check if the SSvaluesFn depends on pi_s, if not then can do it all much
% % % faster. (I have been unable to figure out how to really take advantage of GPU
% % % when there is pi_s).
% % nargin_vec=zeros(numel(SSvaluesFn),1);
% % for ii=1:numel(SSvaluesFn)
% %     nargin_vec(ii)=nargin(SSvaluesFn{ii});
% % end
% % if max(nargin_vec)==(l_d+2*l_a+l_z+1+length(SSvalueParams)) && Parallel==2
% %     SSvalues_AggVars=SSvalues_AggVars_Case1_NoPi(SteadyStateDist, PolicyIndexes, SSvaluesFn, SSvalueParams, n_d, n_a, n_z, d_grid, a_grid, z_grid, p_val, Parallel);
% %     return 
% % end

try % if Parallel==2
    SSvalues_AggVars=zeros(length(SSvaluesFn),1,'gpuArray');
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]

    for i=1:length(SSvaluesFn)
        % Includes check for cases in which no parameters are actually required
        if isempty(SSvalueParamNames(i).Names)  % check for 'SSvalueParamNames={}'
            SSvalueParamsCell=[];
        else
            SSvalueParamsCell=gpuArray(CreateVectorFromParams(Parameters,SSvalueParamNames(i).Names));
        end
        Values=ValuesOnSSGrid_Case1(SSvaluesFn{i}, SSvalueParamsCell,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        % When evaluating value function (which may sometimes give -Inf
        % values) on StationaryDistVec (which at those points will be
        % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
        temp=Values.*StationaryDistVec;
        SSvalues_AggVars(i)=sum(temp(~isnan(temp)));
    end
catch % else
    StationaryDistVec=gather(StationaryDistVec);
    
    SSvalues_AggVars=zeros(length(SSvaluesFn),1);
    if l_d>0
        d_val=zeros(l_d,1);
    end
    aprime_val=zeros(l_a,1);
    
    z_gridvals=cell(N_z,length(n_z));
    for i1=1:N_z
        sub=zeros(1,length(n_z));
        sub(1)=rem(i1-1,n_z(1))+1;
        for ii=2:length(n_z)-1
            sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
        end
        sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
        
        if length(n_z)>1
            sub=sub+[0,cumsum(n_z(1:end-1))];
        end
        z_gridvals(i1,:)=num2cell(z_grid(sub));
    end
    a_gridvals=cell(N_a,length(n_a));
    for i2=1:N_a
        sub=zeros(1,length(n_a));
        sub(1)=rem(i2-1,n_a(1)+1);
        for ii=2:length(n_a)-1
            sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
        end
        sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
        
        if length(n_a)>1
            sub=sub+[0,cumsum(n_a(1:end-1))];
        end
        a_gridvals(i2,:)=num2cell(a_grid(sub));
    end  
    
    if l_d>0
        d_gridvals=cell(N_a*N_z,l_d);
        aprime_gridvals=cell(N_a*N_z,l_a);
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            daprime_ind=PolicyIndexes(:,j1,j2);
            d_ind=daprime_ind(1:l_d);
            aprime_ind=daprime_ind((l_d+1):(l_d+l_a));
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
            d_gridvals(ii,:)=num2cell(d_val);
            aprime_gridvals(ii,:)=num2cell(aprime_val);
        end
        
        for i=1:length(SSvaluesFn)
            % Includes check for cases in which no parameters are actually required
            if isempty(SSvalueParamNames(i).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     d_val=d_gridvals{j1+(j2-1)*N_a,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val);
                    Values(ii)=SSvaluesFn{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                SSvalues_AggVars(i)=sum(temp(~isnan(temp)));
            else
                SSvalueParamsCell=num2cell(CreateVectorFromParams(Parameters,SSvalueParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     d_val=d_gridvals{j1+(j2-1)*N_a,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val,SSvalueParamsVec);
                    Values(ii)=SSvaluesFn{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},SSvalueParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                SSvalues_AggVars(i)=sum(temp(~isnan(temp)));
            end
        end
    
    else %l_d=0
        aprime_gridvals=cell(N_a*N_z,l_a);
        for ii=1:N_a*N_z
            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
            j1=rem(ii-1,N_a)+1;
            j2=ceil(ii/N_a);
            aprime_ind=PolicyIndexes(:,j1,j2);
            for kk2=1:l_a
                if kk2==1
                    aprime_val(kk2)=a_grid(aprime_ind(kk2));
                else
                    aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                end
            end
            aprime_gridvals(ii,:)=num2cell(aprime_val);
        end
        
        for i=1:length(SSvaluesFn)
            % Includes check for cases in which no parameters are actually required
            if isempty(SSvalueParamNames(i).Names) % check for 'SSvalueParamNames={}'
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(aprime_val,a_val,z_val);
                    Values(ii)=SSvaluesFn{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                SSvalues_AggVars(i)=sum(temp(~isnan(temp)));
            else
                SSvalueParamsCell=num2cell(CreateVectorFromParams(Parameters,SSvalueParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(aprime_val,a_val,z_val,SSvalueParamsVec);
                    Values(ii)=SSvaluesFn{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},SSvalueParamsCell{:});
                end
                % When evaluating value function (which may sometimes give -Inf
                % values) on StationaryDistVec (which at those points will be
                % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
                temp=Values.*StationaryDistVec;
                SSvalues_AggVars(i)=sum(temp(~isnan(temp)));
            end
        end
    end
    
end



% % % if Parallel==2 % This Parallel==2 case is only used if there is pi_s
% % %     SSvalues_AggVars=zeros(length(SSvaluesFn),1,'gpuArray');
% % %     d_val=zeros(l_d,1,'gpuArray');
% % %     aprime_val=zeros(l_a,1,'gpuArray');
% % %     a_val=zeros(l_a,1,'gpuArray');
% % %     z_val=zeros(l_z,1,'gpuArray');
% % %     
% % %     SteadyStateDistVec=reshape(SteadyStateDist,[N_a*N_z,1]);
% % % 
% % %     for i=1:length(SSvaluesFn)
% % %     
% % % % %      % HOW CAN I DEAL WITH pi_s when using arrayfun????
% % %                 
% % %         Values=zeros(N_a,N_z,'gpuArray');
% % %         for j1=1:N_a
% % %             a_ind=ind2sub_homemade_gpu([n_a],j1);
% % %             for jj1=1:l_a
% % %                 if jj1==1
% % %                     a_val(jj1)=a_grid(a_ind(jj1));
% % %                 else
% % %                     a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
% % %                 end
% % %             end
% % %             for j2=1:N_z
% % %                 s_ind=ind2sub_homemade_gpu([n_z],j2);
% % %                 for jj2=1:l_z
% % %                     if jj2==1
% % %                         z_val(jj2)=z_grid(s_ind(jj2));
% % %                     else
% % %                         z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
% % %                     end
% % %                 end
% % %                 if l_d==0
% % %                     [aprime_ind]=PolicyIndexes(:,j1,j2);
% % %                 else
% % %                     d_ind=PolicyIndexes(1:l_d,j1,j2);
% % %                     aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2);
% % %                     for kk1=1:l_d
% % %                         if kk1==1
% % %                             d_val(kk1)=d_grid(d_ind(kk1));
% % %                         else
% % %                             d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
% % %                         end
% % %                     end
% % %                 end
% % %                 for kk2=1:l_a
% % %                     if kk2==1
% % %                         aprime_val(kk2)=a_grid(aprime_ind(kk2));
% % %                     else
% % %                         aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
% % %                     end
% % %                 end
% % %                 Values(j1,j2)=SSvaluesFn{i}(d_val(:),aprime_val(:),a_val(:),z_val(:),pi_z,p_val);
% % %             end
% % %         end
% % %         Values=reshape(Values,[N_a*N_z,1]);
% % %         
% % %         SSvalues_AggVars(i)=sum(Values.*SteadyStateDistVec);
% % %     end
% % %     
% % % else
% % %     SSvalues_AggVars=zeros(length(SSvaluesFn),1);
% % %     d_val=zeros(l_d,1);
% % %     aprime_val=zeros(l_a,1);
% % %     a_val=zeros(l_a,1);
% % %     z_val=zeros(l_z,1);
% % %     SteadyStateDistVec=reshape(SteadyStateDist,[N_a*N_z,1]);
% % %     
% % %     z_gridvals=zeros(N_z,length(n_z));
% % %     for i1=1:N_z
% % %         sub=zeros(1,length(n_z));
% % %         sub(1)=rem(i1-1,n_z(1))+1;
% % %         for ii=2:length(n_z)-1
% % %             sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
% % %         end
% % %         sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
% % %         
% % %         if length(n_z)>1
% % %             sub=sub+[0,cumsum(n_z(1:end-1))];
% % %         end
% % %         z_gridvals(i1,:)=z_grid(sub);
% % %     end
% % %     a_gridvals=zeros(N_a,length(n_a));
% % %     for i2=1:N_a
% % %         sub=zeros(1,length(n_a));
% % %         sub(1)=rem(i2-1,n_a(1)+1);
% % %         for ii=2:length(n_a)-1
% % %             sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
% % %         end
% % %         sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
% % %         
% % %         if length(n_a)>1
% % %             sub=sub+[0,cumsum(n_a(1:end-1))];
% % %         end
% % %         a_gridvals(i2,:)=a_grid(sub);
% % %     end  
% % %     
% % %     for i=1:length(SSvaluesFn)
% % %         Values=zeros(N_a,N_z);
% % %         for j1=1:N_a
% % %             a_val=a_gridvals(j1);
% % %             for j2=1:N_z
% % %                 s_ind=ind2sub_homemade_gpu([n_z],j2);
% % %                 for jj2=1:l_z
% % %                     if jj2==1
% % %                         z_val(jj2)=z_grid(s_ind(jj2));
% % %                     else
% % %                         z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
% % %                     end
% % %                 end
% % %                 if l_d==0
% % %                     [aprime_ind]=PolicyIndexes(:,j1,j2);
% % %                 else
% % %                     d_ind=PolicyIndexes(1:l_d,j1,j2);
% % %                     aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2);
% % %                     for kk1=1:l_d
% % %                         if kk1==1
% % %                             d_val(kk1)=d_grid(d_ind(kk1));
% % %                         else
% % %                             d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
% % %                         end
% % %                     end
% % %                 end
% % %                 for kk2=1:l_a
% % %                     if kk2==1
% % %                         aprime_val(kk2)=a_grid(aprime_ind(kk2));
% % %                     else
% % %                         aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
% % %                     end
% % %                 end
% % %                 Values(j1,j2)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val,pi_z,p_val);
% % %             end
% % %         end
% % %         Values=reshape(Values,[N_a*N_z,1]);
% % %         
% % %         SSvalues_AggVars(i)=sum(Values.*SteadyStateDistVec);
% % %     end
% % % 
% % % end
    

% else
%     SSvalues_AggVars=zeros(length(SSvaluesFn),1);
%     d_val=zeros(l_d,1);
%     aprime_val=zeros(l_a,1);
%     a_val=zeros(l_a,1);
%     z_val=zeros(l_z,1);
%     SteadyStateDistVec=reshape(SteadyStateDist,[N_a*N_s,1]);
%     
%     for i=1:length(SSvaluesFn)
%         Values=zeros(N_a,N_s);
%         for j1=1:N_a
%             a_ind=ind2sub_homemade_gpu([n_a],j1);
%             for jj1=1:l_a
%                 if jj1==1
%                     a_val(jj1)=a_grid(a_ind(jj1));
%                 else
%                     a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                 end
%             end
%             for j2=1:N_s
%                 s_ind=ind2sub_homemade_gpu([n_z],j2);
%                 for jj2=1:l_z
%                     if jj2==1
%                         z_val(jj2)=z_grid(s_ind(jj2));
%                     else
%                         z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                     end
%                 end
%                 if l_d==0
%                     [aprime_ind]=PolicyIndexes(:,j1,j2);
%                 else
%                     d_ind=PolicyIndexes(1:l_d,j1,j2);
%                     aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2);
%                     for kk1=1:l_d
%                         if kk1==1
%                             d_val(kk1)=d_grid(d_ind(kk1));
%                         else
%                             d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
%                         end
%                     end
%                 end
%                 for kk2=1:l_a
%                     if kk2==1
%                         aprime_val(kk2)=a_grid(aprime_ind(kk2));
%                     else
%                         aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
%                     end
%                 end
%                 Values(j1,j2)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val,pi_z,p_val);
%             end
%         end
%         Values=reshape(Values,[N_a*N_s,1]);
%         
%         SSvalues_AggVars(i)=sum(Values.*SteadyStateDistVec);
%     end
% 
% end


end
