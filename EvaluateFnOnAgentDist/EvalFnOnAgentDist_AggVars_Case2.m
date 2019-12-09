function AggVars=EvalFnOnAgentDist_AggVars_Case2(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid,Parallel)
% Evaluates the aggregate value (weighted sum/integral) for each element of SSvaluesFn

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

try % if Parallel==2 % First try to use gpu as this will be faster when it works
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
      
    PolicyValues=PolicyInd2Val_Case2(PolicyIndexes,n_d,n_a,n_z,d_grid,Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names)  % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names);
        end
        Values=EvalFnOnAgentDist_Grid_Case2(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        % When evaluating value function (which may sometimes give -Inf
        % values) on StationaryDistVec (which at those points will be
        % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
        temp=Values.*StationaryDistVec;
        AggVars(i)=sum(temp(~isnan(temp)));
    end
catch % else % Use the CPU
    StationaryDistVec=gather(StationaryDistVec);
        
    AggVars=zeros(length(FnsToEvaluate),1);
    
%     [d_gridvals, ~, a_gridvals, z_gridvals]=CreateGridvals(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,z_grid,2,2);
    [d_gridvals, ~]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,2, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
%     d_val=zeros(l_d,1);
    
%     z_gridvals=cell(N_z,l_z);
%     for i1=1:N_z
%         sub=zeros(1,l_z);
%         sub(1)=rem(i1-1,n_z(1))+1;
%         for ii=2:l_z-1
%             sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
%         end
%         sub(l_z)=ceil(i1/prod(n_z(1:l_z-1)));
%         
%         if l_z>1
%             sub=sub+[0,cumsum(n_z(1:end-1))];
%         end
%         z_gridvals(i1,:)=num2cell(z_grid(sub));
%     end
%     a_gridvals=cell(N_a,l_a);
%     for i2=1:N_a
%         sub=zeros(1,l_a);
%         sub(1)=rem(i2-1,n_a(1))+1;
%         for ii=2:l_a-1
%             sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
%         end
%         sub(l_a)=ceil(i2/prod(n_a(1:l_a-1)));
%         
%         if l_a>1
%             sub=sub+[0,cumsum(n_a(1:end-1))];
%         end
%         a_gridvals(i2,:)=num2cell(a_grid(sub));
%     end
%     
%     PolicyIndexes=reshape(PolicyIndexes,[size(PolicyIndexes,1),N_a,N_z]);
%     d_gridvals=cell(N_a*N_z,l_d);
%     for ii=1:N_a*N_z
% %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
%         j1=rem(ii-1,N_a)+1;
%         j2=ceil(ii/N_a);
%         d_ind=PolicyIndexes(:,j1,j2);
%         for kk1=1:l_d
%             if kk1==1
%                 d_val(kk1)=d_grid(d_ind(kk1));
%             else
%                 d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
%             end
%         end
%         d_gridvals(ii,:)=num2cell(d_val);
%     end
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
%                 a_val=a_gridvals{j1,:};
%                 z_val=z_gridvals{j2,:};
%                 d_val=d_gridvals{j1+(j2-1)*N_a,:};
%                 Values(ii)=SSvaluesFn{i}(d_val,a_val,z_val);
                Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
            end
            % When evaluating value function (which may sometimes give -Inf
            % values) on StationaryDistVec (which at those points will be
            % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
            temp=Values.*StationaryDistVec;
            AggVars(i)=sum(temp(~isnan(temp)));
        else
            SSvalueParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
%                 a_val=a_gridvals(j1,:);
%                 z_val=z_gridvals(j2,:);
%                 d_val=d_gridvals(j1+(j2-1)*N_a,:);
%                 Values(ii)=SSvaluesFn{i}(d_val,a_val,z_val,SSvalueParamsVec);
                Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},SSvalueParamsCell{:});
            end
            % When evaluating value function (which may sometimes give -Inf
            % values) on StationaryDistVec (which at those points will be
            % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
            temp=Values.*StationaryDistVec;
            AggVars(i)=sum(temp(~isnan(temp)));
        end
    end
    
end


end

