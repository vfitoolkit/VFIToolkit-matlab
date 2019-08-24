function AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case2(StationaryDist, PolicyIndexes, FnsToEvaluateFn, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, options, AgeDependentGridParamNames) %pi_z,p_val
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluateFn
% options and AgeDependentGridParamNames is only needed when you are using Age Dependent Grids, otherwise this is not a required input.

if isa(StationaryDist,'struct')
    % Using Age Dependent Grids so send there
    % Note that in this case: d_grid is d_gridfn, a_grid is a_gridfn,
    % z_grid is z_gridfn. Parallel is options. AgeDependentGridParamNames is also needed. 
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case2_AgeDepGrids(StationaryDist, PolicyIndexes, FnsToEvaluateFn, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, options, AgeDependentGridParamNames);
    return
end

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

if isa(StationaryDist,'gpuArray')% Parallel==2
    AggVars=zeros(length(FnsToEvaluateFn),1,'gpuArray');
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);
    
    PolicyValues=PolicyInd2Val_FHorz_Case2(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,2);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]

    for i=1:length(FnsToEvaluateFn)
        Values=nan(N_a*N_z,N_j,'gpuArray');
        for jj=1:N_j
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames) %|| strcmp(SSvalueParamNames(i).Names(1),'')) % check for 'SSvalueParamNames={} or SSvalueParamNames={''}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names,jj);
            end
            Values(:,jj)=reshape(ValuesOnSSGrid_Case2(FnsToEvaluateFn{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,2),[N_a*N_z,1]);
        end
%         Values=reshape(Values,[N_a*N_z,N_j]);
        AggVars(i)=sum(sum(Values.*StationaryDistVec));
    end
    
else
    AggVars=zeros(length(FnsToEvaluateFn),1);
%     d_val=zeros(l_d,1);
%     a_val=zeros(l_a,1);
%     z_val=zeros(l_z,1);
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);
    
    a_gridvals=CreateGridvals(n_a,a_grid,1);
    z_gridvals=CreateGridvals(n_z,z_grid,1);
    dPolicy_gridvals=zeros(N_a*N_z,N_j);
    for jj=1:N_j
        dPolicy_gridvals(:,jj)=CreateGridvals_Policy(PolicyIndexes(:,:,jj),n_d,[],n_a,n_z,d_grid,[],2,1);
    end
    
    for i=1:length(FnsToEvaluateFn)
        Values=zeros(N_a,N_z,N_j);
        for a_c=1:N_a
            a_val=a_gridvals(a_c,:);
%             a_ind=ind2sub_homemade_gpu([n_a],j1);
%             for jj1=1:l_a
%                 if jj1==1
%                     a_val(jj1)=a_grid(a_ind(jj1));
%                 else
%                     a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                 end
%             end
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:);
%                 s_ind=ind2sub_homemade_gpu([n_z],j2);
%                 for jj2=1:l_z
%                     if jj2==1
%                         z_val(jj2)=z_grid(s_ind(jj2));
%                     else
%                         z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                     end
%                 end
%                 d_ind=PolicyIndexes(1:l_d,j1,j2);
%                 for kk1=1:l_d
%                     if kk1==1
%                         d_val(kk1)=d_grid(d_ind(kk1));
%                     else
%                         d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
%                     end
%                 end
                az_c=sub2ind_homemade([N_a,N_z],[a_c,z_c]);
                for jj=1:N_j
                    d_val=dPolicy_gridvals(az_c,jj);
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
                    Values(a_c,z_c,jj)=FnsToEvaluateFn{i}(tempcell{:});
                end
            end
        end
        Values=reshape(Values,[N_a*N_z*N_j,1]);
        
        AggVars(i)=sum(Values.*StationaryDistVec);
    end

end

% % % Includes check for cases in which no parameters are actually required
% % if (isempty(SSvalueParamNames) || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
% %     SSvalueParamsVec=[];
% % else
% %     SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames);
% % end
% % 
% % % Have used this setup as for some functions in the toolkit it is more convenient if they can call the vector version directly.
% % % Not true for FHorz as result of dependence of parameter values on age.
% % SSvalues_AggVars=SSvalues_AggVars_Case2_vec(StationaryDist, PolicyIndexes, SSvaluesFn, SSvalueParamsVec, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p_val, Parallel);
% 
% 
% if nargin<12 %Default is to assume SteadyStateDist exists on the gpu
%     Parallel=2;
% end
% 
% l_d=length(n_d);
% l_a=length(n_a);
% l_z=length(n_z);
% l_j=length(N_j); % This is anyway just 1
% 
% % Check if the SSvaluesFn depends on pi_s, if not then can do it all much
% % faster. (I have been unable to figure out how to really take advantage of GPU
% % when there is pi_s).
% nargin_vec=zeros(numel(SSvaluesFn),1);
% for ii=1:numel(SSvaluesFn)
%     nargin_vec(ii)=nargin(SSvaluesFn{ii});
% end
% if max(nargin_vec)==(l_d+l_a+l_z+length(p_val)+length(SSvalueParamNames)) && Parallel==2 % (l_d+l_a+l_z+1+length(SSvalueParamsVec))
%     SSvalues_AggVars=SSvalues_AggVars_FHorz_Case2_NoPi(StationaryDist, PolicyIndexes, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, p_val, Parallel);
%     return 
% end
% 
% N_a=prod(n_a);
% N_z=prod(n_z);
% 
% if Parallel==2
%     PolicyIndexesKron=gather(reshape(PolicyIndexes,[l_d,N_a,N_z,N_j]));
%     SteadyStateDistVec=gather(reshape(StationaryDist,[N_a*N_z*N_j,1]));
%     d_grid=gather(d_grid);
%     a_grid=gather(a_grid);
%     z_grid=gather(z_grid);
%     pi_z=gather(pi_z);
% else 
%     PolicyIndexesKron=reshape(PolicyIndexes,[l_d,N_a,N_z,N_j]);
%     SteadyStateDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);
% end
% 
% d_val=zeros(l_d,1);
% a_val=zeros(l_a,1);
% z_val=zeros(l_z,1);
% Values=zeros(N_a,N_z,N_j,length(SSvaluesFn));
% % Should be able to speed this up with a parfor
% for j1=1:N_a
%     a_ind=ind2sub_homemade([n_a],j1);
%     for jj1=1:l_a
%         if jj1==1
%             a_val(jj1)=a_grid(a_ind(jj1));
%         else
%             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%         end
%     end
%     for j2=1:N_z
%         s_ind=ind2sub_homemade([n_z],j2);
%         for jj2=1:l_z
%             if jj2==1
%                 z_val(jj2)=z_grid(s_ind(jj2));
%             else
%                 z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%             end
%         end
%         for jj=1:N_j
%             d_ind=PolicyIndexesKron(:,j1,j2,jj);
%             for kk=1:l_d
%                 if kk==1
%                     d_val(kk)=d_grid(d_ind(kk));
%                 else
%                     d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
%                 end
%             end
%             for i=1:length(SSvaluesFn)
%                 Values(j1,j2,jj,i)=SSvaluesFn{i}(d_val(:),a_val(:),z_val(:),pi_z,p_val);
%             end
%         end
%     end
% end
% Values=reshape(Values,[N_a*N_z*N_j,length(SSvaluesFn)]);
% SSvalues_AggVars=sum(Values.*(SteadyStateDistVec*ones(1,length(SSvaluesFn))),1);

end

