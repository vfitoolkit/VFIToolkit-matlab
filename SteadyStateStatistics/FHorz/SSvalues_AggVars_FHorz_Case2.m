function SSvalues_AggVars=SSvalues_AggVars_FHorz_Case2(StationaryDist, PolicyIndexes, SSvaluesFn, Parameters,SSvalueParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, Parallel) %pi_z,p_val
% Evaluates the aggregate value (weighted sum/integral) for each element of SSvaluesFn

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

if Parallel==2
    SSvalues_AggVars=zeros(length(SSvaluesFn),1,'gpuArray');
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);
    
    PolicyValues=PolicyInd2Val_FHorz_Case2(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]

    for i=1:length(SSvaluesFn)
        Values=nan(N_a*N_z,N_j,'gpuArray');
        for jj=1:N_j
            % Includes check for cases in which no parameters are actually required
            if isempty(SSvalueParamNames) %|| strcmp(SSvalueParamNames(i).Names(1),'')) % check for 'SSvalueParamNames={} or SSvalueParamNames={''}'
                SSvalueParamsVec=[];
            else
                SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames(i).Names,jj);
            end
            Values(:,jj)=ValuesOnSSGrid_Case2(SSvaluesFn{i}, SSvalueParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        end
%         Values=reshape(Values,[N_a*N_z,N_j]);
        SSvalues_AggVars(i)=sum(sum(Values.*StationaryDistVec));
    end
    
else
    SSvalues_AggVars=zeros(length(SSvaluesFn),1);
    d_val=zeros(l_d,1);
    a_val=zeros(l_a,1);
    z_val=zeros(l_z,1);
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);
    
    for i=1:length(SSvaluesFn)
        Values=zeros(N_a,N_z,N_j);
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
                for jj=1:N_j
                    % Includes check for cases in which no parameters are actually required
                    if isempty(SSvalueParamNames)% || strcmp(SSvalueParamNames(i).Names(1),'')) % check for 'SSvalueParamNames={}'
                        Values(j1,j2,jj)=SSvaluesFn{i}(d_val,a_val,z_val); %,p_val
                    else
                        SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames(i).Names,jj);
                        Values(j1,j2,jj)=SSvaluesFn{i}(d_val,a_val,z_val,SSvalueParamsVec); %,p_val
                    end
                end
            end
        end
        Values=reshape(Values,[N_a*N_z*N_j,1]);
        
        SSvalues_AggVars(i)=sum(Values.*StationaryDistVec);
    end

end

% 
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

