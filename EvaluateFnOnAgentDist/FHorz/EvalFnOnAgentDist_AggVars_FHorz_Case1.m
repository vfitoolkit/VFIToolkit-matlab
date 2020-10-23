function AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist,PolicyIndexes, FnsToEvaluateFn,Parameters,FnsToEvaluateFnParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,Parallel)
% Parallel is an optional input. If not given, will guess based on where StationaryDist

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

if exist('Parallel','var')==0
    if isa(StationaryDist, 'gpuArray')
        Parallel=2;
    else
        Parallel=1;
    end
end

if Parallel==2
    AggVars=zeros(length(FnsToEvaluateFn),1,'gpuArray');
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);
    
    PolicyValues=PolicyInd2Val_FHorz_Case1(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]);
    for i=1:length(FnsToEvaluateFn)
        Values=nan(N_a*N_z,N_j,'gpuArray');
        for jj=1:N_j
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateFnParamNames(i).Names)% || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateFnParamNames(i).Names,jj);
            end
            Values(:,jj)=reshape(ValuesOnSSGrid_Case1(FnsToEvaluateFn{i}, FnToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,Parallel),[N_a*N_z,1]);
        end
        %         Values=reshape(Values,[N_a*N_z,N_j]);
        AggVars(i)=sum(sum(Values.*StationaryDistVec));
    end
    
else
    AggVars=zeros(length(FnsToEvaluateFn),1);
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
    
    for i=1:length(FnsToEvaluateFn)
        Values=zeros(N_a,N_z,N_j);
        if l_d==0
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
                    for jj=1:N_j
                        [aprime_ind]=PolicyIndexes(:,j1,j2,jj);
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        % Includes check for cases in which no parameters are actually required
                        if isempty(FnsToEvaluateFnParamNames(i).Names)
                            tempv=[aprime_val,a_val,z_val];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        else
                            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateFnParamNames(i).Names,jj);
                            tempv=[aprime_val,a_val,z_val,FnToEvaluateParamsVec];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        end
                        Values(j1,j2,jj)=FnsToEvaluateFn{i}(tempcell{:});
                    end
                end
            end
        else
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
                            z_val(jj2)=z_grid(s_ind(jj2));
                        else
                            z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                        end
                    end
                    for jj=1:N_j
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
                        if isempty(FnsToEvaluateFnParamNames(i).Names)
                            tempv=[d_val,aprime_val,a_val,z_val];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        else
                            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateFnParamNames(i).Names,jj);
                            tempv=[d_val,aprime_val,a_val,z_val,FnToEvaluateParamsVec];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        end
                        Values(j1,j2,jj)=FnsToEvaluateFn{i}(tempcell{:});
                    end
                end
            end
        end
        Values=reshape(Values,[N_a*N_z*N_j,1]);
        AggVars(i)=sum(Values.*StationaryDistVec);
    end
    
end


end