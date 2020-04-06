function OtherPolicies=OtherPolicyFunctions_FHorz_Case1(PolicyIndexes, ValuesFn, ValuesFnsParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,Parallel)
% Creates OtherPolicies, which is the same thing as Policy but rather than being for the underlying policies it evaluates those in ValuesFn.
%
% Is doing essentially the same thing as EvalFnOnAgentDist_AggVars_Case1 but now it just returns the whole grid rather than taking weighted sum.
%
% Parallel is an optional input

if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
end


if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

if Parallel==2
    OtherPolicies=zeros([length(ValuesFn),N_a*N_z,N_j],'gpuArray');
        
    PolicyValues=PolicyInd2Val_FHorz_Case1(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*(l_d+l_a),N_j]);
    for i=1:length(ValuesFn)
        Values=nan(N_a*N_z,N_j,'gpuArray');
        for jj=1:N_j
            % Includes check for cases in which no parameters are actually required
            if isempty(ValuesFnsParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
                SSvalueParamsVec=[];
            else
                SSvalueParamsVec=CreateVectorFromParams(Parameters,ValuesFnsParamNames(i).Names,jj);
            end
            Values(:,jj)=ValuesOnSSGrid_Case1(ValuesFn{i}, SSvalueParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,Parallel);
        end
        OtherPolicies(i,:,:)=shiftdim(Values,-1);
    end
    
else
    OtherPolicies=zeros([length(ValuesFn),N_a,N_z,N_j]);
    aprime_val=zeros(1,l_a);
    a_val=zeros(1,l_a);
    z_val=zeros(1,l_z);
    if l_d>0
        d_val=zeros(1,l_d);
    end
    
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
    
    sizePolicyIndexes=size(PolicyIndexes);
    if  l_d==0 && l_a==1 && length(sizePolicyIndexes)~=3 % 3 is length([N_a,N_z,N_j]) % If not in vectorized form
        PolicyIndexes=reshape(PolicyIndexes,[N_a,N_z,N_j]);
    elseif length(sizePolicyIndexes(2:end))~=3 % 3 is length([N_a,N_z,N_j]) % If not in vectorized form
        PolicyIndexes=reshape(PolicyIndexes,[l_d+l_a,N_a,N_z,N_j]);
    end
    
    for i=1:length(ValuesFn)
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
%                         fprintf('Currently doing %d of %d \n',sub2ind_homemade([N_a,N_z,N_j],[j1,j2,jj]), N_a*N_z*N_j)
%                         [aprime_ind]=PolicyIndexes(j1,j2,jj);
%                         aprime_ind=ind2sub_homemade_gpu([n_a],aprime_ind);
                        aprime_ind=PolicyIndexes(:,j1,j2,jj)';
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        % Includes check for cases in which no parameters are actually required
                        if isempty(ValuesFnsParamNames(i).Names)
                            tempv=[aprime_val,a_val,z_val];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        else
                            SSvalueParamsVec=CreateVectorFromParams(Parameters,ValuesFnsParamNames(i).Names,jj);
                            tempv=[aprime_val,a_val,z_val,SSvalueParamsVec];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        end
                        Values(j1,j2,jj)=ValuesFn{i}(tempcell{:});
                    end
                end
            end
        else
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
                        fprintf('Currently doing %d of %d \n',sub2ind_homemade([N_a,N_z,N_j],[j1,j2,jj]), N_a*N_z*N_j)
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
                        if isempty(ValuesFnsParamNames(i).Names)
                            tempv=[d_val,aprime_val,a_val,z_val];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        else
                            SSvalueParamsVec=CreateVectorFromParams(Parameters,ValuesFnsParamNames(i).Names,jj);
                            tempv=[d_val,aprime_val,a_val,z_val,SSvalueParamsVec];
                            tempcell=cell(1,length(tempv));
                            for temp_c=1:length(tempv)
                                tempcell{temp_c}=tempv(temp_c);
                            end
                        end
                        Values(j1,j2,jj)=ValuesFn{i}(tempcell{:});
                    end
                end
            end
        end
        %Values=reshape(Values,[N_a*N_z*N_j,1]);
        OtherPolicies(i,:,:,:)=shiftdim(Values,-1);
    end
    
end

% Turn into appropriate shape for output
OtherPolicies=reshape(OtherPolicies,[length(ValuesFn),n_a,n_z,N_j]);

end